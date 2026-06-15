# Orchestral Score OMR Pipeline

扫描印刷管弦乐总谱的光学音乐识别（OMR）系统，输出多声部 MusicXML。

基于 [HOMR](https://github.com/liebharc/homr) 的 TrOMR 模型做逐谱表音符识别，在此之上构建了管弦乐总谱所需的**乐器识别、谱表分组、跨声部校正、多页合并**四层逻辑。

## 架构

```
输入图片 (.png)
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: 谱表检测与符号分割  │  HOMR 内置的语义分割 + staff 检测
│           碎片过滤            │  过滤宽度 < 20% 中位数的窄碎片
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 1.5: 乐器名识别       │  VLM API (Qwen3-VL) 主路径
│                             │  RapidOCR + 缩写字典 备用路径
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 2: 逐谱表识别         │  HOMR TrOMR transformer
│           N parts × M sys   │  按乐器数强制均分谱表
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 3: 跨声部后处理       │  Layer 0: 拍号推断（无拍号时从音符时值推算）
│                             │  Layer 1: 拍号/调号多数投票对齐
│                             │  Layer 2: 小节数统一
│                             │  Layer 3: 时值修正（position tracking）
└─────────────┬───────────────┘
              ▼
         MusicXML 输出
```

## 与原始 HOMR 的区别

| 功能 | HOMR | 本项目 |
|---|---|---|
| 乐器识别 | 无（Part 1, Part 2...） | VLM API + RapidOCR 缩写字典 |
| 谱表分组 | 几何检测（密集总谱易出错） | OCR 确定声部数 → 强制均分 |
| 跨声部校正 | 无 | 4 层后处理（拍号推断/对齐/结构/时值） |
| 多页合并 | 不支持 | 乐器并集 + divisions 归一化 + 小节拼接 |
| 鲁棒性 | 窄碎片崩溃 | 自动过滤 + 质量检查报告 |

## 安装

```bash
pip install -r requirements.txt
```

**VLM API 配置**（可选，用于乐器名识别的主路径）：

```bash
cp .env.example .env
# 编辑 .env，填入 API_KEY 和 BASE_URL
```

不配置 VLM 时，自动退回到 RapidOCR 识别乐器名。也可用 `--no-vlm` 强制使用 OCR 路径。

## 使用方法

### 单页识别

```bash
python pipeline.py score_page.png -o output.musicxml --check
```

### 多页合并

```bash
python pipeline.py page5.png page6.png page7.png -o merged.musicxml --check
```

多页模式会：
1. 逐页独立运行完整 pipeline
2. 取所有页面乐器的并集，按标准管弦乐顺序排列
3. 归一化 divisions（LCM），拼接小节，缺席声部补空拍

### 批量处理（不合并）

```bash
python pipeline.py image_directory/ --check
```

### 参数

| 参数 | 说明 |
|---|---|
| `-o, --output` | 输出路径（默认与输入同名 .musicxml） |
| `--check` | 运行质量检查（音域/时值异常报告 + piano roll 图） |
| `--no-vlm` | 禁用 VLM API，使用 RapidOCR 识别乐器名 |
| `--no-gpu` | 禁用 GPU 推理 |

## 示例输出

**单页（Mahler 7, page 5）— 17 声部，5 小节：**

![单页 piano roll](examples/page-005_pianoroll.png)

**多页合并（pages 5-7）— 22 声部，15 小节：**

![合并 piano roll](examples/page-005-007_merged_pianoroll.png)

## 已知局限

- **Tremolo 记谱**：TrOMR 无法识别 tremolo（成对黑块），产生空声部。这是模型训练数据的限制，无预处理方案。
- **VLM 稳定性**：Qwen3-VL 偶尔对同一图片返回不同结果（已用 `temperature=0` 缓解）。多谱表共享 bracket 标注时偶尔计数不准。
- **多页声部匹配**：依赖乐器名精确匹配。如果同一乐器在不同页面被 VLM 识别为不同名称，会被视为不同声部。
