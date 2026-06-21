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
│  Stage 1.5: 乐器名识别       │  VLM API (Qwen3-VL) 两阶段主路径
│             移调信息          │    Pass 1: 逐谱表自由识别（附谱表坐标+OCR标签提示）
│                             │    Pass 2: 按谱表数格式化为精确 N 行
│                             │  RapidOCR + 缩写字典 备用路径
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 2: 逐谱表识别         │  HOMR TrOMR transformer
│           N parts × M sys   │  按乐器数强制均分谱表
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 3: 跨声部后处理       │  Layer 0:  拍号推断（双小节线分段 + VLM投票确认 + 跨页传播）
│                             │  Layer 0b: 调号传播（双小节线感知 + 跨页继承与回溯修正）
│                             │  Layer 1:  拍号/调号多数投票对齐
│                             │  Layer 2:  小节数统一
│                             │  Layer 3:  时值修正（position tracking）
│                             │  + 记谱溢出修复 / 三连音标记 / 双dot清理
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 4: 力度标记检测        │  YOLOv8n 微调模型检测力度符号
│                             │  f/p/s/hairpin 等 8 类
│                             │  定位到小节，注入 <direction> 元素
└─────────────┬───────────────┘
              ▼
         MusicXML 输出
```

## 与原始 HOMR 的区别

| 功能 | HOMR | 本项目 |
|---|---|---|
| 乐器识别 | 无（Part 1, Part 2...） | 两阶段 VLM API + RapidOCR 缩写字典 |
| 移调乐器 | 不支持 | 自动检测调性，注入 `<transpose>` 元素 |
| 力度标记 | 不支持 | YOLOv8n 微调模型检测 + 小节定位注入 |
| 谱表分组 | 几何检测（密集总谱易出错） | OCR 确定声部数 → 强制均分 |
| 跨声部校正 | 无 | 多层后处理（拍号推断/对齐/结构/时值/溢出修复） |
| 多页合并 | 不支持 | 乐器并集 + divisions 归一化 + 小节拼接 |
| MIDI 音色 | 通用钢琴 | 按乐器分配正确的 instrument-sound 和 midi-program |
| 鲁棒性 | 窄碎片崩溃 | 自动过滤 + 质量检查报告 |

## 乐器与谱表识别流程

### 整体架构

```
输入图片
    │
    ▼
┌───────────────────────────────────────────────┐
│ 1. 谱表检测 (HOMR SegNet)                       │
│    → staffs_sorted: 按 y 排序的谱表列表           │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│ 2. 括号/大括号检测                                │
│    → bracket_groups: 谱表分组 [[0,1],[2],...]    │
│    → system_groups: 行划分 [[s0..s12],[s13..]]   │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│ 3. OCR 扫描左边栏 (RapidOCR)                     │
│    → ocr_hint: "Fl.", "Klar. 1.in A", ...       │
│    若 ocr_hint 为空 → 跳过 VLM，返回 []           │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│ 4. 两阶段 VLM 识别 (Qwen3-VL-235B)               │
│    Pass 1: 逐谱表自由推理                         │
│      输入: 裁切图 + 谱表坐标提示 + 每谱表最近OCR标签 │
│      输出: 每行 "Staff N (y=...): 乐器 — 理由"    │
│    Pass 2: 严格格式化                             │
│      输入: Pass 1 结果 + 裁切图 + 谱表坐标提示     │
│      输出: 精确 N 行乐器名                         │
│    验证: ≥50% 名字在 INSTRUMENT_MIDI 中           │
│    失败 → RapidOCR 备用路径                       │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│ 5. 跨页传播 & Override 匹配                       │
│    第一页结果 → detected_names                    │
│    后续页 OCR 为空 → 直接复用 detected_names       │
│    override 数 > 当前检测数 → 子序列匹配（tacet）  │
└───────────────────────────────────────────────┘
```

### OCR 预扫描（OCR-gated VLM）

`_ocr_margin_labels` 用 RapidOCR 扫描谱表左侧区域，提取文字标签并按 y 坐标排序。扫描范围限定在第一个 system 的 y 区间内，避免后续 system 的标签干扰。

**关键设计**：若 OCR 未检测到任何标签（`ocr_hint` 为空），说明该页没有乐器标注（管弦乐总谱中常见——只有第一页/第一 system 有标注），此时**跳过 VLM 调用**，直接返回空列表 `[]`。这避免了 VLM 在无标签页面上胡乱猜测乐器名。

```
页面有标签 → OCR 提取 → 传给 VLM 做精确识别
页面无标签 → OCR 返回空 → 跳过 VLM → 返回 [] → 由跨页传播机制复用第一页名字
```

### 两阶段 VLM 乐器识别

单次调用 VLM 并要求同时"识别乐器"和"输出精确 N 行"会相互干扰：VLM 倾向于把大括号内的多个谱表合并为一个条目，或者把"Oboen 1.2"展开为两行但依赖标签数字而非实际谱表数量，导致行数不符。**两阶段方法**将这两个目标解耦：

**Pass 1 — 自由推理（无行数约束）**

每个谱表的 y 坐标（全图坐标系）由 HOMR 精确测量，并附上距该 y 坐标最近的 OCR 标签，构成提示：

```
HOMR stave layout (17 staves, crop starts at full-image y=320):
  Stave 1 (y=410): nearest label 'Fl.' (dist=12px)
  Stave 2 (y=480): nearest label 'Ob.' (dist=8px)
  ...
  Stave 16 (y=1820): nearest label 'Vcll.' (dist=15px)
  Stave 17 (y=1890): nearest label 'B.' (dist=11px)
```

Pass 1 要求 VLM 对每个谱表逐行给出 `Staff N (y=...): 乐器名 — 理由`，不限行数，允许自由推理。这一步解决了"一个标签对应几个谱表"的歧义：VLM 看到 Stave 16 和 Stave 17 的最近标签都是 `Vcll.`，就会把两者都识别为 Cello，而不会只识别一个。

**Pass 2 — 严格格式化**

以 Pass 1 的输出为上下文，加上谱表坐标提示，要求输出**精确 N 行**乐器名（无解释、无编号）。两行共用同一谱表时用 `Trombone/Tuba` 合并。关键规则写入 prompt：
- `Name:Key` 格式，仅在乐谱上明确标注时才加 key（如 `Kl.(A)` → `Clarinet:A`）
- `B` 作为 key 表示 B♭；`B.`/`Kb.` 作为乐器名表示 Contrabass
- 德文缩写表：Fl./Ob./Kl./Fg./Hr./Trp./Pos./Pk. 等

#### 名称与移调分离

VLM 返回 `Name:Key` 格式，将乐器名和移调调性分开：

| VLM 返回 | 解析结果 |
|---|---|
| `Clarinet:A` | base=`Clarinet`, key=`A` |
| `Horn:F` | base=`Horn`, key=`F` |
| `Trumpet:Bb` | base=`Trumpet`, key=`Bb` |
| `Violin` | base=`Violin`, key=`None` |
| `Timpani:E` | base=`Timpani`, key=`E`（不移调，见下文） |

`_parse_instrument_key` 同时支持 `Name:Key`（VLM 格式）和 `Name in Key`（旧格式/显示格式）两种写法。

### 移调处理

#### 计算方式

移调值由调性字母**计算**得出（`_compute_transpose`），而非查表：

```python
_KEY_SEMITONES = {"C": 0, "D": 2, "Eb": 3, "E": 4, "F": 5, "G": 7, "A": 9, "Bb": 10, ...}
```

- **方向规则**：Horn/English Horn 始终向下移（如 F → chromatic=-7）；其他乐器取最近方向（semitones > 6 时向下）。
- **八度补偿**：Bass Clarinet 额外 octave-change=-1。
- **仅限已知移调乐器**：只有 `DEFAULT_TRANSPOSE_KEY` 中列出的乐器（Clarinet, Bass Clarinet, Horn, Trumpet, English Horn）才会生成 `<transpose>` 元素。即使 VLM 为 Timpani 返回了 `Timpani:E`，也不会产生错误的移调。

#### DEFAULT_TRANSPOSE_KEY（未检测到调性时的默认值）

| 乐器 | 默认调 |
|---|---|
| Clarinet | B♭ |
| Bass Clarinet | B♭ |
| Horn | F |
| Trumpet | B♭ |
| English Horn | F |

### System 检测与合并

`_detect_system_breaks` 通过谱表间的垂直间距和括号位置来划分行（system）。

后处理会合并被错误拆分的 system：当相邻的小组谱表数之和等于第一个 system 的谱表数时，自动合并。

```
检测结果: [13, 9, 4]  →  合并为: [13, 13]
                          ↑ 因为 9+4=13=第一组的大小
```

### Multi-system 页面处理

当一页包含多个 system 且谱表数不同时（如 system 1 有 5 个谱表、system 2 有 13 个），进入 multi-system 模式：

1. **System 0**：使用 OCR→VLM 检测到的 `part_names`
2. **System 1+**：调用 `_ocr_extra_system_names`，裁切左边栏图像给 VLM，在已知乐器名列表（`master_names`）中匹配
   - `master_names` 优先使用跨页传播的 `part_names_override`（完整乐器列表），而非当前页 OCR 检测到的部分名字
   - 验证时用 `_instrument_base` 比较，使 `Horn` 能匹配 `Horn:F`
3. **Override 匹配**：仅当 override 乐器数**多于**当前检测数时（说明部分乐器 tacet），才触发子序列匹配（`_match_override_to_detected`）。检测数与 override 数相等时，直接使用 VLM 检测到的名字（包含当前页的正确调性信息），不替换。

### 跨页乐器名传播

```python
# main() 中的核心逻辑
detected_names = None
for img in pages:
    _, names = run_pipeline(img, out, part_names_override=detected_names)
    if detected_names is None and names:
        detected_names = names   # 第一页的结果作为后续页的 override
```

1. **第一页**：正常 OCR→VLM 检测，结果存入 `detected_names`
2. **后续页**：
   - 若 OCR 检测到标签 → VLM 识别 → override 仅在 tacet 时介入
   - 若 OCR 无标签 → 跳过 VLM → 直接复用 `detected_names`
3. **Override 匹配算法** (`_match_override_to_detected`)：当 override 有 N 个名字但当前页只有 M 个谱表（M < N，部分乐器 tacet）时，遍历 C(N,M) 种子序列组合，按名字匹配度 + 音高范围验证选出最佳匹配，同时报告哪些乐器 tacet

### 跨页拍号/调号传播（ts_context）

`run_pipeline` 返回一个 `ts_context` 元组，供下一页调用时传入：

```python
ts_context = (last_ts, ended_with_bar, last_ks, ks_changed, pending_ks_fixup)
```

| 字段 | 含义 |
|---|---|
| `last_ts` | 本页最后生效的拍号 `(beats, beat_type)` |
| `ended_with_bar` | 本页最后一小节是否以双小节线结尾（下一页新段落开始）|
| `last_ks` | 本页末尾各声部的调号上下文 `[(name, chromatic, fifths), ...]` |
| `ks_changed` | 本页是否出现过双小节线（下一页应信任 HOMR 读取的调号）|
| `pending_ks_fixup` | 回溯修正待定项 `(file_path, from_m, to_m)` |

#### Layer 0：拍号推断（per-page）

`_cross_part_post_process` 内的 Layer 0 在每页独立处理拍号，并利用跨页上下文：

1. **双小节线分段**：用 `_double_bar_measures` 找到页内所有双小节线位置，对每段独立推断拍号。
2. **幻象小节剥离**：双小节线后内容稀疏的尾随小节（< 1/3 声部有音符）作为 TrOMR 识别末端伪影被剥离。
3. **段落内拍号**：
   - 无前页上下文或上页以双小节线结尾（`prev_bar_ended=True`）：信任 HOMR 自身的多数投票拍号。
   - 有继承上下文（`prev_ts`）：若 HOMR 分母变化或 ≥90% 置信度则认为有新拍号打印，否则继承 `prev_ts`。
4. **VLM 辅助（`_retry_tromr_post_double`）**：双小节线后 TrOMR 拍号置信度 ≤75% 时，裁取双小节线右侧区域，向 VLM 采集最多 5 个谱表的投票：
   - VLM 读出数字拍号 → 若与前段不同则注入；若相同则确认并覆写任何错误读数。
   - VLM 判断无拍号符号 → 继承前段拍号，覆盖 TrOMR 的错误读数。
   - VLM 出错 → 退回 TrOMR 重跑后双小节线区域。

#### Layer 0b：调号传播（per-page）

调号传播依赖同一 `ts_context` 中的 `last_ks` / `ks_changed`：

- **无双小节线页（`prev_ks_changed=False`）**：继承上页/system 末尾的调号，覆盖 HOMR 在无打印调号时的乱读。
- **有双小节线页（`prev_ks_changed=True`）**：信任 HOMR 在新段落开头读取的调号（TrOMR 能正确读出段落首的调号符号）。
- **回溯修正（`ks_fixup_range`）**：若双小节线出现在页面**中间**，双小节线之后到页末的小节带有旧调号。等下一页处理完成、正确调号已知后，`_fixup_ks_in_file` 对之前写入磁盘的文件做回溯修正。

```
页 N 处理完 → pending_ks_fixup = (page_N.musicxml, m_dbl+1, m_end)
页 N+1 处理完 → prev_ks 即为新调号
              → _fixup_ks_in_file(page_N.musicxml, prev_ks, m_dbl+1, m_end)
```

#### 合并阶段的拍号（merge_pages）

`merge_pages` 中的 `page_measure_ts` 为每个小节位置确定拍号，规则如下：

1. **仅统计显式标记**：只有包含 `<time>` 元素的小节，以及之后拥有实音符（非全休止）的小节才参与投票。全休止符在 4/4 和 6/4 下外观完全相同，无法区分，不参与投票。
2. **多数投票**：同一小节位置各声部拍号不同时，取票数最多的那个。
3. **跨页传播**：从前一页末尾拍号继承，避免无标记小节被默认为 4/4。

这解决了多 system 页面（如 system 1 = 6/4，system 2 新乐器入场无标记）中新乐器的全休止小节被错误地判断为 4/4 的问题。

### 显示名称

`_display_name` 将内部格式转为 MusicXML 显示名：`Clarinet:A` → `Clarinet in A`。用于 `<part-name>` 和 `<instrument-name>` 元素。

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

### PDF 一键转换（推荐）

`run_score.py` 是端到端的入口脚本，从 PDF 到最终合并的 MusicXML 一条龙完成：

```bash
# 基本用法：PDF + 起止页号（1-based，闭区间）
python run_score.py Bruckner7.pdf 1 3

# 自定义输出路径
python run_score.py Tchai1.pdf 3 5 -o tchai_mvt1.musicxml

# 单页
python run_score.py Brahms4.pdf 93 93

# 更高分辨率（默认 300 DPI）
python run_score.py score.pdf 1 10 --dpi 400
```

默认输出到 `outputs/{pdf名}_{起始页}_{结束页}.musicxml`，中间产物（逐页 PNG 和 MusicXML）存在 `outputs/{pdf名}/` 下。

内部流程：
1. `pdf2image` 将指定页渲染为 PNG（存于 `outputs/{stem}/`）
2. 逐页调用 `run_pipeline`，自动跨页传播**乐器名、拍号与调号**（`ts_context`）
3. 多页时调用 `merge_pages` 合成最终 MusicXML

| 参数 | 说明 |
|---|---|
| `pdf` | PDF 文件路径 |
| `start` | 起始页号（1-based，包含） |
| `end` | 结束页号（1-based，包含，等于 start 时只转一页） |
| `-o, --output` | 输出 .musicxml 路径（默认 `outputs/{stem}_{start}_{end}.musicxml`） |
| `--dpi` | PDF 渲染 DPI（默认 300） |
| `--no-gpu` | 禁用 GPU 推理 |
| `--no-vlm` | 禁用 VLM，使用 RapidOCR 识别乐器名 |

### 底层接口：pipeline.py

如果已有裁好的 PNG 页面图片，可以直接调用 `pipeline.py`：

```bash
# 单页识别
python pipeline.py score_page.png -o output.musicxml --check

# 多页合并（自动跨页传播乐器名）
python pipeline.py page5.png page6.png page7.png -o merged.musicxml --check

# 批量处理目录（不合并）
python pipeline.py image_directory/ --check
```

### 编程接口

```python
from pipeline import run_pipeline, merge_pages

# 单页，指定乐器名（跳过 VLM/OCR 识别）
out_path, part_names, ts_ctx = run_pipeline(
    "page.png", "out.musicxml",
    part_names_override=["Flute", "Clarinet:A", "Horn:F", "Violin", ...]
)

# 多页跨页传播（拍号/调号/乐器名全部跨页传播）
detected_names = None
ts_context = None
for img, out in zip(png_paths, xml_paths):
    _, names, ts_context = run_pipeline(
        img, out,
        part_names_override=detected_names,
        ts_context=ts_context,
    )
    if detected_names is None and names:
        detected_names = names
merge_pages(xml_paths, "merged.musicxml")
```

`run_pipeline` 返回 `(output_path, part_names, ts_context)` 三元组。`ts_context` 应原样传给下一页的 `run_pipeline` 调用，以实现拍号/调号的跨页传播（详见[跨页拍号/调号传播](#跨页拍号调号传播ts_context)）。

pipeline.py 参数：

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

## Stage 4: 力度标记检测

### 模型

YOLOv8n 在 DeepScoresV2 dense 子集上微调 15 epoch（736张训练图，160张测试图）。

- 基础权重：`yolo26n.pt`（DeepScoresV2 全类预训练）
- 微调权重：`runs_dynamics/dynamics_finetune_15ep/weights/best.pt`
- 检测类别（8类）：`dynamicCrescendoHairpin`, `dynamicDiminuendoHairpin`, `dynamicF`, `dynamicM`, `dynamicP`, `dynamicR`, `dynamicS`, `dynamicZ`

15 epoch 优于 30 epoch，原因是 YOLO 在 epoch 21 关闭 mosaic 增强后出现分布偏移，30ep 轻微过拟合。

| | 无微调 | 微调30ep | 微调15ep |
|--|--------|---------|---------|
| mAP50 all | 0.812 | 0.809 | **0.894** |
| mAP50 f | 0.943 | 0.888 | **0.956** |
| mAP50 p | 0.898 | 0.972 | **0.980** |
| mAP50 s | 0.595 | 0.566 | **0.745** |

### 置信度阈值设计

**f、p 类**优先保证精度（漏报比误报代价低），阈值取 P=0.95 对应的最低置信度。  
**s 类**（`dynamicS`，即 sforzando）只在与 f 或 p 共同出现时才有音乐意义；单独出现的 s 检测无效，因此优先保召回，取 F1 最优点。

| 类别 | 置信度阈值 | Precision | Recall | 策略 |
|------|-----------|-----------|--------|------|
| dynamicF | **0.65** | ~0.952 | ~0.875 | precision-first (conf at P=0.95 → rounded up) |
| dynamicP | **0.65** | ~0.952 | ~0.875 | precision-first |
| dynamicS | **0.31** | 0.877 | 0.906 | recall-first (F1最优) |
| 其余类别 | 0.25 | — | — | 默认 |

### 力度标记后处理

YOLO 检测到的原始标签（`dynamicF`、`dynamicP`、`dynamicS` 等字符）经过规则化后写入 MusicXML：

- `ff`/`sf`/`mf` → `f`；`pp`/`mp` → `p`（简化为标准两级）
- 三个及以上字母（`fff`、`ppp` 等）丢弃
- `fp`/`pf` 丢弃（复合力度超出当前模型能力）
- Hairpin 直接对应 `<wedge type="crescendo/diminuendo">`

## 已知局限

- **Tremolo 记谱**：TrOMR 无法识别 tremolo（成对黑块），产生空声部。这是模型训练数据的限制。
- **多页声部匹配**：依赖乐器名精确匹配。如果同一乐器在不同页面被 VLM 识别为不同名称（如移调标记变化），会被视为不同声部。可通过 `part_names_override` 手动指定乐器名绕过。
- **VLM 调用开销**：两阶段各调用一次 API，识别耗时约为单阶段的两倍。若需加速，可用 `--no-vlm` 切换到 RapidOCR 路径（精度稍低）。
