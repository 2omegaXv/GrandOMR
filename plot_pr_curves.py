import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = "/efm-vepfs/group-jt/intern/gzh/roboarena/project/CV"
DATA = f"{ROOT}/dynamics_yolo_dataset/data.yaml"

MODELS = {
    "DeepScoresV2": f"{ROOT}/runs_dynamics/dynamics_yolov8n/weights/best.pt",
}
COLORS = {"DeepScoresV2": "steelblue"}
STYLES = {"DeepScoresV2": "-"}

# classes to highlight
SHOW_CLASSES = ["dynamicF", "dynamicP", "dynamicS", "all"]
SHORT = {"dynamicF": "f", "dynamicP": "p", "dynamicS": "s", "all": "mean"}

curves = {}
for name, path in MODELS.items():
    model = YOLO(path)
    res = model.val(data=DATA, imgsz=1280, batch=4, device="0", verbose=False)
    box = res.box
    # p_curve: (nc, 1000) precision at each confidence
    # r_curve: (nc, 1000) recall at each confidence  (both indexed by px)
    curves[name] = {
        "px": box.px,           # confidence thresholds (1000,)
        "p":  box.p_curve,      # (nc, 1000)
        "r":  box.r_curve,      # (nc, 1000)
        "f1": box.f1_curve,     # (nc, 1000)
        "names": box.ap_class_index,  # class indices present
        "nc": res.names,        # {idx: name}
    }

# build name->idx map from first model
nc_map = {v: k for k, v in next(iter(curves.values()))["nc"].items()}

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, cls in zip(axes, SHOW_CLASSES):
    for name, c in curves.items():
        if cls == "all":
            p_vals = c["p"].mean(axis=0)
            r_vals = c["r"].mean(axis=0)
        else:
            idx = nc_map.get(cls)
            p_vals = c["p"][idx]
            r_vals = c["r"][idx]
        # sort by recall for clean PR curve
        order = np.argsort(r_vals)
        ax.plot(r_vals[order], p_vals[order], color=COLORS[name], label=name, lw=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {SHORT.get(cls, cls)}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("PR curves: base vs ft-30ep vs ft-15ep", fontsize=12)
plt.tight_layout()
out = f"{ROOT}/pr_curves_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# also print best F1 confidence per model per class
print("\nBest F1 confidence thresholds:")
for name, c in curves.items():
    print(f"\n  [{name}]")
    for cls in ["dynamicF", "dynamicP", "dynamicS"]:
        idx = nc_map.get(cls)
        best_i = np.argmax(c["f1"][idx])
        print(f"    {cls:20s}  conf={c['px'][best_i]:.2f}  F1={c['f1'][idx][best_i]:.3f}  P={c['p'][idx][best_i]:.3f}  R={c['r'][idx][best_i]:.3f}")
