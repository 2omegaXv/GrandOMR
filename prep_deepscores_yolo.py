import json
import os
import shutil
from collections import defaultdict

import yaml


def _load_categories(categories_obj):
    if isinstance(categories_obj, dict):
        return categories_obj
    return {str(idx): cat for idx, cat in enumerate(categories_obj)}


def _load_annotations(annotations_obj):
    if isinstance(annotations_obj, dict):
        return annotations_obj.items()
    return enumerate(annotations_obj)


def _get_dynamic_classes(categories):
    # deduplicate by name, keep canonical (lower numeric) id
    seen = {}
    for cat_id, cat in sorted(categories.items(), key=lambda x: int(x[0])):
        name = cat["name"]
        if name.lower().startswith("dynamic") and not name.startswith("dynamicLetter"):
            if name not in seen:
                seen[name] = cat_id
    return sorted(seen.keys())


def convert_split(json_path, images_dir, output_dir, dynamic_names, yolo_class_by_name):
    print(f"Processing {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    categories = _load_categories(data["categories"])
    category_name_by_id = {str(cat_id): cat["name"] for cat_id, cat in categories.items()}

    images = {str(img["id"]): img for img in data["images"]}
    annotations_by_image = defaultdict(list)

    for _, ann in _load_annotations(data["annotations"]):
        img_id = str(ann["img_id"])
        cat_name = None
        for cat_id in ann.get("cat_id", []):
            name = category_name_by_id.get(str(cat_id))
            if name and name in yolo_class_by_name:
                cat_name = name
                break
        if cat_name is None:
            continue
        bbox = ann.get("a_bbox")
        if not bbox or len(bbox) != 4:
            continue
        annotations_by_image[img_id].append((cat_name, bbox))

    out_imgs = os.path.join(output_dir, "images")
    out_lbls = os.path.join(output_dir, "labels")
    os.makedirs(out_imgs, exist_ok=True)
    os.makedirs(out_lbls, exist_ok=True)

    n_images = 0
    for img_id, anns in annotations_by_image.items():
        img_info = images.get(img_id)
        if not img_info:
            continue
        src = os.path.join(images_dir, img_info["filename"])
        if not os.path.exists(src):
            continue

        base = os.path.splitext(os.path.basename(img_info["filename"]))[0]
        dst_img = os.path.join(out_imgs, os.path.basename(img_info["filename"]))
        if not os.path.exists(dst_img):
            shutil.copy2(src, dst_img)

        w, h = float(img_info["width"]), float(img_info["height"])
        with open(os.path.join(out_lbls, f"{base}.txt"), "w") as f:
            for cat_name, bbox in anns:
                # a_bbox format: [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = map(float, bbox)
                bw, bh = x2 - x1, y2 - y1
                cx, cy = (x1 + bw / 2) / w, (y1 + bh / 2) / h
                f.write(f"{yolo_class_by_name[cat_name]} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}\n")
        n_images += 1

    print(f"  -> {n_images} images with dynamics annotations")
    return n_images


def prepare_dataset(root_dir, ds_dir, output_dir):
    # Determine class list from train split
    with open(os.path.join(ds_dir, "deepscores_train.json")) as f:
        data = json.load(f)
    categories = _load_categories(data["categories"])
    dynamic_names = _get_dynamic_classes(categories)
    yolo_class_by_name = {name: idx for idx, name in enumerate(dynamic_names)}

    print(f"Dynamic classes ({len(dynamic_names)}): {dynamic_names}")

    images_dir = os.path.join(ds_dir, "images")
    convert_split(
        os.path.join(ds_dir, "deepscores_train.json"),
        images_dir,
        os.path.join(output_dir, "train"),
        dynamic_names,
        yolo_class_by_name,
    )
    convert_split(
        os.path.join(ds_dir, "deepscores_test.json"),
        images_dir,
        os.path.join(output_dir, "val"),
        dynamic_names,
        yolo_class_by_name,
    )

    data_yaml = {
        "path": output_dir,
        "train": "train/images",
        "val": "val/images",
        "nc": len(dynamic_names),
        "names": dynamic_names,
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    print(f"Wrote {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    root_dir = "/efm-vepfs/group-jt/intern/gzh/roboarena/project/CV"
    prepare_dataset(
        root_dir=root_dir,
        ds_dir=os.path.join(root_dir, "ds2_dense"),
        output_dir=os.path.join(root_dir, "dynamics_yolo_dataset"),
    )
