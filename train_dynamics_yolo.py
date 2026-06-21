import os
import argparse
from ultralytics import YOLO

ROOT = "/efm-vepfs/group-jt/intern/gzh/roboarena/project/CV"


def train(args):
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.join(ROOT, "runs_dynamics"),
        name=args.name,
        device=args.device,
        workers=4,
        exist_ok=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(ROOT, "yolo26n.pt"))
    parser.add_argument("--data", default=os.path.join(ROOT, "dynamics_yolo_dataset", "data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="dynamics_yolov8n")
    args = parser.parse_args()
    train(args)
