#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


from ultralytics import YOLO
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg on prepared dataset")
    parser.add_argument("--dataset", type=Path, default=Path("baseline_recreation/dataset/dataset.yaml"))
    parser.add_argument("--weights", type=str, default="baseline_recreation/yolo11s-seg.pt", help="init weights or yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path("baseline_recreation/runs"))
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    project_path = args.project.resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    # Initialize model (supports both .pt and .yaml)
    model = YOLO(args.weights)

    # Device auto-selection that supports Apple Silicon (MPS)
    device_arg = args.device
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_arg = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_arg = "mps"
        else:
            device_arg = "cpu"

    # Train
    model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device_arg,
        project=str(project_path),
        name=args.name,
        optimizer="auto",
        weight_decay=0.0005,
        patience=100,
        exist_ok=True,
        task="segment",
    )

    # Export best weights to baseline_recreation/model.pt for convenience
    best = project_path / args.name / "weights" / "best.pt"
    if best.exists():
        target = Path("baseline_recreation/model.pt").resolve()
        target.write_bytes(best.read_bytes())
        print(f"Saved best weights to {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


