#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import os


from ultralytics import YOLO
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg on prepared dataset")
    parser.add_argument("--weights", type=str, default="baseline_recreation/yolo11s-seg.pt", help="Initial weights or model yaml. If resuming, provide baseline_recreation/train/weights/last.pt")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input size that Ultralytics resizes to on-the-fly; source tiles are typically 1024x1024; using a smaller value speeds training/inference at some quality cost")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset sampled each epoch [0,1]")
    parser.add_argument("--no-val", action="store_true", help="Disable external evaluation (internal validation stays enabled)")
    parser.add_argument("--metric", type=str, choices=["qual", "final"], default="qual", help="External evaluation metric to compute and report")
    parser.add_argument("--ext-val-every", type=int, default=3, help="Раз в сколько эпох запускать официальную метрику на валидационных данных")
    args = parser.parse_args()

    # Resolve absolute project paths from this file location
    root = Path(__file__).resolve().parent.parent
    dataset_path = (root / "baseline_recreation/dataset/dataset.yaml").resolve()
    project_path = (root / "baseline_recreation/runs").resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    # Initialize model (supports both .pt and .yaml)
    model = YOLO(args.weights)

    # Device auto-selection that supports Apple Silicon (MPS)
    if torch.cuda.is_available():
        device_arg = "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_arg = "mps"
    else:
        device_arg = "cpu"

    # Prepare external eval env once (symlinks + GT) and register callback
    ext_eval_enabled = not bool(args.no_val)
    # Ensure sane frequency (>=1)
    try:
        ext_eval_every = int(args.ext_val_every)
    except Exception:
        ext_eval_every = 3
    if ext_eval_every < 1:
        ext_eval_every = 1
    ext_eval_tmp_root: Path | None = None
    ext_eval_gt_path: Path | None = None

    def _prepare_ext_eval_env(save_dir: Path) -> tuple[Path, Path]:
        val_list_path = (root / "splits" / "val_sites.txt").resolve()
        val_names = [line.strip() for line in val_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tmp_root = save_dir / "ext_eval" / "val_input"
        tmp_root.mkdir(parents=True, exist_ok=True)
        for name in val_names:
            src = (root / "train" / name).resolve()
            dst = tmp_root / name
            if not src.exists():
                raise FileNotFoundError(f"Validation region not found: {src}")
            if dst.is_symlink():
                # ensure target is correct
                current = Path(os.path.realpath(dst))
                if current != src:
                    dst.unlink()
                    os.symlink(src, dst)
                continue
            if dst.exists():
                # exists but not a symlink → avoid clobbering
                raise FileExistsError(f"Destination exists and is not a symlink: {dst}")
            os.symlink(src, dst)
        gt_path = save_dir / "ext_eval" / "val_gt.geojson"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python3", str((root / "metrics/merge_ground_truth.py").resolve()),
            "--input-root", str(tmp_root.resolve()),
            "--output", str(gt_path.resolve()),
        ], check=True)
        if not gt_path.exists():
            raise RuntimeError("Failed to create val_gt.geojson")
        return tmp_root, gt_path

    def _external_eval_callback(trainer):
        if not ext_eval_enabled:
            return
        epoch = int(getattr(trainer, "epoch", -1))
        # run only on epochs matching the requested frequency
        if epoch >= 0 and ((epoch + 1) % ext_eval_every) != 0:
            return
        save_dir = Path(trainer.save_dir)
        weights_last = save_dir / "weights" / "last.pt"
        if not weights_last.exists():
            return
        out_dir = save_dir / "ext_eval" / f"epoch_{epoch+1}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Log solution output to file to avoid mixing with trainer logs
        with (out_dir / "solution.log").open("w", encoding="utf-8") as logf:
            cmd = [
                "python3", str((root / "baseline_solution/solution.py").resolve()),
                str(ext_eval_tmp_root.resolve()), str(out_dir.resolve()),
                "--model", str(weights_last.resolve())
            ]
            # Limit threads to avoid starving the trainer / triggering OOM on macOS
            env_vars = os.environ.copy()
            env_vars.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            run_kwargs = {"stdout": logf, "stderr": subprocess.STDOUT, "check": True, "env": env_vars}
            try:
                subprocess.run(cmd, **run_kwargs)
            except subprocess.TimeoutExpired:
                (out_dir / "metric.txt").write_text("solution_timeout\n", encoding="utf-8")
                print(f"[ext-eval][epoch {epoch+1}] solution.py timed out")
                return
            except subprocess.CalledProcessError as e:
                (out_dir / "metric.txt").write_text(f"solution_failed rc={e.returncode}\n", encoding="utf-8")
                print(f"[ext-eval][epoch {epoch+1}] solution.py failed rc={e.returncode}")
                return
        pred_geojson = out_dir / "result.geojson"
        metric_script = (root / ("metrics/compute_metrics_qual.py" if args.metric == "qual" else "metrics/compute_metrics.py")).resolve()
        try:
            res = subprocess.run([
                "python3", str(metric_script),
                "--predictions", str(pred_geojson.resolve()),
                "--ground-truth", str(ext_eval_gt_path.resolve()),
            ], capture_output=True, text=True, check=True, env=env_vars)
            (out_dir / "metric.txt").write_text(f"{args.metric}={res.stdout.strip()}\n", encoding="utf-8")
            print(f"[ext-eval][epoch {epoch+1}] {args.metric} = {res.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            (out_dir / "metric.txt").write_text(f"metric_failed rc={e.returncode}\n", encoding="utf-8")
            print(f"[ext-eval][epoch {epoch+1}] metric failed rc={e.returncode}")
            # keep training

    if ext_eval_enabled:
        # Build env now; if fails, disable eval
        save_dir_hint = project_path / "train"
        save_dir_hint.mkdir(parents=True, exist_ok=True)
        try:
            ext_eval_tmp_root, ext_eval_gt_path = _prepare_ext_eval_env(save_dir_hint)
            model.add_callback("on_fit_epoch_end", _external_eval_callback)
            print(f"[ext-eval] Prepared at {save_dir_hint / 'ext_eval'}; GT: {ext_eval_gt_path}; every {ext_eval_every} epoch(s)")
        except Exception as e:
            ext_eval_enabled = False
            print(f"[ext-eval] Disabled: {e}")

    # Train
    model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=device_arg,
        project=str(project_path),
        name="train",
        optimizer="auto",
        weight_decay=0.0005,
        patience=100,
        exist_ok=True,
        task="segment",
        fraction=float(args.fraction),
        val=True,
    )

    # Export best weights to baseline_recreation/model.pt for convenience
    best = project_path / "train" / "weights" / "best.pt"
    if best.exists():
        target = (root / "baseline_recreation/model.pt").resolve()
        target.write_bytes(best.read_bytes())
        print(f"Saved best weights to {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


