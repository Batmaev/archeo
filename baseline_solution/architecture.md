## Model overview

- **Family**: Ultralytics YOLO11s-seg (instance segmentation)
- **Layers / Params / FLOPs**: 203 layers, 10,086,158 params, ~33.1 GFLOPs
- **Input channels**: 3
- **Scale**: s (yaml: `yolo11s-seg.yaml`)
- **Task**: segment

### Classes (nc=10)
- gorodishcha, fortifikatsii, arkhitektury, selishcha, kurgany, dorogi, yamy, pashni, mezha, inoe

## Architecture details

- **Backbone**:
  - Conv → Conv → C3k2 → Conv → C3k2 → Conv → C3k2 (with shortcut) → Conv → C3k2 (with shortcut) → SPPF → C2PSA
- **Head**:
  - Upsample → Concat(skip) → C3k2 → Upsample → Concat(skip) → C3k2 →
    Downsample(Conv) → Concat(skip) → C3k2 → Downsample(Conv) → Concat(skip) → C3k2 →
    Segment head with 32 prototypes and 256 channels

## Training setup (from checkpoint train_args)

- **Dataset config**: `dataset/dataset.yaml`
- **Epochs**: 50
- **Batch size**: 56
- **Image size**: 1024
- **Optimizer**: auto
- **LR**: lr0=0.01, lrf=0.01
- **Momentum**: 0.937
- **Weight decay**: 0.0005
- **Early stop patience**: 100
- **Devices**: 0,1
- **Project/Name**: `y11s_1024_50_128/` / `train`
- **Checkpoint date**: 2025-10-24T21:10:40.566421

## Metrics (from checkpoint)

- Boxes (B):
  - precision: 0.8776, recall: 0.7763, mAP50: 0.8453, mAP50-95: 0.6496
- Masks (M):
  - precision: 0.8794, recall: 0.7400, mAP50: 0.8163, mAP50-95: 0.5790
- Validation losses:
  - val/box_loss: 1.0891, val/seg_loss: 1.8889, val/cls_loss: 1.0730, val/dfl_loss: 1.0367
- Fitness: 1.2286

Note: `train_results` present with per-epoch curves (epochs tracked up to ~300 in the log), showing steady loss decrease and metric convergence; see checkpoint for full arrays.

## Inference usage (project pipeline)

- Weights path: `baseline_solution/model.pt`
- Tiled inference: 1024×1024 tiles with 30% overlap
- Post-processing: mask→polygon conversion, CRS transform to EPSG:3857, dedup (IoU 0.95), merge with buffer 0.5 px

