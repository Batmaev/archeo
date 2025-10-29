#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import random

import cv2
import numpy as np
import rasterio
from rasterio import windows
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape, box
from shapely.validation import make_valid
from PIL import Image


# Classes order must match the baseline solution
CLASS_NAMES: List[str] = [
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
    "selishcha",
    "kurgany",
    "dorogi",
    "yamy",
    "pashni",
    "mezha",
    "inoe",
]


# Mapping (subset) from Russian names (by filename suffix) to canonical class names
RUS_TO_CLASS: Dict[str, Optional[str]] = {
    "городища": "gorodishcha",
    "фортификации": "fortifikatsii",
    "архитектура": "arkhitektury",
    "архитектуры": "arkhitektury",
    "селища": "selishcha",
    "курганы": "kurgany",
    "дороги": "dorogi",
    "ямы": "yamy",
    "пашня": "pashni",
    "пашни": "pashni",
    "межа": "mezha",
    "иное": "inoe",
}


def prepare_image_for_png(data: np.ndarray) -> np.ndarray:
    """Convert rasterio (C,H,W) or (H,W) array to (H,W,3) uint8 for saving.

    Follows the same normalization approach as baseline_solution.prepare_image.
    """
    if data.ndim == 3 and data.shape[0] <= 4:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        max_val, min_val = float(np.max(data)), float(np.min(data))
        if max_val > min_val:
            data = (255.0 * (data - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    if data.shape[-1] > 3:
        data = data[:, :, :3]
    return data


def yolo_seg_line(class_id: int, polygon_xy: np.ndarray, width: int, height: int) -> str:
    """Build a YOLO-seg label line for a single polygon.

    polygon_xy: array of shape (N, 2) with pixel coordinates (x, y).
    """
    if polygon_xy.shape[0] < 3:
        return ""
    xs = polygon_xy[:, 0].astype(np.float32) / float(max(width, 1))
    ys = polygon_xy[:, 1].astype(np.float32) / float(max(height, 1))
    coords: List[str] = []
    for x, y in zip(xs, ys):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return f"{class_id} " + " ".join(coords)


def contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """Extract outer polygon contours from a binary mask (H,W) -> list of (N,2) float arrays."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=0.5, closed=True)
        if len(approx) < 3:
            continue
        xy = approx.reshape(-1, 2).astype(np.float32)
        polys.append(xy)
    return polys


def _class_id(name: str) -> Optional[int]:
    try:
        return CLASS_NAMES.index(name)
    except ValueError:
        return None


def _infer_class_from_filename(path: Path) -> Optional[str]:
    stem = path.stem
    parts = stem.split("_")
    rus = parts[-1] if parts else stem
    return RUS_TO_CLASS.get(rus, rus)


@dataclass
class TileSpec:
    height: int
    width: int
    stride_y: int
    stride_x: int


def tile_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = [0]
    while pos[-1] + tile < length:
        nxt = pos[-1] + stride
        if nxt + tile >= length:
            pos.append(max(length - tile, 0))
            break
        pos.append(nxt)
    return sorted(dict.fromkeys(max(0, p) for p in pos))


def collect_region_annotations(region_dir: Path) -> Dict[str, List[Polygon]]:
    ann_root = next(iter(sorted(region_dir.glob("06_*_разметка"))), None)
    collected: Dict[str, List[Polygon]] = {name: [] for name in CLASS_NAMES}
    if ann_root is None or not ann_root.exists():
        return collected
    for gj in ann_root.rglob("*.geojson"):
        cls = _infer_class_from_filename(gj)
        if cls is None:
            continue
        if cls not in collected:
            # skip unknown classes
            continue
        try:
            with gj.open("r", encoding="utf-8") as f:
                fc = json.load(f)
        except Exception:
            continue
        for feat in fc.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                poly = make_valid(shape(geom))
                if poly.is_empty:
                    continue
                if poly.geom_type == "Polygon":
                    collected[cls].append(Polygon(poly.exterior.coords))
                elif poly.geom_type == "MultiPolygon":
                    for sub in poly.geoms:
                        if not sub.is_empty:
                            collected[cls].append(Polygon(sub.exterior.coords))
            except Exception:
                continue
    return collected


def save_image_and_labels(
    image_arr: np.ndarray,
    label_lines: List[str],
    out_images_dir: Path,
    out_labels_dir: Path,
    basename: str,
    image_format: str,
    quality: int,
) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    fmt = image_format.lower()
    ext = {
        "jpeg": "jpg",
        "jpg": "jpg",
        "png": "png",
        "webp": "webp",
    }.get(fmt, "png")
    image_path = out_images_dir / f"{basename}.{ext}"
    label_path = out_labels_dir / f"{basename}.txt"
    img = Image.fromarray(image_arr)
    save_kwargs: Dict[str, object] = {}
    target_format = fmt
    if fmt in ("jpg", "jpeg"):
        target_format = "JPEG"
        save_kwargs = {"quality": int(quality), "subsampling": 0, "optimize": True}
    elif fmt == "png":
        target_format = "PNG"
        save_kwargs = {"optimize": True}
    elif fmt == "webp":
        target_format = "WEBP"
        save_kwargs = {"quality": int(quality)}
    try:
        img.save(str(image_path), format=target_format, **save_kwargs)
    except Exception:
        # Generic fallback to PNG
        fallback_path = out_images_dir / f"{basename}.png"
        img.save(str(fallback_path), format="PNG", optimize=True)
    with label_path.open("w", encoding="utf-8") as f:
        for line in label_lines:
            if line:
                f.write(line + "\n")


def process_single_raster(
    tif_path: Path,
    region_name: str,
    class_to_polys: Dict[str, List[Polygon]],
    split: str,
    out_root: Path,
    tile_spec: TileSpec,
    min_poly_area_px: float,
    empty_fraction: float,
    rng: random.Random,
    image_format: str,
    quality: int,
    max_nodata_frac: float,
    min_annot_valid_frac: float,
) -> int:
    saved = 0
    with rasterio.open(tif_path) as src:
        h = src.height
        w = src.width
        transform = src.transform

        ys = tile_positions(h, tile_spec.height, tile_spec.stride_y)
        xs = tile_positions(w, tile_spec.width, tile_spec.stride_x)

        # Precompute image windows list to avoid reopening
        for y0 in ys:
            y1 = min(y0 + tile_spec.height, h)
            for x0 in xs:
                x1 = min(x0 + tile_spec.width, w)
                win = windows.Window.from_slices((y0, y1), (x0, x1))
                # Read image window
                data = src.read(window=win)
                img = prepare_image_for_png(data)

                # Compute nodata fraction using raster mask if available, else fallback
                try:
                    mask_arr = src.read_masks(1, window=win)
                except Exception:
                    mask_arr = None
                if mask_arr is not None:
                    valid_ratio = float((mask_arr > 0).mean())
                    nodata_frac = max(0.0, min(1.0, 1.0 - valid_ratio))
                else:
                    # fallback: treat pixels all-zero across bands as nodata
                    try:
                        nonzero_any = (data != 0).any(axis=0)
                        nodata_frac = max(0.0, min(1.0, 1.0 - float(nonzero_any.mean())))
                    except Exception:
                        nodata_frac = 0.0

                # Build window polygon in image CRS
                window_poly = box(*windows.bounds(win, transform))

                label_lines: List[str] = []
                any_mask = False
                combined_annot_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                for cls_name in CLASS_NAMES:
                    cls_id = _class_id(cls_name)
                    if cls_id is None:
                        continue

                    # Filter polygons intersecting this window
                    polys = [p for p in class_to_polys.get(cls_name, []) if p.intersects(window_poly)]
                    if not polys:
                        continue

                    # Rasterize polygons into the window
                    try:
                        mask = rasterize(
                            [(p, 1) for p in polys],
                            out_shape=(y1 - y0, x1 - x0),
                            transform=windows.transform(win, transform),
                            fill=0,
                            all_touched=False,
                            dtype=np.uint8,
                        )
                    except Exception:
                        continue

                    if np.count_nonzero(mask) == 0:
                        continue

                    any_mask = True
                    combined_annot_mask |= (mask > 0).astype(np.uint8)
                    # Extract contours per class
                    contours = contours_from_mask(mask)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < float(min_poly_area_px):
                            continue
                        line = yolo_seg_line(cls_id, cnt, width=img.shape[1], height=img.shape[0])
                        if line:
                            label_lines.append(line)

                if not any_mask:
                    # Drop heavy-nodata empty tiles entirely
                    if nodata_frac >= float(max_nodata_frac):
                        continue
                    # Keep a fraction of remaining empty tiles to reduce bias
                    if empty_fraction <= 0.0 or rng.random() >= float(empty_fraction):
                        continue
                else:
                    # Positive tile: ensure annotation mostly lies on valid data
                    # Build valid mask
                    if mask_arr is not None:
                        valid_mask = (mask_arr > 0)
                    else:
                        try:
                            valid_mask = (data != 0).any(axis=0)
                        except Exception:
                            valid_mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
                    annot_pixels = int(combined_annot_mask.sum())
                    if annot_pixels > 0:
                        valid_annot_pixels = int((combined_annot_mask.astype(bool) & valid_mask).sum())
                        annot_valid_ratio = valid_annot_pixels / float(annot_pixels)
                        if annot_valid_ratio < float(min_annot_valid_frac):
                            # Likely misaligned annotation over nodata → drop
                            continue

                basename = f"{region_name}__{tif_path.stem}__y{y0}_x{x0}"
                out_images = out_root / "images" / split
                out_labels = out_root / "labels" / split
                save_image_and_labels(
                    img,
                    label_lines,
                    out_images,
                    out_labels,
                    basename,
                    image_format=image_format,
                    quality=quality,
                )
                saved += 1
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO-seg dataset from GeoJSON annotations")
    parser.add_argument("--train-root", type=Path, default=Path("train"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--out-root", type=Path, default=Path("baseline_recreation/dataset"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--min-poly-area", type=float, default=10.0, help="min polygon area in pixels")
    parser.add_argument("--empty-fraction", type=float, default=0.1, help="fraction of empty tiles to keep [0..1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-format", type=str, default="webp", choices=["png", "jpg", "jpeg", "webp"])
    parser.add_argument("--quality", type=int, default=90, help="quality for lossy formats (JPEG/WEBP)")
    parser.add_argument("--max-nodata-frac", type=float, default=0.8, help="skip empty tiles with nodata fraction above this")
    parser.add_argument("--min-annot-valid-frac", type=float, default=0.5, help="for positive tiles, minimal fraction of annotation over valid data")

    args = parser.parse_args()

    train_root: Path = args.train_root
    splits_dir: Path = args.splits_dir
    out_root: Path = args.out_root
    tile_size: int = args.tile_size
    overlap: float = args.overlap
    min_poly_area: float = args.min_poly_area
    empty_fraction: float = max(0.0, min(1.0, float(args.empty_fraction)))
    rng = random.Random(int(args.seed))
    image_format: str = args.image_format
    quality: int = int(args.quality)
    max_nodata_frac: float = max(0.0, min(1.0, float(args.max_nodata_frac)))
    min_annot_valid_frac: float = max(0.0, min(1.0, float(args.min_annot_valid_frac)))

    train_sites_path = splits_dir / "train_sites.txt"
    val_sites_path = splits_dir / "val_sites.txt"

    train_regions: List[str] = []
    val_regions: List[str] = []
    if train_sites_path.exists():
        train_regions = [line.strip() for line in train_sites_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if val_sites_path.exists():
        val_regions = [line.strip() for line in val_sites_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # stride inferred from overlap
    stride = max(int(round(tile_size * (1.0 - float(overlap)))), 1)
    spec = TileSpec(height=tile_size, width=tile_size, stride_y=stride, stride_x=stride)

    total_saved = {"train": 0, "val": 0}

    region_dirs = [p for p in sorted(train_root.glob("*_FINAL")) if p.is_dir()]
    for region in region_dirs:
        region_name = region.name
        split = "train"
        if val_regions and region_name in val_regions:
            split = "val"
        elif train_regions and region_name not in train_regions:
            # if train list provided and region not in it, skip unless in val
            if region_name not in val_regions:
                continue

        class_to_polys = collect_region_annotations(region)

        # Collect LiDAR rasters
        tif_paths: List[Path] = []
        for tif in region.rglob("*.tif"):
            if "lidar" in tif.stem.lower():
                tif_paths.append(tif)
        tif_paths = sorted(tif_paths)
        if not tif_paths:
            continue

        for tif_path in tif_paths:
            saved = process_single_raster(
                tif_path=tif_path,
                region_name=region_name,
                class_to_polys=class_to_polys,
                split=split,
                out_root=out_root,
                tile_spec=spec,
                min_poly_area_px=min_poly_area,
                empty_fraction=empty_fraction,
                rng=rng,
                image_format=image_format,
                quality=quality,
                max_nodata_frac=max_nodata_frac,
                min_annot_valid_frac=min_annot_valid_frac,
            )
            total_saved[split] += saved

    # Write dataset.yaml
    names_yaml = "\n".join([f"  - {n}" for n in CLASS_NAMES])
    dataset_yaml = (
        f"path: {out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_yaml}\n"
    )
    (out_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    print(json.dumps({"saved_train": total_saved["train"], "saved_val": total_saved["val"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


