#!/usr/bin/env python3
"""
Упрощенная версия baseline_solution/solution.py
(чтобы уменьшить вероятность падения)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple tiled inference → GeoJSON (EPSG:3857)")
    p.add_argument("input_dir", type=Path, help="Root with *_FINAL regions or a single *_FINAL directory")
    p.add_argument("output_dir", type=Path, help="Directory to write result.geojson")
    p.add_argument("--model", required=True, type=str, help="Path to YOLO segmentation weights (.pt)")
    p.add_argument("--tile", type=int, default=1024, help="Tile size in pixels for inference (e.g., 1024)")
    p.add_argument("--overlap", type=float, default=0.3, help="Tile overlap ratio [0..1], e.g., 0.3 = 30%")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold to keep detections")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS during prediction")
    p.add_argument("--min-valid-tile", type=float, default=0.25, help="skip tiles with valid fraction below this")
    p.add_argument("--min-poly-valid", type=float, default=0.6, help="drop polygons with coverage below this")
    p.add_argument("--max-regions", type=int, default=0, help="Limit number of regions to process (0 = all)")
    p.add_argument("--downscale", type=float, default=1.0, help="Downscale factor for tiles before prediction (1.0 = no scaling)")
    return p.parse_args()


def prepare_image_uint8_rgb(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3 and data.shape[0] <= 4:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        vmax, vmin = float(np.max(data)), float(np.min(data))
        if vmax > vmin:
            data = (255.0 * (data - vmin) / (vmax - vmin)).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    if data.shape[-1] > 3:
        data = data[:, :, :3]
    return data


def tile_positions(length: int, tile: int, step: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = [0]
    while pos[-1] + tile < length:
        nxt = pos[-1] + step
        if nxt + tile >= length:
            pos.append(max(length - tile, 0))
            break
        pos.append(nxt)
    return sorted(dict.fromkeys(max(0, p) for p in pos))


def mask_to_polygons(mask_bool: np.ndarray) -> Iterable[Polygon]:
    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if len(c) < 3:
            continue
        pts = [(float(pt[0][0]), float(pt[0][1])) for pt in c]
        try:
            poly = make_valid(Polygon(pts))
            if not poly.is_empty and poly.area >= 1.0:
                yield poly
        except Exception:
            continue


def pixelpoly_to_epsg3857(poly: Polygon, transform: Affine, src_crs_wkt: str) -> Optional[Polygon]:
    try:
        # pixel → georef using affine
        def pix_to_geo_xy(x: float, y: float) -> Tuple[float, float]:
            X = transform.a * x + transform.b * y + transform.c
            Y = transform.d * x + transform.e * y + transform.f
            return X, Y

        poly_geo = shapely_transform(lambda x, y: (transform.a * x + transform.b * y + transform.c,
                                                   transform.d * x + transform.e * y + transform.f), poly)
        # georef → EPSG:3857
        src = src_crs_wkt if src_crs_wkt else "EPSG:32636"
        tr = Transformer.from_crs(src, "EPSG:3857", always_xy=True)
        poly_3857 = shapely_transform(tr.transform, poly_geo)
        poly_3857 = make_valid(poly_3857)
        if poly_3857.is_empty:
            return None
        return poly_3857
    except Exception:
        return None


def process_region(model: YOLO, names: Dict[int, str], region_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    feats: List[Dict[str, Any]] = []
    tifs = [p for p in region_dir.rglob("*.tif") if "lidar" in p.stem.lower() or "li" in p.stem.lower()]
    for tif in sorted(tifs):
        try:
            with rasterio.open(tif) as src:
                data = src.read()  # C,H,W
                transform = src.transform
                crs_text = src.crs.to_string() if src.crs else "EPSG:32636"
                try:
                    valid_full = (src.read_masks(1) > 0)
                except Exception:
                    # Fallback: any non-zero across bands
                    valid_full = (data != 0).any(axis=0)
            image = prepare_image_uint8_rgb(data)
            valid_resized = valid_full
            h, w = image.shape[:2]
            tile = int(args.tile)
            step = max(int(round(tile * (1.0 - float(args.overlap)))), 1)
            ys = tile_positions(h, tile, step)
            xs = tile_positions(w, tile, step)

            for y in ys:
                y2 = min(y + tile, h)
                for x in xs:
                    x2 = min(x + tile, w)
                    tile_img = image[y:y2, x:x2]
                    if tile_img.size == 0:
                        continue
                    # Skip tile dominated by nodata
                    valid_ratio = float(valid_resized[y:y2, x:x2].mean()) if (y2>y and x2>x) else 0.0
                    if valid_ratio < float(args.min_valid_tile):
                        continue
                    # Optional downscale before prediction
                    dh = y2 - y
                    dw = x2 - x
                    scale = max(1e-6, float(args.downscale))
                    if scale < 1.0:
                        dwh = max(1, int(round(dh * scale)))
                        dww = max(1, int(round(dw * scale)))
                        small = cv2.resize(tile_img, (dww, dwh), interpolation=cv2.INTER_AREA)
                        tile_bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                    else:
                        tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
                    res = model.predict(source=tile_bgr, conf=float(args.conf), iou=float(args.iou), verbose=False)
                    if not res:
                        continue
                    r0 = res[0]
                    if r0.masks is None or r0.boxes is None:
                        continue
                    mask_data = r0.masks.data.cpu().numpy()  # N,h_s,w_s (shape of input to model)
                    # Resize masks back to original tile size (dh, dw) when downscaled
                    clses = r0.boxes.cls.cpu().numpy().tolist()
                    confs = r0.boxes.conf.cpu().numpy().tolist()
                    for i, m in enumerate(mask_data):
                        # map mask to full tile size
                        m_resized = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
                        mask_full = np.zeros((h, w), dtype=bool)
                        mask_bin = (m_resized > 0.5)
                        mask_full[y:y + dh, x:x + dw] = mask_bin[:dh, :dw]
                        for poly in mask_to_polygons(mask_full):
                            # coverage over valid data
                            poly_mask = np.zeros_like(mask_full, dtype=bool)
                            cv2.fillPoly(poly_mask, [np.array(poly.exterior.coords, dtype=np.int32)], True)
                            inter = (poly_mask & valid_resized).sum()
                            total = max(1, poly_mask.sum())
                            coverage = inter / float(total)
                            if coverage < float(args.min_poly_valid):
                                continue
                            poly_geo = pixelpoly_to_epsg3857(poly, transform, crs_text)
                            if poly_geo is None:
                                continue
                            cls_id = int(clses[i]) if i < len(clses) else 0
                            cls_name = names.get(cls_id, str(cls_id))
                            conf = float(confs[i]) if i < len(confs) else float(args.conf)
                            feats.append({
                                "type": "Feature",
                                "properties": {
                                    "region_name": region_dir.name,
                                    "sub_region_name": "",
                                    "class_name": cls_name,
                                    "confidence": round(conf, 3),
                                },
                                "geometry": mapping(poly_geo),
                            })
        except Exception:
            continue
    return feats


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = getattr(model.model, "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    regions = [d for d in input_dir.iterdir() if d.is_dir() and d.name.endswith("_FINAL")]
    if not regions:
        regions = [input_dir]
    if args.max_regions and args.max_regions > 0:
        regions = regions[: int(args.max_regions)]

    all_feats: List[Dict[str, Any]] = []
    for reg in regions:
        all_feats.extend(process_region(model, names, reg, args))

    result = {"type": "FeatureCollection", "features": all_feats}
    out_file = output_dir / "result.geojson"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"Features: {len(all_feats)}")
    print(f"Result: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


