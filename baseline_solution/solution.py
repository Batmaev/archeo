#!/usr/bin/env python3
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from types import SimpleNamespace

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from affine import Affine
from pyproj import Transformer
from rasterio import features as rio_features
from shapely.affinity import affine_transform as shapely_affine_transform
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.validation import make_valid
from ultralytics import YOLO

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and table-style layout."""

    FORMATS = {
        logging.DEBUG: f"{Colors.BRIGHT_BLACK}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} %(message)s",
        logging.INFO: f"{Colors.BRIGHT_CYAN}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} %(message)s",
        logging.WARNING: f"{Colors.BRIGHT_YELLOW}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} {Colors.YELLOW}%(message)s{Colors.RESET}",
        logging.ERROR: f"{Colors.BRIGHT_RED}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} {Colors.RED}%(message)s{Colors.RESET}",
        logging.CRITICAL: f"{Colors.BG_RED}{Colors.WHITE}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} {Colors.BRIGHT_RED}%(message)s{Colors.RESET}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Configure logging with colored formatter
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("solution")


def log_header(title: str, char: str = "═", width: int = 80):
    """Log a formatted header."""
    logger.info(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{char * width}{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{char * width}{Colors.RESET}")


def log_section(title: str, char: str = "─", width: int = 80):
    """Log a formatted section divider."""
    logger.info(f"{Colors.CYAN}{char * width}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{Colors.BOLD}  {title}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{char * width}{Colors.RESET}")


def log_subsection(title: str, width: int = 80):
    """Log a formatted subsection."""
    logger.info(f"{Colors.DIM}  {'·' * width}{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}    {title}{Colors.RESET}")
    logger.info(f"{Colors.DIM}  {'·' * width}{Colors.RESET}")


def log_table_row(label: str, value: Any, label_width: int = 30):
    """Log a table-style row with label and value."""
    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} "
        f"{Colors.BRIGHT_WHITE}{label:<{label_width}}{Colors.RESET} "
        f"{Colors.DIM}│{Colors.RESET} "
        f"{Colors.BRIGHT_GREEN}{value}{Colors.RESET}"
    )


def log_metric(label: str, value: Any, unit: str = "", label_width: int = 30):
    """Log a metric with highlighting."""
    display_value = f"{value} {unit}".strip() if unit else str(value)
    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} "
        f"{Colors.WHITE}{label:<{label_width}}{Colors.RESET} "
        f"{Colors.DIM}→{Colors.RESET} "
        f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}{display_value}{Colors.RESET}"
    )


def log_file_info(filename: str, size_mb: float):
    """Log file information in a formatted way."""
    logger.info(
        f"{Colors.DIM}  ├─{Colors.RESET} "
        f"{Colors.BRIGHT_YELLOW}📄 {filename}{Colors.RESET} "
        f"{Colors.DIM}({size_mb:.2f} MB){Colors.RESET}"
    )


def log_timing(label: str, elapsed_seconds: float, label_width: int = 30):
    """Log timing information with formatting."""
    if elapsed_seconds < 1:
        time_str = f"{elapsed_seconds * 1000:.0f}ms"
    elif elapsed_seconds < 60:
        time_str = f"{elapsed_seconds:.2f}s"
    else:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        time_str = f"{minutes}m {seconds:.1f}s"

    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} "
        f"{Colors.WHITE}{label:<{label_width}}{Colors.RESET} "
        f"{Colors.DIM}⏱{Colors.RESET} "
        f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}{time_str}{Colors.RESET}"
    )


def prepare_image(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3 and data.shape[0] <= 4:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        max_val, min_val = np.max(data), np.min(data)
        if max_val > min_val:
            data = (255 * (data - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    if data.shape[-1] > 3:
        data = data[:, :, :3]
    return data


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Convert to LAB color space for better results
    if image.shape[-1] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel = clahe.apply(l_channel)

        # Merge channels back
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)

    return enhanced


def downscale_image(image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= target_size:
        return image.copy(), 1.0
    scale = target_size / max_dim
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = Image.fromarray(image).resize(new_size, Image.LANCZOS)
    return np.asarray(resized), scale


def align_and_crop_image(image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[float, float]]:
    img_rgb = image if image.shape[-1] == 3 else np.repeat(image[:, :, :1], 3, axis=-1)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edge = max(1, min(20, h // 10, w // 10))
    edge_regions = [
        gray[:edge, :],
        gray[-edge:, :],
        gray[:, :edge],
        gray[:, -edge:]
    ]
    edge_means = [float(region.mean()) for region in edge_regions if region.size]
    background_sample = sum(edge_means) / len(edge_means) if edge_means else float(gray.mean())
    dark_background = background_sample < 128
    thresh_type = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, 30 if dark_background else 0, 255, thresh_type)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        center = (w / 2.0, h / 2.0)
        return img_rgb, 0.0, (0, 0), center
    main = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main)
    center, size, angle = rect
    if size[0] < size[1]:
        angle += 90.0
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = (0, 0, 0) if dark_background else (255, 255, 255)
    rotated = cv2.warpAffine(img_bgr, rot_mat, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh_rot = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray_rotated, 30 if dark_background else 250, 255, thresh_rot)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), angle, (0, 0), center
    x, y, w_box, h_box = cv2.boundingRect(coords)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(rotated.shape[1] - x, w_box + 2 * margin)
    h_box = min(rotated.shape[0] - y, h_box + 2 * margin)
    crop = rotated[y:y + h_box, x:x + w_box]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), angle, (x, y), center


def load_model(model_path: str, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps:0"
        else:
            device = "cpu"
    log_metric("Device", device, "🚀")
    model = YOLO(model_path)
    target_device = "mps" if device.startswith("mps") else device
    model.to(target_device)
    model.overrides["device"] = target_device

    # Enable GPU optimizations for NVIDIA
    if "cuda" in target_device:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        model.overrides["half"] = False  # Mixed precision (FP16)

    names = getattr(model.model, "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, list):
        names = {idx: name for idx, name in enumerate(names)}
    return model, names


def _tile_positions(length: int, tile_size: int, step: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = [0]
    while positions[-1] + tile_size < length:
        next_pos = positions[-1] + step
        if next_pos + tile_size >= length:
            positions.append(max(length - tile_size, 0))
            break
        positions.append(next_pos)
    return sorted(dict.fromkeys(max(0, p) for p in positions))


def run_tiled_inference(
    model,
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    tile_height: int,
    tile_width: int,
    overlap_ratio: float,
    class_names: Dict[int, str]
) -> List[SimpleNamespace]:
    image_uint8 = prepare_image(image)
    if image_uint8.shape[-1] != 3:
        image_uint8 = prepare_image(image_uint8)
    img_h, img_w = image_uint8.shape[:2]
    stride_y = max(int(tile_height * (1.0 - overlap_ratio)), 1)
    stride_x = max(int(tile_width * (1.0 - overlap_ratio)), 1)
    ys = _tile_positions(img_h, tile_height, stride_y)
    xs = _tile_positions(img_w, tile_width, stride_x)
    num_tiles = len(ys) * len(xs)
    log_metric("Tiles to Process", f"{len(ys)}×{len(xs)} = {num_tiles}")
    detections: List[SimpleNamespace] = []

    # Collect tiles for batch processing
    tiles_to_process = []
    tile_coords = []
    for y in ys:
        y_end = min(y + tile_height, img_h)
        for x in xs:
            x_end = min(x + tile_width, img_w)
            tile = image_uint8[y:y_end, x:x_end]
            if tile.size == 0:
                continue
            tiles_to_process.append(tile)
            tile_coords.append((y, x, y_end, x_end))

    # Process tiles
    for tile_idx, tile in enumerate(tiles_to_process):
        y, x, y_end, x_end = tile_coords[tile_idx]
        tile_bgr = np.ascontiguousarray(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        results = model.predict(
            source=tile_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        if not results:
            continue
        result = results[0]
        if result.masks is None or result.boxes is None:
            continue
        mask_data = result.masks.data.cpu().numpy()
        orig_h, orig_w = result.masks.orig_shape
        box_cls = result.boxes.cls.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy()
        for idx, raw_mask in enumerate(mask_data):
            mask = raw_mask
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            binary = mask > 0.5
            h_limit = min(orig_h, img_h - y)
            w_limit = min(orig_w, img_w - x)
            if h_limit <= 0 or w_limit <= 0:
                continue
            if not binary[:h_limit, :w_limit].any():
                continue
            full_mask = np.zeros((img_h, img_w), dtype=bool)
            full_mask[y:y + h_limit, x:x + w_limit] = binary[:h_limit, :w_limit]
            class_id = int(box_cls[idx]) if idx < len(box_cls) else 0
            confidence = float(box_conf[idx]) if idx < len(box_conf) else float(conf_threshold)
            class_name = class_names.get(class_id, str(class_id))
            detections.append(
                SimpleNamespace(
                    category=SimpleNamespace(id=class_id, name=class_name),
                    score=SimpleNamespace(value=confidence),
                    mask=SimpleNamespace(bool_mask=full_mask)
                )
            )

    # Clear GPU cache after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detections


def normalize_crs(crs_value: str) -> str:
    text = crs_value.strip()
    lower = text.lower()
    if lower.startswith("urn:ogc:def:crs:epsg::"):
        return f"EPSG:{text.split('::')[-1]}"
    if lower.startswith("epsg") and ":" not in text:
        suffix = text.replace("epsg", "").strip(": ")
        return f"EPSG:{suffix}" if suffix else "EPSG:32636" #Fallback to 36N for rare cases
    return text


def utm_to_epsg(utm_value: str) -> Optional[str]:
    cleaned = utm_value.strip().upper()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits:
        return None
    hemisphere = next((ch for ch in reversed(cleaned) if ch.isalpha()), "N")
    try:
        zone = int(digits)
    except ValueError:
        return None
    base = 326 if hemisphere == "N" else 327
    return f"EPSG:{base}{zone:02d}"


def load_region_crs(dir: Path) -> Optional[str]:
    # !!! ALWAYS Load CRS from UTM.json file if available !!!!
    utm_file = dir / "UTM.json"
    if not utm_file.exists():
        return None
    try:
        with utm_file.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", utm_file.name, exc)
        return None
    crs_value = data.get("crs") or data.get("CRS")
    if crs_value:
        return normalize_crs(str(crs_value))
    utm_value = data.get("utm") or data.get("UTM")
    return utm_to_epsg(str(utm_value)) if utm_value else None


def get_hillshades(dir: Path) -> List[Path]:
    files: List[Path] = []
    for filename in dir.rglob("*.tif"):
        if "lidar" in filename.stem.lower():
            files.append(filename)
    return sorted(files)


def mask_to_polygons(mask: np.ndarray, max_polygons: int = 100) -> Iterable[Polygon]:
    """Convert binary mask to polygons using contours (fast) instead of rasterio shapes (slow)."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        if count >= max_polygons:
            break

        # Simplify contour to reduce vertices
        epsilon = 0.5  # Adjust for tolerance
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        if len(simplified) < 3:
            continue

        # Convert contour to polygon coordinates
        coords = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]

        try:
            poly = Polygon(coords)
            poly = make_valid(poly)

            if poly.is_empty or poly.area < 0.1:
                continue

            if isinstance(poly, MultiPolygon):
                for sub in poly.geoms:
                    if not sub.is_empty:
                        yield sub
                        count += 1
            else:
                yield poly
                count += 1
        except Exception:
            continue


def detections_to_features(detections, class_names: Dict[int, str], image_shape: Tuple[int, int],
                           transform: Affine, source_crs: str, scale: float, angle: float,
                           crop_offset: Tuple[int, int], rotation_center: Tuple[float, float],
                           region_name: str, markup_type: str = "li") -> List[Dict[str, Any]]:
    """Converts tiled detections with masks to GeoJSON features."""
    _ = image_shape
    affine_params = (transform.a, transform.b, transform.d, transform.e, transform.c, transform.f)
    transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    source_urn = f"urn:ogc:def:crs:{source_crs.replace(':', '::')}"
    target_urn = "urn:ogc:def:crs:EPSG::3857"
    features: List[Dict[str, Any]] = []
    rotate_origin = (float(rotation_center[0]), float(rotation_center[1]))

    # Pre-compute transformation matrices
    need_rotate = abs(angle) > 1e-3
    need_scale = abs(scale - 1.0) > 1e-6

    def apply_transforms(poly: Polygon) -> Optional[Polygon]:
        """Apply all geometric transformations efficiently."""
        try:
            # Translate
            if crop_offset[0] != 0 or crop_offset[1] != 0:
                poly = shapely_translate(poly, xoff=float(crop_offset[0]), yoff=float(crop_offset[1]))

            # Rotate (skip if negligible)
            if need_rotate:
                poly = shapely_rotate(poly, angle=-angle, origin=rotate_origin, use_radians=False)

            # Scale (skip if 1.0)
            if need_scale:
                poly = shapely_scale(poly, xfact=1.0 / scale, yfact=1.0 / scale, origin=(0.0, 0.0))

            # Affine + CRS transform (combined)
            poly = shapely_affine_transform(poly, affine_params)
            poly = shapely_transform(transformer.transform, poly)

            poly = make_valid(poly)
            return poly if not poly.is_empty else None
        except Exception:
            return None

    for det in detections:
        mask = getattr(det, "mask", None)
        if mask is None or not mask.bool_mask.any():
            continue

        polygons = mask_to_polygons(det.mask.bool_mask.astype(np.uint8))
        for poly in polygons:
            poly = apply_transforms(poly)
            if poly is None or poly.area < 1.0:
                continue

            # Aggressive simplification to reduce coordinates
            poly_simplified = poly.simplify(2.0, preserve_topology=True)

            features.append({
                "type": "Feature",
                "properties": {
                    "class_name": class_names.get(det.category.id, det.category.name),
                    "region_name": region_name,
                    "sub_region_name": "",
                    "markup_type": markup_type,
                    "original_crs": source_urn,
                    "crs": target_urn,
                    "fid": 0,
                    "confidence": round(det.score.value, 3),
                },
                "geometry": mapping(poly_simplified),
            })

    return features


def deduplicate_polygons(features: List[Dict[str, Any]], iou_threshold: float = 0.95) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for feature in features:
        grouped[feature["properties"]["class_name"]].append(feature)
    kept: List[Dict[str, Any]] = []
    for items in grouped.values():
        unique: List[Tuple[Polygon, Dict[str, Any]]] = []
        for feat in items:
            poly = make_valid(shape(feat["geometry"]))
            if poly.is_empty:
                continue
            duplicate = False
            for stored_poly, _ in unique:
                intersection = poly.intersection(stored_poly).area
                union = poly.union(stored_poly).area
                if union and intersection / union > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                unique.append((poly, feat))
        kept.extend(feat for _, feat in unique)
    return kept


def merge_features(features: List[Dict[str, Any]], buffer_distance: float = 0.5,
                   min_area: float = 1.0) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for feat in features:
        props = feat["properties"]
        key = (
            props["class_name"],
            props["region_name"],
            props["sub_region_name"],
            props["markup_type"],
        )
        groups[key].append(feat)
    merged: List[Dict[str, Any]] = []
    for key, feats in groups.items():
        polygons: List[Polygon] = []
        confidences: List[float] = []
        for feat in feats:
            poly = make_valid(shape(feat["geometry"]))
            if poly.is_empty:
                continue
            polygons.append(poly)
            conf = feat["properties"].get("confidence")
            if conf is not None:
                confidences.append(conf)
        if not polygons:
            continue
        buffered = [poly.buffer(buffer_distance) for poly in polygons]
        union = unary_union(buffered).buffer(-buffer_distance)
        geometries = union.geoms if isinstance(union, MultiPolygon) else [union]
        template = feats[0]["properties"].copy()
        for geom in geometries:
            geom = make_valid(geom)
            if not isinstance(geom, Polygon) or geom.is_empty or geom.area < min_area:
                continue
            props = template.copy()
            if confidences:
                props["confidence"] = round(sum(confidences) / len(confidences), 3)
            merged.append({
                "type": "Feature",
                "properties": props,
                "geometry": mapping(geom),
            })
    return merged


def process_region(
    model,
    class_names: Dict[int, str],
    region_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    tile_height: int,
    tile_width: int,
    overlap_ratio: float,
    enable_downscale: bool,
    enable_alignment: bool,
    enable_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_size: int
) -> List[Dict[str, Any]]:
    region_start_time = time.time()
    log_section(f"🏗️  Processing Region: {region_dir.name}")
    region_crs = load_region_crs(region_dir)
    if region_crs:
        log_table_row("Region CRS", f"{region_crs} (from UTM.json)")
    else:
        log_table_row("Region CRS", "Auto-detect")
    lidar_files = get_hillshades(region_dir)
    if not lidar_files:
        logger.warning(f"{Colors.YELLOW}  ⚠️  No hillshades found{Colors.RESET}")
        return []
    log_table_row("Hillshades Found", len(lidar_files))
    logger.info("")
    features_all: List[Dict[str, Any]] = []
    for idx, tif_path in enumerate(lidar_files, 1):
        file_start_time = time.time()
        file_size_mb = tif_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"{Colors.DIM}  ┌{'─' * 76}┐{Colors.RESET}"
        )
        logger.info(
            f"{Colors.DIM}  │{Colors.RESET} "
            f"{Colors.BRIGHT_CYAN}{Colors.BOLD}File {idx}/{len(lidar_files)}: {tif_path.name}{Colors.RESET}"
            f"{Colors.DIM}{' ' * (75 - len(f'File {idx}/{len(lidar_files)}: {tif_path.name}'))}│{Colors.RESET}"
        )
        logger.info(
            f"{Colors.DIM}  ├{'─' * 76}┤{Colors.RESET}"
        )
        log_table_row("File Size", f"{file_size_mb:.2f} MB")
        try:
            load_start = time.time()
            with rasterio.open(tif_path) as src:
                data = src.read()
                transform = src.transform
                file_crs = src.crs.to_string() if src.crs else None
            load_elapsed = time.time() - load_start
            log_timing("Load Time", load_elapsed)

            prep_start = time.time()
            prepared = prepare_image(data)
            prep_elapsed = time.time() - prep_start
            log_table_row("Original Image Size", f"{prepared.shape[1]}×{prepared.shape[0]}")
            log_timing("Prepare Time", prep_elapsed)

            working_image = prepared
            scale = 1.0

            if enable_clahe:
                clahe_start = time.time()
                working_image = apply_clahe(working_image, clahe_clip_limit, (clahe_tile_size, clahe_tile_size))
                clahe_elapsed = time.time() - clahe_start
                log_timing("CLAHE Time", clahe_elapsed)

            if enable_downscale:
                scale_start = time.time()
                original_size = f"{working_image.shape[1]}×{working_image.shape[0]}"
                working_image, scale = downscale_image(working_image)
                scale_elapsed = time.time() - scale_start
                log_table_row("Downscaled To", f"{working_image.shape[1]}×{working_image.shape[0]} (scale: {scale:.3f})")
                log_timing("Downscale Time", scale_elapsed)

            if enable_alignment:
                align_start = time.time()
                aligned_image, angle, crop_offset, rotation_center = align_and_crop_image(working_image)
                align_elapsed = time.time() - align_start
                log_table_row("Aligned & Cropped", f"{aligned_image.shape[1]}×{aligned_image.shape[0]} (rotated {angle:.2f}°)")
                log_timing("Alignment Time", align_elapsed)
            else:
                aligned_image = working_image
                angle = 0.0
                crop_offset = (0, 0)
                rotation_center = (
                    working_image.shape[1] / 2.0,
                    working_image.shape[0] / 2.0
                )

            log_metric("Processing Size", f"{aligned_image.shape[1]}×{aligned_image.shape[0]}")

            infer_start = time.time()
            detections = run_tiled_inference(
                model,
                aligned_image,
                conf_threshold,
                iou_threshold,
                tile_height,
                tile_width,
                overlap_ratio,
                class_names
            )
            infer_elapsed = time.time() - infer_start
            log_timing("Inference Time", infer_elapsed)

            if not detections:
                log_table_row("Detections", f"{Colors.DIM}0 (no objects found){Colors.RESET}")
                logger.info(
                    f"{Colors.DIM}  └{'─' * 76}┘{Colors.RESET}\n"
                )
                continue
            crs = region_crs or file_crs or "EPSG:32636"
            if "LOCAL_CS" in crs.upper():
                logger.warning(f"{Colors.YELLOW}  ⚠️  LOCAL_CS detected, falling back to EPSG:32636{Colors.RESET}")
                crs = "EPSG:32636"

            feat_start = time.time()
            feature_batch = detections_to_features(
                detections,
                class_names,
                aligned_image.shape,
                transform,
                crs,
                scale,
                angle,
                crop_offset,
                rotation_center,
                region_dir.name
            )
            feat_elapsed = time.time() - feat_start
            log_metric("Features Generated", len(feature_batch), "✓")
            log_timing("Feature Conversion Time", feat_elapsed)
            features_all.extend(feature_batch)
        except Exception as exc:
            logger.error(f"{Colors.RED}  ✗ Failed: {exc}{Colors.RESET}")

        file_elapsed = time.time() - file_start_time
        log_timing("Total File Time", file_elapsed)
        logger.info(
            f"{Colors.DIM}  └{'─' * 76}┘{Colors.RESET}\n"
        )

    region_elapsed = time.time() - region_start_time
    log_timing("Total Region Time", region_elapsed)
    return features_all


def assign_fids(features: List[Dict[str, Any]]) -> None:
    for idx, feat in enumerate(features):
        feat["properties"]["fid"] = idx


def predict(
    input_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    merge_buffer: float = 0.5,
    min_area: float = 1.0,
    enable_downscale: bool = True,
    enable_alignment: bool = True,
    enable_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    tile_height: int = 1024,
    tile_width: int = 1024,
    overlap_ratio: float = 0.3,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    pipeline_start_time = time.time()
    log_header("🗿🔎 Archeology segmentation pipeline 🔎🗿", "═", 80)
    logger.info("⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    output_path.mkdir(parents=True, exist_ok=True)
    weights = Path(model_path) if model_path else Path(__file__).parent / "model.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    log_section("📋 Configuration")
    log_table_row("Input Directory", input_dir)
    log_table_row("Output Directory", output_dir)
    log_table_row("Model Weights", weights.name)
    logger.info("")

    log_subsection("Model Parameters")
    log_table_row("Confidence Threshold", f"{conf_threshold:.2f}")
    log_table_row("IOU Threshold", f"{iou_threshold:.2f}")
    logger.info("")

    log_subsection("Tiling Parameters")
    log_table_row("Tile Size", f"{tile_width}×{tile_height}")
    log_table_row("Tile Overlap", f"{overlap_ratio:.1%}")
    logger.info("")

    log_subsection("Preprocessing Options")
    log_table_row("CLAHE Enhancement", "✓ Enabled" if enable_clahe else "✗ Disabled")
    if enable_clahe:
        log_table_row("  └─ Clip Limit", f"{clahe_clip_limit:.1f}")
        log_table_row("  └─ Tile Grid Size", f"{clahe_tile_size}×{clahe_tile_size}")
    log_table_row("Downscaling", "✓ Enabled" if enable_downscale else "✗ Disabled")
    log_table_row("Alignment & Rotation", "✓ Enabled" if enable_alignment else "✗ Disabled")
    logger.info("")

    log_subsection("Post-Processing")
    log_table_row("Merge Buffer", f"{merge_buffer:.1f} px")
    log_table_row("Minimum Area", f"{min_area:.1f} px²")
    logger.info("")

    log_subsection("GPU Optimization")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log_table_row("GPU Device", f"{gpu_name}")
        log_table_row("GPU Memory", f"{gpu_memory:.1f} GB")
        log_table_row("CUDA Enabled", "✓ Yes")
        log_table_row("Mixed Precision", "✓ FP32 (Disabled)")
    else:
        log_table_row("GPU Available", "✗ No (CPU Mode)")
    logger.info("")

    model_load_start = time.time()
    model, class_names = load_model(str(weights))
    model_load_elapsed = time.time() - model_load_start
    log_timing("Model Load Time", model_load_elapsed)
    logger.info("")

    region_dirs = sorted(d for d in input_path.iterdir() if d.is_dir() and d.name.endswith("_FINAL"))
    if not region_dirs:
        region_dirs = [input_path]
    log_table_row("Regions to Process", len(region_dirs))
    logger.info("")

    processing_start = time.time()
    collected: List[Dict[str, Any]] = []
    for region_dir in region_dirs:
        collected.extend(
            process_region(
                model,
                class_names,
                region_dir,
                conf_threshold,
                iou_threshold,
                tile_height,
                tile_width,
                overlap_ratio,
                enable_downscale,
                enable_alignment,
                enable_clahe,
                clahe_clip_limit,
                clahe_tile_size
            )
        )
    processing_elapsed = time.time() - processing_start
    log_timing("Total Processing Time", processing_elapsed)
    logger.info("")

    log_section("📊 Post-Processing Statistics")
    log_metric("Raw Features", len(collected))

    dedup_elapsed = 0.0
    merge_elapsed = 0.0
    dedup_start = time.time()
    if collected:
        unique = deduplicate_polygons(collected)
        dedup_elapsed = time.time() - dedup_start
        log_metric("After Deduplication", len(unique), f"(-{len(collected) - len(unique)})")
        log_timing("Deduplication Time", dedup_elapsed)

        merge_start = time.time()
        merged = merge_features(unique, buffer_distance=merge_buffer, min_area=min_area)
        merge_elapsed = time.time() - merge_start
        log_metric("After Merging", len(merged), f"(-{len(unique) - len(merged)})")
        log_timing("Merge Time", merge_elapsed)
    else:
        merged = []
    logger.info("")

    fid_start = time.time()
    assign_fids(merged)
    fid_elapsed = time.time() - fid_start
    log_timing("Assign FID Time", fid_elapsed)

    save_start = time.time()
    result = {"type": "FeatureCollection", "features": merged}
    output_file = output_path / "result.geojson"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    save_elapsed = time.time() - save_start
    log_timing("Save GeoJSON Time", save_elapsed)

    log_section("✅ Processing Complete")
    log_metric("Total Features", len(merged), "🎯")
    log_metric("Output File", str(output_file), "💾")
    logger.info("")

    pipeline_elapsed = time.time() - pipeline_start_time
    log_section("⏱️  Timing Summary")
    log_timing("Model Load", model_load_elapsed)
    log_timing("Region Processing", processing_elapsed)
    log_timing("Post-Processing", dedup_elapsed + merge_elapsed + fid_elapsed + save_elapsed)
    log_timing("Total Pipeline Time", pipeline_elapsed)
    logger.info("")

    log_header("Done!", "═", 80)

    return {
        "total_features": len(merged),
        "output_file": str(output_file),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python solution.py <input_dir> <output_dir> [options]")
        print("Options:")
        print("  --model <path>           Path to model weights")
        print("  --downscale              Enable image downscaling")
        print("  --align                  Enable image alignment and rotation")
        print("  --clahe                  Enable CLAHE contrast enhancement")
        print("  --clahe-clip <float>     CLAHE clip limit (default: 2.0)")
        print("  --clahe-tile <int>       CLAHE tile grid size (default: 8)")
        sys.exit(1)
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    model_path = None
    enable_downscale = False
    enable_alignment = False
    enable_clahe = True
    clahe_clip_limit = 2.0
    clahe_tile_size = 8
    idx = 3
    while idx < len(sys.argv):
        arg = sys.argv[idx]
        if arg == "--downscale":
            enable_downscale = True
        elif arg == "--align":
            enable_alignment = True
        elif arg == "--clahe":
            enable_clahe = True
        elif arg == "--clahe-clip":
            if idx + 1 >= len(sys.argv):
                print("Missing value for --clahe-clip", file=sys.stderr)
                sys.exit(1)
            clahe_clip_limit = float(sys.argv[idx + 1])
            idx += 1
        elif arg == "--clahe-tile":
            if idx + 1 >= len(sys.argv):
                print("Missing value for --clahe-tile", file=sys.stderr)
                sys.exit(1)
            clahe_tile_size = int(sys.argv[idx + 1])
            idx += 1
        elif arg == "--model":
            if idx + 1 >= len(sys.argv):
                print("Missing value for --model", file=sys.stderr)
                sys.exit(1)
            model_path = sys.argv[idx + 1]
            idx += 1
        elif arg.startswith("--"):
            print(f"Unknown option: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            if model_path is None:
                model_path = arg
            else:
                logger.warning("Ignoring extra argument: %s", arg)
        idx += 1
    try:
        stats = predict(
            input_dir,
            output_dir,
            model_path=model_path,
            enable_downscale=enable_downscale,
            enable_alignment=enable_alignment,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size
        )
        print(f"Features: {stats['total_features']}")
        print(f"Result: {stats['output_file']}")
    except Exception as exc:
        logger.error("Processing failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
