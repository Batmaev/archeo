#!/usr/bin/env python3

"""
Официальные скрипты для расчета метрик ожидают один файл ground truth.

А в датасете отдельные geojson-файлы для каждого региона и класса.
Этот скрипт объединяет их в один файл.

Ground truth and predictions must share the same region_name/sub_region_name keys. The baseline predictions use empty sub_region_name, so the merger sets it to "" for GT to match.

Аргументы:
    --input-root (optional, default: train)
    --output (required)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Mapping must match metrics/compute_metrics.py
CLASS_NAME_MAPPING: Dict[str, Optional[str]] = {
    "селище": "selishcha", "Селище": "selishcha", "селища": "selishcha", "Селища": "selishcha",
    "пашня": "pashni", "Пашня": "pashni", "пашни": "pashni", "Пашни": "pashni",
    "пахота": "pashni", "pashnya": "pashni", "Pashnya": "pashni",
    "глубин": "pashni", "Глубин": "pashni",
    "распаханные курганы": "kurgany", "курган": "kurgany", "Курган": "kurgany",
    "курганы": "kurgany", "Курганы": "kurgany", "kurgani": "kurgany", "Kurgani": "kurgany",
    "караванные": "karavannye_puti", "Караванные": "karavannye_puti",
    "караванные пути": "karavannye_puti", "Караванные пути": "karavannye_puti",
    "пути": "karavannye_puti", "Пути": "karavannye_puti",
    "фортификация": "fortifikatsii", "Фортификация": "fortifikatsii",
    "фортификации": "fortifikatsii", "Фортификации": "fortifikatsii",
    "городище": "gorodishcha", "Городище": "gorodishcha",
    "городища": "gorodishcha", "Городища": "gorodishcha",
    "gorodishche": "gorodishcha", "Gorodishche": "gorodishcha",
    "архитектура": "arkhitektury", "Архитектура": "arkhitektury",
    "архитектуры": "arkhitektury", "Архитектуры": "arkhitektury",
    "дорога": "dorogi", "Дорога": "dorogi", "дороги": "dorogi", "Дороги": "dorogi",
    "dorogi": "dorogi", "Dorogi": "dorogi",
    "яма": "yamy", "Яма": "yamy", "ямы": "yamy", "Ямы": "yamy",
    "межа": "mezha", "Межа": "mezha",
    "артефакты лидара": None, "Артефакты лидара": None,
    "артефакты_лидара": None, "Артефакты_лидара": None,
    "лидара": None, "артефакт": None, "Артефакт": None,
    "иное": "inoe", "Иное": "inoe", "inoe": "inoe", "Inoe": "inoe",
}

def normalize_class_name(rus: str) -> Optional[str]:
    return CLASS_NAME_MAPPING.get(rus, rus)

def class_from_filename(path: Path) -> Optional[str]:
    # Expect patterns like "..._городища.geojson"
    stem = path.stem
    parts = stem.split("_")
    rus = parts[-1] if parts else stem
    return normalize_class_name(rus)

def explode_features(region_name: str, class_name: str, fc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "Polygon":
            out.append({
                "type": "Feature",
                "properties": {
                    "region_name": region_name,
                    "sub_region_name": "",
                    "class_name": class_name,
                },
                "geometry": geom,
            })
        elif gtype == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                out.append({
                    "type": "Feature",
                    "properties": {
                        "region_name": region_name,
                        "sub_region_name": "",
                        "class_name": class_name,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly,
                    },
                })
        # silently skip other geometry types
    return out

def _collect_region_dirs(input_root: Path) -> List[Path]:
    """Return a list of region directories to process.

    Supports two modes:
    - input_root is the train root containing many `*_FINAL` subdirs
    - input_root is a single region dir itself (name contains or ends with `FINAL`)
    """
    if not input_root.exists() or not input_root.is_dir():
        return []
    name = input_root.name
    if name.endswith("_FINAL") or "FINAL" in name:
        return [input_root]
    return [p for p in sorted(input_root.glob("*_FINAL")) if p.is_dir()]


def main(input_root: Path, output_path: Path) -> None:
    features: List[Dict[str, Any]] = []
    region_dirs = _collect_region_dirs(input_root)
    for region_dir in region_dirs:
        region_name = region_dir.name
        for ann_dir in sorted(region_dir.glob("06_*_разметка")):
            for gj in ann_dir.rglob("*.geojson"):
                cls = class_from_filename(gj)
                if cls is None:
                    continue  # skip lidar artifacts
                with gj.open("r", encoding="utf-8") as f:
                    fc = json.load(f)
                features.extend(explode_features(region_name, cls, fc))
    merged = {"type": "FeatureCollection", "features": features}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", default="train", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    main(args.input_root, args.output)