#!/usr/bin/env python3
"""
Deterministic group-wise split of sites into train/val.

Rules:
- Split unit is a top-level site directory in `train/*_FINAL/`.
- Related variants (e.g., suffixes like "_1.3км" or duplicate indices) are grouped and kept in the same split.

Outputs:
- splits/train_sites.txt
- splits/val_sites.txt

The script always prints a lightweight summary (counts and a small sample).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "train"
DEFAULT_OUT_DIR = REPO_ROOT / "splits"


def normalize_group_key(dir_name: str) -> str:
    """Return a normalized grouping key for a site directory name.

    Examples:
    - "037_ОСЕЧКИ_1.3км_FINAL" -> "ОСЕЧКИ"
    - "060_НОВОТИТОРОВСКАЯ_2.8км_FINAL" -> "НОВОТИТОРОВСКАЯ"
    - "002_ДЕМИДОВКА_FINAL" -> "ДЕМИДОВКА"
    """
    name = dir_name
    # strip trailing _FINAL or any case variant
    name = re.sub(r"_FINAL$", "", name, flags=re.IGNORECASE)
    # strip km variants like _1.3км, _2км
    name = re.sub(r"_[0-9]+(?:\.[0-9]+)?км$", "", name, flags=re.IGNORECASE)
    # remove leading numeric id like 037_
    name = re.sub(r"^[0-9]+_", "", name)
    # collapse underscores
    name = re.sub(r"_+", "_", name).strip("_ ")
    return name


def stable_hash_to_float(seed: str, key: str) -> float:
    """Map (seed,key) -> [0,1) deterministically using md5.
    """
    h = hashlib.md5((seed + "|" + key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def list_site_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    # prefer *_FINAL, but keep any dir as a site if mixed
    final_dirs = [p for p in dirs if p.name.endswith("_FINAL") or "FINAL" in p.name]
    return final_dirs if final_dirs else dirs


def group_sites(site_dirs: Iterable[Path]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for site in site_dirs:
        key = normalize_group_key(site.name)
        groups.setdefault(key, []).append(site.name)
    return groups


def split_groups(groups: Dict[str, List[str]], val_fraction: float, seed: str) -> Tuple[Set[str], Set[str]]:
    """Return (train_site_names, val_site_names)."""
    group_keys = sorted(groups.keys())
    val_keys: Set[str] = set()
    for g in group_keys:
        if stable_hash_to_float(seed, g) < val_fraction:
            val_keys.add(g)
    train: Set[str] = set()
    val: Set[str] = set()
    for g, names in groups.items():
        target = val if g in val_keys else train
        for n in names:
            target.add(n)
    return train, val


def write_list(path: Path, items: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in sorted(items):
            f.write(it + "\n")


def print_summary(train_sites: Set[str], val_sites: Set[str]) -> None:
    print(f"TRAIN: {len(train_sites)} sites")
    print(f"VAL:   {len(val_sites)} sites")
    # show a couple examples for quick sanity
    ts = sorted(list(train_sites))[:5]
    vs = sorted(list(val_sites))[:5]
    if ts:
        print("  train sample:", ", ".join(ts))
    if vs:
        print("  val   sample:", ", ".join(vs))


def main() -> int:
    p = argparse.ArgumentParser(description="Generate deterministic train/val site splits")
    p.add_argument("--root", default=DEFAULT_ROOT, type=Path, help="Path to train root with *_FINAL dirs")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path, help="Output directory for split lists")
    p.add_argument("--val-fraction", type=float, default=0.20, help="Validation fraction by groups")
    p.add_argument("--seed", type=str, default="42", help="Deterministic seed")
    args = p.parse_args()

    site_dirs = list_site_dirs(args.root)
    groups = group_sites(site_dirs)
    train_sites, val_sites = split_groups(groups, args.val_fraction, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_list(args.out_dir / "train_sites.txt", train_sites)
    write_list(args.out_dir / "val_sites.txt", val_sites)

    # Always print a short summary
    print_summary(train_sites, val_sites)

    print(f"Wrote: {args.out_dir}/train_sites.txt and {args.out_dir}/val_sites.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


