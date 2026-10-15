#!/usr/bin/env python3
"""Build raw + edited-mask QC overlays so mask_qc/ can be checked against raw/.

After reviewing a split CSV in ``qc_gui.py`` you get a ``<split>.qc.csv`` plus
hand-edited masks in ``mask_qc/``. This script renders one overlay PNG per
reviewed section -- the raw image in grey with the edited mask outline in red,
in the final flip/rotation the reviewer chose -- so the results can be eyeballed
quickly, exactly like the overlays ``prepare_qc_data.py`` writes to ``qc/``.

Usage
  python make_qc_overlays.py csv/splits/MR1_L_afdx.qc.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import pandas as pd

from common import apply_transform, normalize_flip, normalize_rotation, resolve_path
from prepare_qc_data import binarize, make_overlay, resize_mask_to, to_uint8


def find_root(csv_path: Path, override: object | None) -> Path:
    """Locate the directory that holds the raw/ and mask_qc/ folders."""
    if override:
        return Path(str(override)).expanduser().resolve()
    for parent in [csv_path.parent, *csv_path.parents]:
        if (parent / "raw").is_dir():
            return parent
    return Path.cwd()


def build_overlay(task):
    """Render one overlay; returns ``(base, status, message)``."""
    base = task["base"]
    out_path = task["out_path"]
    try:
        if os.path.exists(out_path) and not task["overwrite"]:
            return base, "skipped", ""
        raw = to_uint8(imageio.imread(task["raw_path"]))
        mask = binarize(imageio.imread(task["mask_path"]), 0)
        if mask.shape[:2] != raw.shape[:2]:
            mask = resize_mask_to(mask, raw.shape[:2])
        raw_t = apply_transform(raw, task["flip"], task["rotation"])
        mask_t = apply_transform(mask, task["flip"], task["rotation"])
        make_overlay(raw_t, mask_t, out_path, task["title"], task["dpi"], task["max_display"])
        return base, "ok", ""
    except Exception as exc:  # keep the batch alive, record the failure
        return base, "error", "{}: {}".format(type(exc).__name__, exc)


def build_tasks(df, root_dir, out_dir, args):
    tasks = []
    for _, row in df.iterrows():
        qc_mask = str(row.get("qc_mask", "")).strip()
        if not qc_mask:
            continue
        mask_path = resolve_path(root_dir, qc_mask)
        if not mask_path.exists():
            continue
        base = str(row.get("base", "")) or mask_path.stem
        raw_path = resolve_path(root_dir, str(row.get("raw", "")).strip())
        flip = normalize_flip(row["flip"]) if "flip" in df.columns else ""
        rotation = normalize_rotation(row["rotation"]) if "rotation" in df.columns else 0
        ligand = str(row.get("ligand", ""))
        title = "{}\n{}  flip:{} rot:{}".format(base, ligand, flip or "none", rotation)
        tasks.append({
            "base": base,
            "raw_path": str(raw_path),
            "mask_path": str(mask_path),
            "out_path": str(out_dir / (base + ".png")),
            "flip": flip,
            "rotation": rotation,
            "title": title,
            "dpi": args.qc_dpi,
            "max_display": args.qc_max_display,
            "overwrite": args.overwrite,
        })
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build raw + edited-mask QC overlays from a qc_gui *.qc.csv.",
    )
    parser.add_argument("csv", help="the *.qc.csv written by qc_gui.py, e.g. csv/splits/MR1_L_afdx.qc.csv")
    parser.add_argument("--root", default=None, help="directory holding raw/ and mask_qc/ (auto-detected by default)")
    parser.add_argument("--out-dir", default=None, help="output directory for overlays (default <root>/mask_qc_overlays)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="parallel worker processes")
    parser.add_argument("--qc-dpi", type=int, default=100, help="dpi of the overlay figures")
    parser.add_argument("--qc-max-display", type=int, default=1200, help="downsample overlays so the longest side is ~this many px")
    parser.add_argument("--limit", type=int, default=0, help="only build the first N overlays (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild overlays that already exist")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit("CSV not found: {}".format(csv_path))

    root_dir = find_root(csv_path, args.root)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root_dir / "mask_qc_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, keep_default_na=False)
    if "qc_mask" not in df.columns:
        raise SystemExit(
            "No 'qc_mask' column in {} -- pass the *.qc.csv written by qc_gui.py.".format(csv_path)
        )

    tasks = build_tasks(df, root_dir, out_dir, args)
    if not tasks:
        print("No edited masks found to overlay (nothing with a qc_mask in {}).".format(csv_path))
        return 0

    print("Building {} overlay(s) -> {}".format(len(tasks), out_dir))
    results = []
    total = len(tasks)
    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            results.append(build_overlay(task))
            if i % 50 == 0 or i == total:
                print("  {}/{}".format(i, total), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_overlay, task) for task in tasks]
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 50 == 0 or done == total:
                    print("  {}/{}".format(done, total), flush=True)

    written = sum(1 for _, status, _ in results if status == "ok")
    skipped = sum(1 for _, status, _ in results if status == "skipped")
    errors = [(base, message) for base, status, message in results if status == "error"]
    print("\nSummary")
    print("  overlays written : {}".format(written))
    print("  skipped (exist)  : {}".format(skipped))
    print("  errors           : {}".format(len(errors)))
    print("  output dir       : {}".format(out_dir))
    for base, message in errors[:10]:
        print("  ERROR {}: {}".format(base, message))
    return 0


if __name__ == "__main__":
    sys.exit(main())
