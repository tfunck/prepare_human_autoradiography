#!/usr/bin/env python3
"""Render one QC figure per tissue chunk to check the masks in mask_final/.

Every section of a chunk (``sub`` + ``hemisphere`` + ``chunk`` in
``csv/master.csv``) is loaded from raw/ with its mask from mask_final/, any
recorded flip/rotation from ``csv/splits/<sub>_<hemi>_<ligand>.qc.csv`` is
applied, and the section is shifted so the mask's centre of mass sits at the
centre of a common canvas. The sections are stacked in cutting order (repeat,
then ligand) and the figure shows 3 axial and 3 sagittal cuts through the
stack, with image content outside the mask tinted red, plus the middle section marking
where the cuts are taken.

Usage
  python make_chunk_qc.py                       # all chunks -> chunk_qc/
  python make_chunk_qc.py --chunks MR1_L_6 MR2_R_3
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import apply_transform, normalize_flip, normalize_rotation, resolve_path

CROP_MARGIN = 10         # px of context kept around each mask bounding box
CUT_FRACTIONS = (0.25, 0.5, 0.75)


def parse_repeat(base: str) -> int:
    parts = str(base).split("#")
    if len(parts) > 6:
        digits = "".join(ch for ch in parts[6] if ch.isdigit())
        if digits:
            return int(digits)
    return 10**6


def read_transforms(splits_dir: Path, sub: str, hemi: str, ligands) -> dict:
    """base -> (flip, rotation) from any qc_gui ``.qc.csv`` files present."""
    lookup: dict[str, tuple] = {}
    for ligand in ligands:
        path = splits_dir / f"{sub}_{hemi}_{ligand}.qc.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path, keep_default_na=False)
        for row in data.itertuples():
            lookup[str(row.base)] = (
                normalize_flip(getattr(row, "flip", "")),
                normalize_rotation(getattr(row, "rotation", 0)),
            )
    return lookup


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def normalize_section(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Stretch each section to its own in-mask 1-99th percentile so ligands
    with very different signal levels are comparable in the stack."""
    values = raw[mask]
    lo, hi = np.percentile(values, (1, 99))
    if hi <= lo:
        lo, hi = float(raw.min()), float(raw.max()) or 1.0
    scaled = (raw.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def load_section(raw_path: Path, mask_path: Path, flip, rotation):
    """Return (cropped raw, cropped mask, centre of mass in crop) or None."""
    raw = load_gray(raw_path)
    mask = load_gray(mask_path) > 0
    if mask.shape != raw.shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (raw.shape[1], raw.shape[0]), Image.NEAREST
            )
        ) > 0
    raw = apply_transform(raw, flip, rotation)
    mask = apply_transform(mask, flip, rotation)
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    r0 = max(0, rows.min() - CROP_MARGIN)
    r1 = min(mask.shape[0], rows.max() + 1 + CROP_MARGIN)
    c0 = max(0, cols.min() - CROP_MARGIN)
    c1 = min(mask.shape[1], cols.max() + 1 + CROP_MARGIN)
    raw = normalize_section(raw, mask)[r0:r1, c0:c1]
    mask = mask[r0:r1, c0:c1]
    center = ndimage.center_of_mass(mask)
    return raw, mask, center


def build_stack(sections):
    """Paste every section into a shared canvas with its mask CoM centred."""
    half_h = max(max(cy, s[0].shape[0] - cy) for s in sections for cy in [s[2][0]])
    half_w = max(max(cx, s[0].shape[1] - cx) for s in sections for cx in [s[2][1]])
    height = int(np.ceil(half_h)) * 2 + 2
    width = int(np.ceil(half_w)) * 2 + 2
    raw_vol = np.zeros((len(sections), height, width), dtype=np.uint8)
    mask_vol = np.zeros((len(sections), height, width), dtype=bool)
    valid_vol = np.zeros((len(sections), height, width), dtype=bool)
    for z, (raw, mask, (cy, cx)) in enumerate(sections):
        top = int(round(height / 2.0 - cy))
        left = int(round(width / 2.0 - cx))
        h, w = raw.shape
        raw_vol[z, top:top + h, left:left + w] = raw
        mask_vol[z, top:top + h, left:left + w] = mask
        valid_vol[z, top:top + h, left:left + w] = True
    return raw_vol, mask_vol, valid_vol


def overlay(raw_slice, mask_slice, valid_slice):
    """Gray image with image content outside the mask tinted red."""
    gray = raw_slice.astype(np.float32) / 255.0
    rgb = np.stack([gray, gray, gray], axis=-1)
    outside = valid_slice & ~mask_slice
    rgb[outside] = rgb[outside] * 0.5 + np.array([0.5, 0.0, 0.0])
    return rgb


def draw_cut(ax, raw_slice, mask_slice, valid_slice, title):
    ax.imshow(overlay(raw_slice, mask_slice, valid_slice), aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_ylabel("section (cutting order)", fontsize=8)
    ax.tick_params(labelsize=7)


def render_figure(raw_vol, mask_vol, valid_vol, title, out_path, dpi):
    nz, height, width = raw_vol.shape
    rows_cut = [int(f * (height - 1)) for f in CUT_FRACTIONS]
    cols_cut = [int(f * (width - 1)) for f in CUT_FRACTIONS]

    fig = plt.figure(figsize=(22, 14))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1])

    ref = fig.add_subplot(grid[:, 0])
    mid = nz // 2
    ref.imshow(overlay(raw_vol[mid], mask_vol[mid], valid_vol[mid]), interpolation="nearest")
    for i, r in enumerate(rows_cut):
        ref.axhline(r, color="cyan", lw=0.8)
        ref.text(2, r - 2, f"A{i + 1}", color="cyan", fontsize=8)
    for i, c in enumerate(cols_cut):
        ref.axvline(c, color="yellow", lw=0.8)
        ref.text(c + 2, height - 4, f"S{i + 1}", color="yellow", fontsize=8)
    ref.set_title(f"Middle section ({mid + 1}/{nz})", fontsize=9)
    ref.axis("off")

    for i, r in enumerate(rows_cut):
        ax = fig.add_subplot(grid[0, i + 1])
        draw_cut(ax, raw_vol[:, r, :], mask_vol[:, r, :], valid_vol[:, r, :], f"Axial A{i + 1} (row {r})")
    for i, c in enumerate(cols_cut):
        ax = fig.add_subplot(grid[1, i + 1])
        draw_cut(ax, raw_vol[:, :, c], mask_vol[:, :, c], valid_vol[:, :, c], f"Sagittal S{i + 1} (col {c})")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def process_chunk(task):
    """Build one chunk figure; returns ``(name, status, message)``."""
    name = task["name"]
    try:
        if os.path.exists(task["out_path"]) and not task["overwrite"]:
            return name, "skipped", ""
        sections, missing, empty = [], 0, 0
        for raw_path, mask_path, flip, rotation in task["sections"]:
            if not (os.path.exists(raw_path) and os.path.exists(mask_path)):
                missing += 1
                continue
            section = load_section(Path(raw_path), Path(mask_path), flip, rotation)
            if section is None:
                empty += 1
                continue
            sections.append(section)
        if not sections:
            return name, "error", f"no usable sections ({missing} missing files, {empty} empty masks)"
        raw_vol, mask_vol, valid_vol = build_stack(sections)
        title = (f"{task['title']}  -  {len(sections)} sections"
                 f"  ({missing} missing files, {empty} empty masks skipped)  -  red = outside mask")
        render_figure(raw_vol, mask_vol, valid_vol, title, task["out_path"], task["dpi"])
        return name, "ok", f"{len(sections)} sections, {missing} missing, {empty} empty"
    except Exception as exc:  # keep the batch alive, record the failure
        return name, "error", f"{type(exc).__name__}: {exc}"


def build_tasks(master: pd.DataFrame, root_dir: Path, mask_dir: Path, splits_dir: Path,
                out_dir: Path, args) -> list[dict]:
    tasks = []
    for (sub, hemi, chunk), group in master.groupby(["sub", "hemisphere", "chunk"], sort=True):
        name = f"{sub}_{hemi}_{chunk}"
        if args.chunks and name not in args.chunks:
            continue
        group = group.assign(_repeat=group["base"].map(parse_repeat))
        group = group.sort_values(["_repeat", "ligand", "base"], kind="stable")
        transforms = read_transforms(splits_dir, str(sub), str(hemi), group["ligand"].astype(str).unique())
        sections = []
        for row in group.itertuples():
            base = str(row.base)
            flip, rotation = transforms.get(base, ("", 0))
            sections.append((
                str(resolve_path(root_dir, row.raw)),
                str(mask_dir / f"{base}.png"),
                flip,
                rotation,
            ))
        tasks.append({
            "name": name,
            "title": f"{sub} {hemi} chunk {chunk}",
            "sections": sections,
            "out_path": str(out_dir / f"{name}.png"),
            "dpi": args.dpi,
            "overwrite": args.overwrite,
        })
    return tasks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Per-chunk axial/sagittal QC figures of mask_final/.")
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path,
                        help="directory holding csv/, raw/ and mask_final/")
    parser.add_argument("--master", default=None, type=Path, help="master CSV (default <root>/csv/master.csv)")
    parser.add_argument("--mask-dir", default=None, type=Path, help="mask folder (default <root>/mask_final)")
    parser.add_argument("--out-dir", default=None, type=Path, help="output folder (default <root>/chunk_qc)")
    parser.add_argument("--chunks", nargs="*", default=None, help="only these chunks, e.g. MR1_L_6")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                        help="parallel worker processes")
    parser.add_argument("--dpi", type=int, default=120, help="dpi of the output figures")
    parser.add_argument("--overwrite", action="store_true", help="rebuild figures that already exist")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root_dir = args.root.expanduser().resolve()
    master_path = args.master or root_dir / "csv" / "master.csv"
    mask_dir = args.mask_dir or root_dir / "mask_final"
    out_dir = args.out_dir or root_dir / "chunk_qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path, keep_default_na=False)
    tasks = build_tasks(master, root_dir, mask_dir, root_dir / "csv" / "splits", out_dir, args)
    if not tasks:
        print("No chunks selected.")
        return 0

    print(f"Building {len(tasks)} chunk figure(s) -> {out_dir}")
    results = []
    if args.workers <= 1:
        for task in tasks:
            results.append(process_chunk(task))
            print("  {}: {} {}".format(*results[-1]), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_chunk, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                print("  {}: {} {}".format(*results[-1]), flush=True)

    errors = [r for r in results if r[1] == "error"]
    print(f"\nwritten: {sum(r[1] == 'ok' for r in results)}  "
          f"skipped: {sum(r[1] == 'skipped' for r in results)}  errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
