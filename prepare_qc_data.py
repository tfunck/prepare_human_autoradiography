#!/usr/bin/env python3
"""Prepare raw 2D autoradiographs and crop masks for manual QC.

Pipeline (all stages are resumable -- existing outputs are skipped unless
``--overwrite`` is given):

  1. Convert every raw ``.TIF``/``.tif`` to a portable uint8 ``.png`` in ``raw/``,
     downsampled to the crop-mask resolution (~10x smaller) so the images are
     small enough to share. Raw images with no mask are downsampled by
     ``--downsample`` (default 10).
  2. Copy the matching crop mask into ``mask/`` at its native resolution -- no
     upsampling is needed now that the raw image is downsampled to match it --
     and write a raw+mask overlay to ``qc/`` so the correspondence can be
     eyeballed quickly.
  3. Build ``csv/master.csv`` with the raw/mask paths plus the subject,
     hemisphere, chunk and ligand parsed from each filename, then write one
     ``csv/splits/<sub>_<hemi>_<ligand>.csv`` per group so the QC work can be
     divided between several people.

Filename convention (``#`` separated), e.g. ``OL#hg#MR1s6#L#afdx#96594#01#L.TIF``:
    field[2] = subject + chunk (e.g. ``MR1s6`` -> sub ``MR1``, chunk ``6``)
    field[3] = hemisphere (``L``/``R``)
    field[4] = ligand      (e.g. ``afdx``)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window in a worker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image

RAW_EXTENSIONS = (".TIF", ".tif", ".TIFF", ".tiff")

DEFAULT_RAW_SRC = "/data/receptor/human/raw/img_lin"
DEFAULT_MASK_SRC = "/data/receptor/human/crop/combined_final/combined_final/mask_v2"

# Pillow >= 9.1 moved the resampling filters onto Image.Resampling.
try:
    NEAREST = Image.Resampling.NEAREST
    DOWNSAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    NEAREST = Image.NEAREST
    DOWNSAMPLE = Image.LANCZOS


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_metadata(stem):
    """Return (sub, hemisphere, chunk, ligand) parsed from a filename stem.

    Any field that cannot be parsed comes back as an empty string so a bad
    name never aborts the whole run.
    """
    parts = stem.split("#")
    sub = hemisphere = chunk = ligand = ""
    if len(parts) > 2:
        sub_chunk = re.split("[sS]", parts[2])
        sub = sub_chunk[0]
        chunk = sub_chunk[1] if len(sub_chunk) > 1 else ""
    if len(parts) > 3:
        hemisphere = parts[3]
    if len(parts) > 4:
        ligand = parts[4]
    return sub, hemisphere, chunk, ligand


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def to_uint8(arr):
    """Collapse to 2D and return a uint8 copy.

    uint8 input is passed through untouched so the real pixel values are
    preserved; any other dtype is min-max scaled into 0-255 (the only lossless
    option when the target is an 8-bit PNG).
    """
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float64)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def binarize(mask, threshold):
    """Return a 0/255 uint8 mask (foreground = value > threshold)."""
    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=2)
    return ((mask > threshold).astype(np.uint8)) * 255


def resize_mask_to(mask_bin, target_hw):
    """Nearest-neighbour resize a 0/255 mask up to ``(height, width)``."""
    height, width = target_hw
    image = Image.fromarray(mask_bin).resize((width, height), NEAREST)
    return np.asarray(image)


def resize_to_uint8(arr_u8, target_hw):
    """Resize a 2D uint8 image down to ``(height, width)`` with LANCZOS."""
    height, width = target_hw
    if arr_u8.shape[:2] == (height, width):
        return np.ascontiguousarray(arr_u8)
    image = Image.fromarray(arr_u8).resize((width, height), DOWNSAMPLE)
    return np.asarray(image)


def make_overlay(raw_u8, mask_u8, out_path, title, dpi, max_display):
    """Write a QC figure: raw image in grey with the mask outline in red."""
    step = max(1, int(round(max(raw_u8.shape) / max_display)))
    raw_small = raw_u8[::step, ::step]
    mask_small = mask_u8[::step, ::step] > 0

    height, width = raw_small.shape
    fig, ax = plt.subplots(figsize=(width / 150.0, height / 150.0))
    ax.imshow(raw_small, cmap="gray", vmin=0, vmax=255)
    ax.imshow(np.ma.masked_where(~mask_small, mask_small), cmap="autumn", alpha=0.25)
    if mask_small.any():
        ax.contour(mask_small, levels=[0.5], colors="red", linewidths=0.6)
    ax.set_title(title, fontsize=7)
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-image worker
# --------------------------------------------------------------------------- #
def process_one(tif_path, cfg):
    """Convert one raw image + mask and return a dict row for the master CSV."""
    stem = os.path.splitext(os.path.basename(tif_path))[0]
    sub, hemisphere, chunk, ligand = parse_metadata(stem)

    raw_png = os.path.join(cfg["raw_dir"], stem + ".png")
    mask_png = os.path.join(cfg["mask_dir"], stem + ".png")
    qc_png = os.path.join(cfg["qc_dir"], stem + ".png")
    mask_src = os.path.join(cfg["mask_src"], stem + ".png")

    row = {
        "base": stem,
        "sub": sub,
        "hemisphere": hemisphere,
        "chunk": chunk,
        "ligand": ligand,
        "raw": "",
        "mask": "",
        "qc": "",
        "has_mask": False,
        "mask_resized": False,
        "raw_h": "",
        "raw_w": "",
        "raw_src_h": "",
        "raw_src_w": "",
        "mask_src_h": "",
        "mask_src_w": "",
        "error": "",
    }

    try:
        has_mask = os.path.exists(mask_src)
        src_mask = None
        if has_mask:
            src_mask = imageio.imread(mask_src)
            row["mask_src_h"], row["mask_src_w"] = int(src_mask.shape[0]), int(src_mask.shape[1])

        # --- raw: reuse the png on reruns so we don't re-decode the big TIF ---
        if os.path.exists(raw_png) and not cfg["overwrite"]:
            raw = imageio.imread(raw_png)
        else:
            raw_full = to_uint8(imageio.imread(tif_path))
            row["raw_src_h"], row["raw_src_w"] = int(raw_full.shape[0]), int(raw_full.shape[1])
            if has_mask:
                # downsample the raw image onto the mask's pixel grid
                target_hw = (int(src_mask.shape[0]), int(src_mask.shape[1]))
            else:
                factor = max(1, cfg["downsample"])
                target_hw = (max(1, round(raw_full.shape[0] / factor)), max(1, round(raw_full.shape[1] / factor)))
            raw = resize_to_uint8(raw_full, target_hw)
            imageio.imwrite(raw_png, raw)
        row["raw"] = os.path.relpath(raw_png, cfg["out_dir"])
        row["raw_h"], row["raw_w"] = int(raw.shape[0]), int(raw.shape[1])

        if not has_mask:
            return row  # raw with no mask -- recorded, nothing else to do

        row["has_mask"] = True
        mask_bin = binarize(src_mask, cfg["mask_threshold"])
        # raw is downsampled to the mask resolution, so they already match; only
        # realign if integer rounding left them off by a pixel.
        if mask_bin.shape[:2] != raw.shape[:2]:
            mask_bin = resize_mask_to(mask_bin, raw.shape[:2])
            row["mask_resized"] = True

        if os.path.exists(mask_png) and not cfg["overwrite"]:
            mask_out = imageio.imread(mask_png)
        else:
            mask_out = mask_bin
            imageio.imwrite(mask_png, mask_out)
        row["mask"] = os.path.relpath(mask_png, cfg["out_dir"])

        # QC overlay: raw image in grey with the mask outline in red
        if not cfg["no_qc"]:
            if not (os.path.exists(qc_png) and not cfg["overwrite"]):
                if row["raw_src_h"] != "":
                    size_note = "{}x{} (from {}x{})".format(
                        row["raw_h"], row["raw_w"], row["raw_src_h"], row["raw_src_w"]
                    )
                else:
                    size_note = "{}x{}".format(row["raw_h"], row["raw_w"])
                title = "{}\n{}  {}".format(stem, ligand, size_note)
                make_overlay(raw, mask_out, qc_png, title, cfg["qc_dpi"], cfg["qc_max_display"])
            row["qc"] = os.path.relpath(qc_png, cfg["out_dir"])

    except Exception as exc:  # keep the batch alive, record the failure
        row["error"] = "{}: {}".format(type(exc).__name__, exc)
    return row


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def collect_raw_files(raw_src, subjects, hemispheres, ligands, limit):
    files = sorted(
        os.path.join(raw_src, name)
        for name in os.listdir(raw_src)
        if name.endswith(RAW_EXTENSIONS)
    )
    selected = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        sub, hemisphere, _, ligand = parse_metadata(stem)
        if subjects and sub not in subjects:
            continue
        if hemispheres and hemisphere not in hemispheres:
            continue
        if ligands and ligand not in ligands:
            continue
        selected.append(path)
    if limit:
        selected = selected[:limit]
    return selected


def write_csvs(rows, out_dir):
    csv_dir = os.path.join(out_dir, "csv")
    splits_dir = os.path.join(csv_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("base").reset_index(drop=True)
    master_path = os.path.join(csv_dir, "master.csv")
    df.to_csv(master_path, index=False)

    split_paths = []
    valid = df[(df["sub"] != "") & (df["hemisphere"] != "") & (df["ligand"] != "")]
    for (sub, hemisphere, ligand), group in valid.groupby(["sub", "hemisphere", "ligand"]):
        name = "{}_{}_{}.csv".format(sub, hemisphere, ligand)
        path = os.path.join(splits_dir, name)
        group.to_csv(path, index=False)
        split_paths.append(path)
    return df, master_path, split_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-src", default=DEFAULT_RAW_SRC, help="directory of raw .TIF/.tif autoradiographs")
    parser.add_argument("--mask-src", default=DEFAULT_MASK_SRC, help="directory of crop mask .png files")
    parser.add_argument("--out-dir", default=".", help="output root (holds raw/, mask/, qc/, csv/)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="parallel worker processes")
    parser.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all); handy for testing")
    parser.add_argument("--subjects", nargs="*", default=None, help="only these subjects, e.g. MR1 MR2")
    parser.add_argument("--hemispheres", nargs="*", default=None, help="only these hemispheres, e.g. L R")
    parser.add_argument("--ligands", nargs="*", default=None, help="only these ligands, e.g. afdx flum")
    parser.add_argument("--mask-threshold", type=int, default=0, help="mask foreground = value > threshold")
    parser.add_argument("--downsample", type=int, default=10, help="downsample factor for raw images with no mask; with a mask the raw image is resized to the mask resolution")
    parser.add_argument("--qc-dpi", type=int, default=100, help="dpi of the QC overlay figures")
    parser.add_argument("--qc-max-display", type=int, default=1200, help="downsample QC images so the longest side is ~this many px")
    parser.add_argument("--no-qc", action="store_true", help="skip the matplotlib overlay figures")
    parser.add_argument("--overwrite", action="store_true", help="rebuild outputs that already exist")
    parser.add_argument("--csv-only", action="store_true", help="only (re)build the CSVs from existing outputs")
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    cfg = {
        "out_dir": out_dir,
        "raw_dir": os.path.join(out_dir, "raw"),
        "mask_dir": os.path.join(out_dir, "mask"),
        "qc_dir": os.path.join(out_dir, "qc"),
        "mask_src": args.mask_src,
        "mask_threshold": args.mask_threshold,
        "downsample": args.downsample,
        "qc_dpi": args.qc_dpi,
        "qc_max_display": args.qc_max_display,
        "no_qc": args.no_qc,
        "overwrite": args.overwrite,
    }
    for key in ("raw_dir", "mask_dir", "qc_dir"):
        os.makedirs(cfg[key], exist_ok=True)

    tif_files = collect_raw_files(args.raw_src, args.subjects, args.hemispheres, args.ligands, args.limit)
    print("Found {} raw image(s) to process.".format(len(tif_files)))
    if not tif_files:
        print("Nothing to do.")
        return 0

    rows = []
    total = len(tif_files)
    if args.csv_only or args.workers <= 1:
        for i, path in enumerate(tif_files, 1):
            rows.append(process_one(path, cfg))
            if i % 50 == 0 or i == total:
                print("  processed {}/{}".format(i, total), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, path, cfg): path for path in tif_files}
            done = 0
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 50 == 0 or done == total:
                    print("  processed {}/{}".format(done, total), flush=True)

    df, master_path, split_paths = write_csvs(rows, out_dir)

    errors = df[df["error"] != ""]
    missing_mask = df[~df["has_mask"].astype(bool)]
    realigned = df[df["mask_resized"].astype(bool)]
    print("\nSummary")
    print("  images processed : {}".format(len(df)))
    print("  with mask        : {}".format(int(df["has_mask"].astype(bool).sum())))
    print("  missing mask     : {}".format(len(missing_mask)))
    print("  raw downsampled  : ~1/{}x (to mask resolution where available)".format(args.downsample))
    print("  masks realigned  : {}".format(len(realigned)))
    print("  errors           : {}".format(len(errors)))
    print("  master csv       : {}".format(master_path))
    print("  split csv files  : {} (in {})".format(len(split_paths), os.path.join(out_dir, "csv", "splits")))
    if len(missing_mask):
        print("  NOTE: {} raw image(s) had no matching mask (see has_mask=False in master.csv)".format(len(missing_mask)))
    if len(errors):
        print("  NOTE: {} image(s) failed (see the 'error' column in master.csv)".format(len(errors)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
