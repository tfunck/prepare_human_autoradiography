"""Shared helpers for the flip/rotation pipeline.

Used by both rotate_gui.py (interactive preview) and build_outputs.py
(final batch build), so the transform shown to the user while editing is
guaranteed to be identical to the transform applied when the final NIfTI
and QC PNG files are written.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

NIFTI_EXTENSIONS = (".nii", ".nii.gz")

CSV_COLUMNS = ["filename", "input_path", "output_nifti", "qc_png", "flip", "rotation"]


def find_nifti_files(data_dir: Path) -> list[Path]:
    """Return all NIfTI files directly inside data_dir, sorted by name."""
    files = [
        p for p in data_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(NIFTI_EXTENSIONS)
    ]
    return sorted(files, key=lambda p: p.name)


def decode_flip(value: object) -> tuple[bool, bool]:
    """Parse a flip string into (left_right_flip, up_down_flip)."""
    text = str(value).strip().upper().replace(" ", "")
    if text in {"", "NAN", "NONE", "0", "FALSE"}:
        return False, False
    return "LR" in text, "UP" in text or "UD" in text


def encode_flip(lr_flip: bool, up_flip: bool) -> str:
    if lr_flip and up_flip:
        return "LR+UP"
    if lr_flip:
        return "LR"
    if up_flip:
        return "UP"
    return ""


def normalize_flip(value: object) -> str:
    return encode_flip(*decode_flip(value))


def normalize_rotation(value: object) -> int:
    """Normalize a rotation value to an integer count of 90 deg CCW steps
    (the ``k`` argument to ``np.rot90``). Accepts 0-3 directly, or plain
    degree values like 90/180/270/-90."""
    text = str(value).strip()
    if text == "" or text.upper() in {"NAN", "NONE"}:
        return 0
    try:
        rotation = float(text)
    except ValueError:
        return 0
    if not math.isfinite(rotation):
        return 0
    rotation = int(round(rotation))
    if abs(rotation) > 3 and rotation % 90 == 0:
        return rotation // 90
    return rotation


def apply_transform(data: np.ndarray, flip_value: object, rotation_steps: object) -> np.ndarray:
    """Apply flip(s) then rotation to a 2D array.

    Order is fixed: LR flip (np.fliplr), then UP/DOWN flip (np.flipud),
    then rotation (np.rot90, k = rotation_steps). This exact order is what
    both the GUI preview and the final build use, so what you see in the
    GUI is exactly what gets written to disk.
    """
    lr_flip, up_flip = decode_flip(flip_value)
    transformed = data
    if lr_flip:
        transformed = np.fliplr(transformed)
    if up_flip:
        transformed = np.flipud(transformed)
    rotation_steps = normalize_rotation(rotation_steps)
    if rotation_steps:
        transformed = np.rot90(transformed, k=rotation_steps)
    return np.ascontiguousarray(transformed)


def preview_array(data: np.ndarray) -> np.ndarray:
    """Rescale a 2D array to uint8 for display/PNG export."""
    array = np.squeeze(np.asarray(data))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {array.shape}")
    if array.dtype == np.uint8:
        return array
    array = np.nan_to_num(array.astype(np.float32, copy=False))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - minimum) / (maximum - minimum)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def resolve_path(root_dir: Path, path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return root_dir / path


def rel_path(root_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_or_update_csv(
    root_dir: Path,
    data_dir: Path,
    csv_path: Path,
    output_nifti_dir: Path,
    output_qc_dir: Path,
) -> pd.DataFrame:
    """Scan data_dir for NIfTI files and build/refresh the transforms CSV.

    - New files found on disk are appended with flip='' rotation=0.
    - Existing rows in csv_path keep whatever flip/rotation the user has
      already set, matched by filename.
    - Files that no longer exist on disk are simply not included.
    """
    files = find_nifti_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {data_dir}")

    records = pd.DataFrame({
        "filename": [f.name for f in files],
        "input_path": [rel_path(root_dir, f) for f in files],
    })
    records["output_nifti"] = [
        rel_path(root_dir, output_nifti_dir / f.name) for f in files
    ]
    records["qc_png"] = [
        rel_path(root_dir, output_qc_dir / (f.name.split(".nii")[0] + ".png"))
        for f in files
    ]
    records["flip"] = ""
    records["rotation"] = 0

    if csv_path.exists():
        existing = pd.read_csv(csv_path, keep_default_na=False)
        if "filename" in existing.columns:
            existing = existing.drop_duplicates("filename", keep="last").set_index("filename")
            for row_index, filename in records["filename"].items():
                if filename not in existing.index:
                    continue
                saved_row = existing.loc[filename]
                if "flip" in existing.columns:
                    records.at[row_index, "flip"] = normalize_flip(saved_row["flip"])
                if "rotation" in existing.columns:
                    records.at[row_index, "rotation"] = normalize_rotation(saved_row["rotation"])

    records["flip"] = records["flip"].apply(normalize_flip)
    records["rotation"] = records["rotation"].apply(normalize_rotation).astype(int)
    return records.loc[:, CSV_COLUMNS]


def write_csv(records: pd.DataFrame, csv_path: Path) -> None:
    output_df = records.copy()
    output_df["flip"] = output_df["flip"].fillna("").astype(str)
    output_df["rotation"] = output_df["rotation"].fillna(0).astype(int)
    output_df.to_csv(csv_path, index=False)
