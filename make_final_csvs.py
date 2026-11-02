#!/usr/bin/env python3
"""Build csv_final/ from csv/ so every section points at its mask in mask_final/.

* ``mask`` -> ``mask_final/<base>.png`` if that file exists, else blank;
  ``has_mask`` is set to match.
* ``section`` = the integer field just before the trailing ``#L`` of ``base``
  (e.g. ``OL#HG#MR1S6#L#dpmg#6555#01#L`` -> 1). Sections whose field is not
  an integer (``UB``, position unknown) get a blank section.
* csv_final/master.csv and csv_final/splits/*.csv mirror csv/.
* csv_final/chunks/<sub>_<hemisphere>_<chunk>.csv hold master.csv grouped by
  chunk, without the blank-section rows (they cannot be placed in a stack).

Usage
  python make_final_csvs.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def section_number(base: str) -> str:
    field = base.split("#")[-2]
    return str(int(field)) if field.isdigit() else ""


def convert(src: Path, dst: Path, mask_names: set[str], mask_dir_name: str) -> pd.DataFrame:
    data = pd.read_csv(src, dtype=str, keep_default_na=False)
    has = (data["base"] + ".png").isin(mask_names)
    data["mask"] = [f"{mask_dir_name}/{b}.png" if h else "" for b, h in zip(data["base"], has)]
    data["has_mask"] = has.map({True: "True", False: "False"})
    data["section"] = data["base"].map(section_number)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(dst, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path,
                        help="directory holding csv/ and mask_final/")
    parser.add_argument("--out", default=None, type=Path, help="output folder (default <root>/csv_final)")
    args = parser.parse_args()

    root = args.root
    src_dir = root / "csv"
    out_dir = args.out or root / "csv_final"
    mask_dir = root / "mask_final"
    mask_names = {p.name for p in mask_dir.iterdir() if p.is_file()}

    master = convert(src_dir / "master.csv", out_dir / "master.csv", mask_names, mask_dir.name)
    for src in sorted((src_dir / "splits").glob("*.csv")):
        convert(src, out_dir / "splits" / src.name, mask_names, mask_dir.name)

    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    placed = master[master["section"] != ""]
    for (sub, hemi, chunk), group in placed.groupby(["sub", "hemisphere", "chunk"], sort=True):
        group.to_csv(chunk_dir / f"{sub}_{hemi}_{chunk}.csv", index=False)
        print(f"{sub}_{hemi}_{chunk}: {len(group)} sections, {(group['has_mask'] == 'True').sum()} with mask")

    print(f"master: {len(master)} rows, {(master['has_mask'] == 'True').sum()} with mask, "
          f"{(master['section'] == '').sum()} without section number (left out of chunks/)")


if __name__ == "__main__":
    main()
