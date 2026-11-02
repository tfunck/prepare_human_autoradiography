#!/usr/bin/env python3
r"""Rewrite a landmark CSV so its paths are relative to the project folder, and
optionally fill sections without labels from an earlier version's folder.

The CSVs written by the GUI before this fix lived inside the output folder,
with paths relative to it (``../preprocessed/...`` and bare label file names).
This writes the CSV into the project folder instead, like
``landmark_sections_v1.3.csv``:

    raw:    preprocessed/mr#rh#11530#L#musc#5767#01#L_crop.nii.gz
    labels: v1.4/mr#rh#11530#L#musc#5767#01#L_crop_landmark.nii.gz

    python repair_landmark_csv.py v1.4/landmark_sections_v1.4.csv
    python repair_landmark_csv.py v1.4/landmark_sections_v1.4.csv --fill-from v1.3

--fill-from DIR: for each section with no labels, use DIR/<image name>_landmark.nii.gz
if it exists, but only if none of its labels is already on another section of
the same specimen. The GUI writes *every* labelled section to the new folder,
so a blank means the section had no labels when it was saved. A label moved
with Shift+click leaves its old section blank, and filling that section from
the old version would put the label in two sections. Those are reported and
left blank (--no-check fills them anyway).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import landmark_core as core


def label_values(path: Path) -> set[int]:
    if core.is_nifti(path):
        arr = np.asanyarray(core.nib.load(str(path)).dataobj)
    else:
        arr = np.asarray(core.Image.open(path))
    v = np.unique(np.rint(np.nan_to_num(arr))).astype(np.int64)
    return {int(x) for x in v if x > 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV written by the GUI (inside its output folder)")
    ap.add_argument("--out", help="CSV to write (default: <folder containing the CSV's folder>/<same name>)")
    ap.add_argument("--fill-from", help="earlier label folder (e.g. v1.3) to fill sections without labels")
    ap.add_argument("--label-suffix", default=core.LABEL_SUFFIX)
    ap.add_argument("--no-check", action="store_true",
                    help="fill even when the labels are already on another section of the specimen")
    ap.add_argument("--overwrite", action="store_true", help="replace --out if it exists")
    args = ap.parse_args(argv)

    src = Path(args.csv).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else src.parent.parent / src.name
    if out.exists() and not args.overwrite and out != src:
        print(f"error: {out} exists (use --overwrite)", file=sys.stderr)
        return 1
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    cols = core.resolve_columns(df.columns)
    lab_col = cols.get("labels")
    if lab_col is None:
        print("error: no label column in the CSV", file=sys.stderr)
        return 1

    def rebase(v):
        p = core.resolve_path(v, src.parent)
        return "" if p is None else os.path.relpath(p, out.parent)

    raw_abs = [core.resolve_path(v, src.parent) for v in df[cols["raw"]]]
    lab_abs = [core.resolve_path(v, src.parent) for v in df[lab_col]]
    missing = [str(p) for p in lab_abs if p is not None and not p.exists()]
    df[cols["raw"]] = [rebase(v) for v in df[cols["raw"]]]
    df[lab_col] = [rebase(v) for v in df[lab_col]]

    filled, skipped = [], []
    if args.fill_from:
        fill_dir = Path(args.fill_from).expanduser()
        if not fill_dir.is_absolute():
            fill_dir = (out.parent / fill_dir) if (out.parent / fill_dir).is_dir() else fill_dir.resolve()
        spec = [core.specimen_id(s, h) for s, h in zip(df[cols["sub"]], df[cols["hemisphere"]])]
        used: dict[str, dict[int, str]] = {}            # specimen -> label -> section that has it
        if not args.no_check:
            for i, p in enumerate(lab_abs):
                if p is not None and p.exists():
                    for v in label_values(p):
                        used.setdefault(spec[i], {})[v] = str(df.iloc[i][cols["section"]])
        for i, p in enumerate(lab_abs):
            if p is not None:
                continue
            cand = fill_dir / f"{core.strip_ext(raw_abs[i])}{args.label_suffix}.nii.gz"
            if not cand.exists():
                continue
            where = f"{spec[i]} section {df.iloc[i][cols['section']]}"
            if not args.no_check:
                vals = label_values(cand)
                if not vals:
                    skipped.append(f"{where}: {cand.name} has no labels")
                    continue
                clash = {v: used.get(spec[i], {}).get(v) for v in vals if v in used.get(spec[i], {})}
                if clash:
                    skipped.append(f"{where}: label(s) " + ", ".join(f"{v} (now on section {s})"
                                                                      for v, s in sorted(clash.items()))
                                   + " already elsewhere — probably moved; left blank")
                    continue
                for v in vals:
                    used.setdefault(spec[i], {})[v] = str(df.iloc[i][cols["section"]])
            df.at[df.index[i], lab_col] = os.path.relpath(cand.resolve(), out.parent)
            filled.append(f"{where}: {os.path.relpath(cand.resolve(), out.parent)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {int((df[lab_col] != '').sum())} labelled sections; paths relative to {out.parent}")
    if missing:
        print(f"warning: {len(missing)} label file(s) listed in the CSV don't exist, e.g. {missing[0]}")
    if args.fill_from:
        print(f"Filled from {args.fill_from}: {len(filled)}")
        for f in filled:
            print("  +", f)
        if skipped:
            print(f"Not filled: {len(skipped)}")
            for s in skipped:
                print("  -", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
