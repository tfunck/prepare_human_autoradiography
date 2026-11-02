#!/usr/bin/env python3
"""Synthetic donor MRI + coronal slab sections with a known slab->MRI mapping.

    python make_synthetic_slab.py demo_slab            # rigid-ish + nonlinear bump
    python make_synthetic_slab.py demo_slab --mirror   # slab mirror-imaged vs the MRI
    python make_synthetic_slab.py demo_slab --stacked  # like master.csv (see below)

--stacked writes a CSV like the human master.csv instead: PNG sections named
OL#HG#MR9S6#L#<ligand>#<id>#<repeat>#L, section column = repeat number (shared
by all ligands of a repeat; blank for "UB" sections), tissue masks in
mask_final/ (mask, has_mask columns; ~8% without a mask, two at half
resolution) and every section shifted by a random in-plane offset. The true
stacking is stack_view.py's: (repeat, ligand, name) order, one slice thickness
apart, mask centres of mass aligned (truth.npz: c_mask = the mask centre in the
true frame).

Writes:
  mri.nii.gz                      donor MRI (0.8 mm, RAS)
  labels_in/mri_landmarks_v1.0.nii.gz
  sections/*.nii.gz               2-D sections (0.15 mm pixels), slab 6 (30) and slab 7 (5)
  labels_in/*_landmark.nii.gz     existing 2-D labels (14 landmarks in slab 6, 2 in slab 7)
  sections.csv                    sub, hemisphere, chunk, section, ligand, raw, labels
  truth.npz                       parameters of the true mapping (see true_map)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

VOX = 0.8
MRI_SHAPE = (150, 176, 140)
PIX = 0.15
NX, NZ = 560, 440                 # 84 x 66 mm sections
THICK = 0.02


def mri_affine():
    a = np.diag([VOX, VOX, VOX, 1.0])
    a[:3, 3] = -np.array(MRI_SHAPE) * VOX / 2
    return a


def true_map(frame_pts, truth):
    """slab frame (mm) -> MRI LPS (mm)."""
    p = np.atleast_2d(frame_pts).astype(float)
    m = np.diag([-1.0 if truth["mirror"] else 1.0, 1.0, 1.0])
    q = (p - truth["c_f"]) @ (truth["R"] @ m).T + truth["c_m"]
    if truth["bump"] > 0:
        d2 = ((p - truth["bump_c"]) ** 2).sum(1)
        q += truth["bump"] * np.exp(-d2 / (2 * truth["bump_sigma"] ** 2))[:, None] * truth["bump_dir"]
    return q


def lps_to_vox(lps):
    ras = np.atleast_2d(lps) * np.array([-1, -1, 1.0])
    return (np.linalg.inv(mri_affine()) @ np.c_[ras, np.ones(len(ras))].T).T[:, :3]


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_mri(rng):
    g = np.indices(MRI_SHAPE).astype(np.float32)
    c = (np.array(MRI_SHAPE) - 1) / 2
    r = np.sqrt(sum(((g[k] - c[k]) / (0.46 * MRI_SHAPE[k])) ** 2 for k in range(3)))
    brain = r < 1
    tex = ndimage.gaussian_filter(rng.normal(size=MRI_SHAPE), 4).astype(np.float32)
    tex = (tex - tex.min()) / np.ptp(tex)
    vol = np.where(brain, 60 + 120 * tex, 0).astype(np.float32)
    vol[(r > 0.85) & brain] += 40                                         # "cortex"
    vent = (((g[0] - c[0]) / 6) ** 2 + ((g[1] - c[1]) / 30) ** 2 + ((g[2] - c[2] - 8) / 10) ** 2) < 1
    vol[vent] = 20
    return vol


def stamp(vol, center, r, value):
    c = np.round(center).astype(int)
    lo = np.maximum(c - r, 0)
    hi = np.minimum(c + r + 1, vol.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    grids = np.ogrid[sl]
    d2 = sum((gg - cc) ** 2 for gg, cc in zip(grids, center))
    vol[sl][d2 <= r * r + 1e-6] = value


LIGANDS = ("ampa", "cgp5", "damp", "dpat", "dpmg", "flum", "kain", "keta", "mk80", "musc", "oxot", "pire",
           "praz", "sch2", "sr95", "uk14")
S_PIX = 0.3                       # --stacked: PNG pixel size (pass --pixel-size 0.3)
S_NX, S_NZ = 280, 220             # 84 x 66 mm sections


def main_stacked(args, rng, out, vol):
    """master.csv-like slab (see the module docstring)."""
    (out / "raw").mkdir(exist_ok=True)
    (out / "mask_final").mkdir(exist_ok=True)
    ex, ez = S_NX * S_PIX, S_NZ * S_PIX
    slabs = {}
    for chunk, n_rep, ligs in (("6", 25, LIGANDS), ("7", 2, LIGANDS[:3])):
        names = []
        for rep in range(1, n_rep + 1):
            for k, lig in enumerate(ligs):
                if chunk == "6" and rng.random() < 0.05:          # some ligands missing in some repeats
                    continue
                names.append((lig, f"OL#HG#MR9S{chunk}#L#{lig}#{9000 + 37 * rep + k}#{rep:02d}#L", rep))
        if chunk == "6":
            for k, lig in enumerate(("musc", "ampa", "kain")):    # unnumbered "UB" sections
                names.append((lig, f"OL#HG#MR9S{chunk}#L#{lig}#{8800 + k}#UB#L", None))
        # true order = stack_view.py's: (repeat, ligand, name); UB last
        names.sort(key=lambda t: (t[2] if t[2] is not None else 10 ** 6, t[0], t[1]))
        slabs[chunk] = names
    n6 = len(slabs["6"])
    y_mid = (n6 - 1) / 2 * THICK
    c_mask = np.array([ex / 2 + 3.0, ez / 2 - 2.0])             # mask ellipse centre (true frame x, z)
    axes = np.array([25.0, 20.0])
    truth = dict(mirror=args.mirror, R=rot(0.12, -0.08, 0.15), c_f=np.array([ex / 2, y_mid, ez / 2]),
                 c_m=np.array([4.0, 6.0, 3.0]), bump=args.bump, bump_c=np.array([ex * 0.3, y_mid, ez * 0.6]),
                 bump_sigma=12.0, bump_dir=np.array([0.6, 0.3, -0.75]), c_mask=c_mask, stacked=True,
                 pix=S_PIX)
    np.savez(out / "truth.npz", **truth)

    def grid(shape, d, pix):
        nx, nz = shape
        i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        return d[0] + (i + 0.5) * pix, d[1] + (j + 0.5) * pix

    def to_png(a):           # (x, z) array -> raster: row 0 at the top = highest z; columns = x
        return np.flipud(a.T)

    rows, sec = [], {}
    for chunk, names in slabs.items():
        for rank, (lig, base, rep) in enumerate(names):
            d = rng.uniform([-6.0, -5.0], [6.0, 5.0])                # where the image window sits in the frame
            x, z = grid((S_NX, S_NZ), d, S_PIX)
            pts = np.stack([x, np.full(x.shape, rank * THICK), z], -1).reshape(-1, 3)
            img = ndimage.map_coordinates(vol, lps_to_vox(true_map(pts, truth)).T, order=1, cval=0)
            img = 255 * (img.reshape(x.shape) / 220.0).clip(0, 1) ** (1.0 + 0.04 * LIGANDS.index(lig))
            img = (img + rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)
            Image.fromarray(to_png(img)).save(out / "raw" / f"{base}.png")
            has_mask = chunk == "7" or rank % 13 != 5                # ~8% without a mask
            half = has_mask and rank in (7, 40)
            if has_mask:
                mshape, mpix = ((S_NX // 2, S_NZ // 2), 2 * S_PIX) if half else ((S_NX, S_NZ), S_PIX)
                mx, mz = grid(mshape, d, mpix)
                m = ((((mx - c_mask[0]) / axes[0]) ** 2 + ((mz - c_mask[1]) / axes[1]) ** 2) < 1)
                Image.fromarray(to_png(m.astype(np.uint8) * 255)).save(out / "mask_final" / f"{base}.png")
            rows.append(dict(base=base, sub="MR9", hemisphere="L", chunk=chunk, ligand=lig,
                             raw=f"raw/{base}.png", mask=f"mask_final/{base}.png" if has_mask else "",
                             qc=f"qc/{base}.png", has_mask=has_mask, raw_h=S_NZ, raw_w=S_NX,
                             section="" if rep is None else rep, labels="", _rank=rank, _d=d))
            sec[(chunk, rank)] = rows[-1]

    # Landmarks (on sections with a full-resolution mask, inside the mask): 22 in slab 6, 2 in slab 7.
    mri_lab = np.zeros(MRI_SHAPE, np.int16)
    sec_labels: dict[str, np.ndarray] = {}

    def place(label, chunk, on_section):
        cands = [r for (c, _k), r in sec.items() if c == chunk and r["has_mask"] and r["_rank"] not in (7, 40)
                 and r["section"] != ""]
        r = cands[rng.integers(len(cands))]
        f = c_mask + rng.uniform(-0.6, 0.6, 2) * axes / np.sqrt(2)
        frame = np.array([f[0], r["_rank"] * THICK, f[1]])
        stamp(mri_lab, lps_to_vox(true_map(frame, truth))[0], 1, label)
        if on_section:
            i, j = (f - r["_d"]) / S_PIX - 0.5                       # (x, z) pixel of the landmark
            arr = sec_labels.setdefault(r["raw"], np.zeros((S_NX, S_NZ), np.int32))
            stamp(arr, np.array([i, j]), 3, label)
    for lab in range(1, 23):
        place(lab, "6", on_section=lab <= args.preplaced)
    for lab in (30, 31):
        place(lab, "7", on_section=True)
    nib.save(nib.Nifti1Image(mri_lab, mri_affine()), str(out / "labels_in" / "mri_landmarks_v1.0.nii.gz"))
    for raw, arr in sec_labels.items():
        name = Path(raw).name.replace(".png", "_landmark.png")
        Image.fromarray(to_png(arr).astype(np.uint16)).save(out / "labels_in" / name)
        next(r for r in rows if r["raw"] == raw)["labels"] = f"labels_in/{name}"
    df = pd.DataFrame(rows)
    df["true_rank"] = df.pop("_rank")
    d = np.stack(df.pop("_d").to_numpy())
    df["true_dx"], df["true_dz"] = d[:, 0], d[:, 1]                   # image window position in the true frame
    df = df.sample(frac=1.0, random_state=1)                          # CSV row order means nothing
    df.to_csv(out / "sections.csv", index=False)
    print(f"Wrote {len(df)} master.csv-like PNG sections (slab 6: {n6}), {len(np.unique(mri_lab)) - 1} MRI "
          f"landmarks to {out}; open with --pixel-size {S_PIX}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--mirror", action="store_true", help="slab frame mirror-imaged relative to the MRI")
    ap.add_argument("--bump", type=float, default=3.0, help="nonlinear displacement amplitude (mm)")
    ap.add_argument("--conflict", action="store_true", help="also use label 5 in slab 7 (cross-slab error)")
    ap.add_argument("--preplaced", type=int, default=14,
                    help="how many of the 22 slab-6 landmarks are already on the sections")
    ap.add_argument("--png", action="store_true",
                    help="write sections as 8-bit PNG and labels as 16-bit PNG (no header: use --pixel-size 0.15)")
    ap.add_argument("--stacked", action="store_true",
                    help="master.csv-like slab: names with repeat numbers, masks, random in-plane shifts")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir).resolve()
    (out / "labels_in").mkdir(parents=True, exist_ok=True)

    vol = make_mri(rng)
    nib.save(nib.Nifti1Image(vol, mri_affine()), str(out / "mri.nii.gz"))
    if args.stacked:
        return main_stacked(args, rng, out, vol)
    (out / "sections").mkdir(parents=True, exist_ok=True)

    ex, ez = NX * PIX, NZ * PIX
    nums6 = sorted([2000 + 40 * k for k in range(15)] + [2007 + 40 * k for k in range(15)])
    nums7 = [2700 + 50 * k for k in range(5)]
    y_mid = np.mean(nums6) * THICK
    truth = dict(mirror=args.mirror, R=rot(0.12, -0.08, 0.15), c_f=np.array([ex / 2, y_mid, ez / 2]),
                 c_m=np.array([4.0, 6.0, 3.0]), bump=args.bump, bump_c=np.array([ex * 0.3, y_mid, ez * 0.6]),
                 bump_sigma=12.0, bump_dir=np.array([0.6, 0.3, -0.75]))
    np.savez(out / "truth.npz", **truth)

    def section_image(number, shape, ligand):
        nx, nz = shape
        i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        ox, oz = (ex - nx * PIX) / 2, (ez - nz * PIX) / 2
        pts = np.stack([ox + (i + 0.5) * PIX, np.full(i.shape, number * THICK), oz + (j + 0.5) * PIX], -1)
        vox = lps_to_vox(true_map(pts.reshape(-1, 3), truth)).T
        img = ndimage.map_coordinates(vol, vox, order=1, cval=0).reshape(shape)
        gamma = {"ampa": 1.0, "musc": 1.6}[ligand]
        img = 255 * (img / 220.0).clip(0, 1) ** gamma
        return (img + rng.normal(0, 6, img.shape)).clip(0, None).astype(np.float32)

    rows = []
    sec_aff = np.diag([PIX, PIX, 1.0, 1.0])
    for chunk, nums in (("6", nums6), ("7", nums7)):
        for k, num in enumerate(nums):
            lig = "ampa" if num % 10 == 0 else "musc"
            shape = (NX - 40, NZ - 30) if (chunk == "6" and k in (4, 11)) else (NX, NZ)   # a few cropped sections
            root = f"HG#MR9S{chunk}#L#{lig}#{num}"
            img = section_image(num, shape, lig)
            if args.png:   # raster: row 0 at the top = highest z; columns = x
                p = out / "sections" / f"{root}.png"
                Image.fromarray(np.clip(np.flipud(img.T), 0, 255).astype(np.uint8)).save(p)
            else:
                p = out / "sections" / f"{root}.nii.gz"
                nib.save(nib.Nifti1Image(img, sec_aff), str(p))
            rows.append(dict(sub="MR9", hemisphere="L", chunk=chunk, section=num, ligand=lig,
                             raw=f"sections/{p.name}", labels="", _shape=shape))

    # Landmarks: 22 in slab 6 (14 already on the sections), 2 in slab 7.
    mri_lab = np.zeros(MRI_SHAPE, np.int16)
    sec_labels: dict[str, np.ndarray] = {}
    def place(label, chunk, nums, on_section):
        num = nums[rng.integers(len(nums))]
        r = next(r for r in rows if r["chunk"] == chunk and r["section"] == num)
        nx, nz = r["_shape"]
        i, j = rng.uniform(0.15 * nx, 0.85 * nx), rng.uniform(0.15 * nz, 0.85 * nz)
        ox, oz = (ex - nx * PIX) / 2, (ez - nz * PIX) / 2
        frame = np.array([ox + (i + 0.5) * PIX, num * THICK, oz + (j + 0.5) * PIX])
        stamp(mri_lab, lps_to_vox(true_map(frame, truth))[0], 1, label)
        if on_section:
            arr = sec_labels.setdefault(r["raw"], np.zeros(r["_shape"], np.int32))
            stamp(arr, np.array([i, j]), 4, label)
    for lab in range(1, 23):
        place(lab, "6", nums6, on_section=lab <= args.preplaced)
    for lab in (30, 31):
        place(lab, "7", nums7, on_section=True)
    if args.conflict:
        r = next(r for r in rows if r["chunk"] == "7")
        arr = sec_labels.setdefault(r["raw"], np.zeros(r["_shape"], np.int32))
        stamp(arr, np.array([100.0, 100.0]), 4, 5)
    nib.save(nib.Nifti1Image(mri_lab, mri_affine()), str(out / "labels_in" / "mri_landmarks_v1.0.nii.gz"))
    for raw, arr in sec_labels.items():
        if args.png:
            name = Path(raw).name.replace(".png", "_landmark.png")
            Image.fromarray(np.flipud(arr.T).astype(np.uint16)).save(out / "labels_in" / name)
        else:
            name = Path(raw).name.replace(".nii.gz", "_landmark.nii.gz")
            nib.save(nib.Nifti1Image(arr, sec_aff), str(out / "labels_in" / name))
        next(r for r in rows if r["raw"] == raw)["labels"] = f"labels_in/{name}"
    df = pd.DataFrame(rows).drop(columns=["_shape"])
    df.to_csv(out / "sections.csv", index=False)
    print(f"Wrote {len(df)} sections (slabs 6 and 7), {len(np.unique(mri_lab)) - 1} MRI landmarks to {out}")


if __name__ == "__main__":
    main()
