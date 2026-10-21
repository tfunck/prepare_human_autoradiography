#!/usr/bin/env python3
"""Floating 3-D stack viewer for slab-level orientation QC.

Gathers every section of one slab (``sub`` + ``hemisphere`` + ``chunk``) from
``csv/master.csv``, applies each section's recorded QC flip/rotation, recentres
the tissue on its centre of mass and stacks the sections into a small 3-D
volume. The sections are ordered by repeat number (field ``[6]`` of the ``#``
separated base name) and then alphabetically by ligand, which reconstructs the
through-plane cutting order even though the ligand order within a repeat is not
encoded in the filename.

Two orthogonal cross-sections (axial and sagittal) are shown in a floating
window next to the main QC GUI, each with its own scrollbar, plus a scrollbar
to move through the sections. The section currently open in the main GUI is
drawn as a red line across both views so orientation mistakes stand out as a
break in the stack.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from common import apply_transform, normalize_flip, normalize_rotation, resolve_path

# Longest in-plane side of a scaled section, and the square canvas each centred
# section is dropped into (the margin absorbs the centre-of-mass shift).
THUMB_CONTENT = 128
THUMB_CANVAS = 192
BUILD_CHUNK = 24         # sections loaded per idle tick while building
VOLUME_CACHE = 3         # how many slab volumes to keep in memory

# Physical voxel geometry. The stored sections keep their coarse, downsampled
# in-plane sampling; only the orthogonal views stretch the in-plane axis so the
# stack shows the right proportions without resampling every section.
SLICE_THICKNESS_UM = 20  # through-plane spacing between consecutive sections
IN_PLANE_UM = 200        # in-plane spacing of the downsampled raw PNGs

ZOOM_STEP = 1.25         # multiplier applied per zoom step
ZOOM_MAX = 8.0           # deepest zoom for the orthogonal views

try:  # Pillow >= 9.1 moved the resampling filters onto Image.Resampling.
    NEAREST = Image.Resampling.NEAREST
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    NEAREST = Image.NEAREST
    LANCZOS = Image.LANCZOS


# --------------------------------------------------------------------------- #
# Ordering + geometry helpers (pure functions, GUI-free so they are testable)
# --------------------------------------------------------------------------- #
def parse_repeat(base: object) -> int:
    """Return the repeat number (``#``-field 6) of a base name.

    Unparseable repeats sort last but keep a stable relative order.
    """
    parts = str(base).split("#")
    if len(parts) > 6:
        match = re.search(r"\d+", parts[6])
        if match:
            return int(match.group())
    return 10**6


def slab_order(slab_df: pd.DataFrame) -> pd.DataFrame:
    """Order a slab's sections by (repeat, ligand, base) for stacking."""
    ordered = slab_df.copy()
    ordered["_repeat"] = ordered["base"].map(parse_repeat)
    ordered = ordered.sort_values(["_repeat", "ligand", "base"], kind="stable")
    return ordered.reset_index(drop=True)


def center_on_mass(content: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Drop ``content`` into a square canvas so its tissue centre sits centred.

    The centre of mass is taken from ``mask`` when one is given and non-empty,
    otherwise from the intensity-weighted centroid of ``content``.
    """
    canvas = np.zeros((THUMB_CANVAS, THUMB_CANVAS), dtype=np.uint8)
    height, width = content.shape
    if mask is not None and mask.any():
        rows, cols = np.nonzero(mask)
        center_r, center_c = rows.mean(), cols.mean()
    else:
        total = float(content.sum())
        if total > 0:
            grid_r, grid_c = np.mgrid[0:height, 0:width]
            center_r = float((grid_r * content).sum()) / total
            center_c = float((grid_c * content).sum()) / total
        else:
            center_r, center_c = height / 2.0, width / 2.0

    top = int(round(THUMB_CANVAS / 2.0 - center_r))
    left = int(round(THUMB_CANVAS / 2.0 - center_c))
    dst_r0, dst_c0 = max(0, top), max(0, left)
    dst_r1, dst_c1 = min(THUMB_CANVAS, top + height), min(THUMB_CANVAS, left + width)
    if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
        return canvas
    src_r0, src_c0 = dst_r0 - top, dst_c0 - left
    canvas[dst_r0:dst_r1, dst_c0:dst_c1] = content[
        src_r0:src_r0 + (dst_r1 - dst_r0), src_c0:src_c0 + (dst_c1 - dst_c0)
    ]
    return canvas


def make_thumb(
    gray_full: np.ndarray,
    mask_full: np.ndarray | None,
    flip: object,
    rotation: object,
) -> np.ndarray:
    """Transform, scale and tissue-centre one section into a stack slice."""
    gray = apply_transform(gray_full, flip, rotation)
    src_h, src_w = gray.shape
    scale = THUMB_CONTENT / max(src_h, src_w)
    out_h = max(1, int(round(src_h * scale)))
    out_w = max(1, int(round(src_w * scale)))
    content = np.asarray(Image.fromarray(gray).resize((out_w, out_h), LANCZOS), dtype=np.uint8)

    mask_small = None
    if mask_full is not None:
        mask_t = apply_transform(mask_full, flip, rotation)
        resized = Image.fromarray(mask_t).resize((out_w, out_h), NEAREST)
        candidate = np.asarray(resized) > 0
        if candidate.any():
            mask_small = candidate
    return center_on_mass(content, mask_small)


def inplane_stretch(slab_df: pd.DataFrame) -> float:
    """Factor to upsample the in-plane (coronal) axis in the orthogonal views.

    Each thumbnail pixel spans ``IN_PLANE_UM * native_longest / THUMB_CONTENT``
    microns in-plane; dividing by the through-plane slice thickness gives how
    much wider the coronal axis must be drawn so axial/sagittal pixels are
    roughly physically square.
    """
    native_longest = float(THUMB_CONTENT)
    if {"raw_h", "raw_w"}.issubset(slab_df.columns):
        heights = pd.to_numeric(slab_df["raw_h"], errors="coerce")
        widths = pd.to_numeric(slab_df["raw_w"], errors="coerce")
        longest = np.nanmax([heights.max(), widths.max()])
        if np.isfinite(longest) and longest > 0:
            native_longest = float(longest)
    thumb_inplane_um = IN_PLANE_UM * native_longest / THUMB_CONTENT
    return thumb_inplane_um / SLICE_THICKNESS_UM


# --------------------------------------------------------------------------- #
# Floating viewer
# --------------------------------------------------------------------------- #
class StackView(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        root_dir: Path,
        master_df: pd.DataFrame,
        splits_dir: Path,
        mask_qc_dir: Path,
        seek_callback=None,
    ) -> None:
        super().__init__(parent)
        self.title("Slab stack")
        self.root_dir = Path(root_dir)
        self.master_df = master_df
        self.splits_dir = Path(splits_dir)
        self.mask_qc_dir = Path(mask_qc_dir)
        self.seek_callback = seek_callback

        self.volumes: "OrderedDict[tuple, dict]" = OrderedDict()
        self.slab_key: tuple | None = None
        self.vol: np.ndarray | None = None
        self.order_bases: list[str] = []
        self.order_labels: list[str] = []
        self.base_to_z: dict[str, int] = {}

        self.cur_z: int | None = None
        self.cur_y = THUMB_CANVAS // 2
        self.cur_x = THUMB_CANVAS // 2
        self.inplane_stretch = IN_PLANE_UM / SLICE_THICKNESS_UM
        self.visible = False
        self._suppress_seek = False
        self._qc_cache: dict[tuple, dict] = {}
        self._build: dict | None = None
        self._build_after: str | None = None

        self.header_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.section_var = tk.StringVar()

        self.views: dict[str, dict] = {}
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self._build_ui()
        self.withdraw()

    # --- UI ----------------------------------------------------------------- #
    def _build_ui(self) -> None:
        self.geometry("620x700")
        self.minsize(460, 520)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(self, textvariable=self.header_var, font=("TkDefaultFont", 10, "bold"),
                  padding=(8, 4)).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, textvariable=self.progress_var, padding=(8, 0)).grid(
            row=1, column=0, columnspan=2, sticky="w")

        self.views["axial"] = self._make_view(
            "axial", "Axial \u2014 scroll: section \u00b7 wheel/\u00b1: zoom", column=0, scale_cmd=self._on_axial_scale)
        self.views["sagittal"] = self._make_view(
            "sagittal", "Sagittal \u2014 scroll: section \u00b7 wheel/\u00b1: zoom", column=1, scale_cmd=self._on_sagittal_scale)

        self.z_scale = tk.Scale(
            self, from_=0, to=0, orient="horizontal", label="Section (through-plane)",
            command=self._on_z, takefocus=False, highlightthickness=0,
        )
        self.z_scale.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8)

        ttk.Label(self, textvariable=self.section_var, padding=(8, 2)).grid(
            row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(self, text="Rebuild stack", command=self.rebuild).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

    def _make_view(self, name: str, title: str, column: int, scale_cmd) -> dict:
        frame = ttk.Frame(self, padding=(6, 2))
        frame.grid(row=2, column=column, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title).grid(row=0, column=0, sticky="w")
        canvas = tk.Canvas(frame, background="#101010", highlightthickness=0, width=260, height=440)
        canvas.grid(row=1, column=0, sticky="nsew")
        scale = tk.Scale(frame, from_=0, to=0, orient="horizontal",
                         command=scale_cmd, takefocus=False, highlightthickness=0)
        scale.grid(row=2, column=0, sticky="ew")
        view = {"canvas": canvas, "scale": scale, "photo": None, "zoom": 1.0,
                "img_id": None, "line_id": None, "ox": None, "oy": None, "ow": None, "oh": None}
        canvas.bind("<Configure>", lambda _e, n=name: self._render_view(n))
        canvas.bind("<MouseWheel>", lambda e, n=name: self._zoom_view(n, 1 if e.delta > 0 else -1))
        canvas.bind("<Button-4>", lambda e, n=name: self._zoom_view(n, 1))
        canvas.bind("<Button-5>", lambda e, n=name: self._zoom_view(n, -1))
        return view

    def show_adjacent(self, parent: tk.Misc) -> None:
        """Place the window just right of ``parent`` and reveal it."""
        self.deiconify()
        self.visible = True
        try:
            parent.update_idletasks()
            px, py = parent.winfo_rootx(), parent.winfo_rooty()
            pw = parent.winfo_width()
            x = px + pw + 8
            screen_w = self.winfo_screenwidth()
            if x + 480 > screen_w:
                x = max(0, screen_w - 480)
            self.geometry(f"+{x}+{py}")
        except Exception:
            pass
        self._render_all()

    def hide(self) -> None:
        self.visible = False
        self._cancel_build()
        self.withdraw()

    # --- slab loading / building ------------------------------------------- #
    def show_slab(self, slab_key) -> None:
        slab_key = tuple(str(part) for part in slab_key)
        if slab_key == self.slab_key and self.vol is not None and self._build is None:
            return
        cached = self.volumes.get(slab_key)
        if cached is not None:
            self._activate(slab_key, cached)
            return
        self._begin_build(slab_key)

    def rebuild(self) -> None:
        if self.slab_key is None:
            return
        self.volumes.pop(self.slab_key, None)
        self._qc_cache.clear()
        self._begin_build(self.slab_key)

    def _activate(self, slab_key: tuple, entry: dict) -> None:
        self.slab_key = slab_key
        self.vol = entry["vol"]
        self.order_bases = entry["order_bases"]
        self.order_labels = entry["order_labels"]
        self.base_to_z = entry["base_to_z"]
        self.inplane_stretch = entry.get("stretch", self.inplane_stretch)
        self._configure_scales()
        self._update_header()
        self.progress_var.set("")
        self._render_all()

    def _begin_build(self, slab_key: tuple) -> None:
        self._cancel_build()
        if self.master_df is None:
            self.progress_var.set("master.csv not available.")
            return
        sub, hemi, chunk = slab_key
        frame = self.master_df
        selection = frame[
            (frame["sub"].astype(str) == sub)
            & (frame["hemisphere"].astype(str) == hemi)
            & (frame["chunk"].astype(str) == chunk)
        ]
        if selection.empty:
            self.progress_var.set("No sections found for this slab.")
            return
        ordered = slab_order(selection)
        self.inplane_stretch = inplane_stretch(ordered)
        order_bases = ordered["base"].astype(str).tolist()
        order_labels = [
            f"{row.ligand} #{parse_repeat(row.base):02d}" for row in ordered.itertuples()
        ]
        nz = len(order_bases)
        vol = np.zeros((nz, THUMB_CANVAS, THUMB_CANVAS), dtype=np.uint8)

        self.slab_key = slab_key
        self.vol = vol
        self.order_bases = order_bases
        self.order_labels = order_labels
        self.base_to_z = {base: index for index, base in enumerate(order_bases)}
        self._configure_scales()
        self._update_header()

        self._build = {
            "slab_key": slab_key,
            "rows": ordered,
            "vol": vol,
            "index": 0,
            "qc": self._qc_lookup(slab_key, ordered),
        }
        self._build_after = self.after(1, self._build_step)

    def _cancel_build(self) -> None:
        if self._build_after is not None:
            self.after_cancel(self._build_after)
            self._build_after = None
        self._build = None

    def _qc_lookup(self, slab_key: tuple, ordered: pd.DataFrame) -> dict:
        """base -> (flip, rotation) from each ligand's sibling .qc.csv file."""
        sub, hemi, _chunk = slab_key
        lookup: dict[str, tuple] = {}
        for ligand in sorted(ordered["ligand"].astype(str).unique()):
            key = (sub, hemi, ligand)
            if key not in self._qc_cache:
                self._qc_cache[key] = self._read_qc_csv(sub, hemi, ligand)
            lookup.update(self._qc_cache[key])
        return lookup

    def _read_qc_csv(self, sub: str, hemi: str, ligand: str) -> dict:
        path = self.splits_dir / f"{sub}_{hemi}_{ligand}.qc.csv"
        result: dict[str, tuple] = {}
        if not path.exists():
            return result
        try:
            data = pd.read_csv(path, keep_default_na=False)
        except Exception:
            return result
        if "base" not in data.columns:
            return result
        for row in data.itertuples():
            flip = normalize_flip(getattr(row, "flip", ""))
            rotation = normalize_rotation(getattr(row, "rotation", 0))
            result[str(row.base)] = (flip, rotation)
        return result

    def _build_step(self) -> None:
        build = self._build
        if build is None:
            return
        rows = build["rows"]
        vol = build["vol"]
        qc = build["qc"]
        start = build["index"]
        stop = min(start + BUILD_CHUNK, len(rows))
        for position in range(start, stop):
            row = rows.iloc[position]
            base = str(row["base"])
            flip, rotation = qc.get(base, ("", 0))
            thumb = self._thumb_from_disk(row, flip, rotation)
            if thumb is not None:
                vol[position] = thumb
        build["index"] = stop
        self.progress_var.set(f"Building stack… {stop}/{len(rows)}")
        self._render_all()
        if stop < len(rows):
            self._build_after = self.after(1, self._build_step)
            return
        entry = {
            "vol": vol,
            "order_bases": self.order_bases,
            "order_labels": self.order_labels,
            "base_to_z": self.base_to_z,
            "stretch": self.inplane_stretch,
        }
        self.volumes[build["slab_key"]] = entry
        while len(self.volumes) > VOLUME_CACHE:
            self.volumes.popitem(last=False)
        self._build = None
        self._build_after = None
        self.progress_var.set("")
        self._render_all()

    def _thumb_from_disk(self, row: pd.Series, flip: object, rotation: object) -> np.ndarray | None:
        gray = self._load_gray(resolve_path(self.root_dir, row["raw"]))
        if gray is None:
            return None
        mask = self._load_mask(row)
        return make_thumb(gray, mask, flip, rotation)

    def _load_gray(self, path: Path) -> np.ndarray | None:
        try:
            with Image.open(path) as img:
                return np.asarray(img.convert("L"), dtype=np.uint8)
        except Exception:
            return None

    def _load_mask(self, row: pd.Series) -> np.ndarray | None:
        base = str(row["base"])
        edited = self.mask_qc_dir / f"{base}.png"
        candidates = [edited]
        mask_value = str(row.get("mask", "")).strip()
        if mask_value and str(row.get("has_mask", "")).strip().lower() in {"true", "1", "yes"}:
            candidates.append(resolve_path(self.root_dir, mask_value))
        for path in candidates:
            try:
                if path.exists():
                    with Image.open(path) as img:
                        return np.asarray(img.convert("L"), dtype=np.uint8)
            except Exception:
                continue
        return None

    # --- live update of the section currently being edited ----------------- #
    def update_slice(self, base: str, thumb: np.ndarray) -> None:
        if self.vol is None:
            return
        index = self.base_to_z.get(str(base))
        if index is None or thumb.shape != (THUMB_CANVAS, THUMB_CANVAS):
            return
        self.vol[index] = thumb
        self._render_all()

    # --- current-section highlight / navigation sync ----------------------- #
    def set_current(self, base: str) -> None:
        index = self.base_to_z.get(str(base))
        if index is None:
            self.cur_z = None
            self._update_section_label()
            self._render_lines()
            return
        self._set_z(index, seek=False)

    def _set_z(self, index: int, seek: bool) -> None:
        self.cur_z = index
        self._sync_scales(index)
        self._update_section_label()
        self._refresh_section_views()
        if seek and self.seek_callback and 0 <= index < len(self.order_bases):
            self.seek_callback(self.order_bases[index])

    def _sync_scales(self, index: int) -> None:
        """Keep the section, axial and sagittal scrollbars on the same value."""
        self._suppress_seek = True
        self.z_scale.set(index)
        self.views["axial"]["scale"].set(index)
        self.views["sagittal"]["scale"].set(index)
        self._suppress_seek = False

    def _on_z(self, _value: str | None = None) -> None:
        if self._suppress_seek or not self.order_bases:
            return
        self._set_z(int(self.z_scale.get()), seek=True)

    def _on_axial_scale(self, _value: str | None = None) -> None:
        if self._suppress_seek or not self.order_bases:
            return
        self._set_z(int(self.views["axial"]["scale"].get()), seek=True)

    def _on_sagittal_scale(self, _value: str | None = None) -> None:
        if self._suppress_seek or not self.order_bases:
            return
        self._set_z(int(self.views["sagittal"]["scale"].get()), seek=True)

    # --- zoom --------------------------------------------------------------- #
    def zoom_all(self, direction: int) -> None:
        for name in self.views:
            self._zoom_view(name, direction)

    def _zoom_view(self, name: str, direction: int) -> None:
        view = self.views.get(name)
        if view is None:
            return
        factor = ZOOM_STEP if direction > 0 else 1.0 / ZOOM_STEP
        view["zoom"] = float(min(ZOOM_MAX, max(1.0, view["zoom"] * factor)))
        self._render_view(name)

    def _refresh_section_views(self) -> None:
        """Zoomed views follow the current section, so re-place them; views shown
        at fit scale only need the marker line moved."""
        for name, view in self.views.items():
            if view["zoom"] > 1.0:
                self._render_view(name)
            else:
                self._draw_line(name)

    # --- rendering ---------------------------------------------------------- #
    def _configure_scales(self) -> None:
        nz = 0 if self.vol is None else self.vol.shape[0]
        top = max(0, nz - 1)
        self._suppress_seek = True
        self.z_scale.configure(to=top)
        self.views["axial"]["scale"].configure(to=top)
        self.views["sagittal"]["scale"].configure(to=top)
        self._suppress_seek = False

    def _update_header(self) -> None:
        if self.slab_key is None:
            self.header_var.set("")
            return
        sub, hemi, chunk = self.slab_key
        nz = 0 if self.vol is None else self.vol.shape[0]
        self.header_var.set(f"Slab {sub} {hemi} chunk {chunk} — {nz} sections")

    def _update_section_label(self) -> None:
        if self.cur_z is None or not self.order_labels:
            self.section_var.set("Current section not in this slab")
            return
        nz = len(self.order_labels)
        label = self.order_labels[self.cur_z] if self.cur_z < nz else "?"
        self.section_var.set(f"Section {self.cur_z + 1}/{nz}  —  {label}")

    def _slice_for(self, which: str) -> np.ndarray | None:
        if self.vol is None or self.vol.shape[0] == 0:
            return None
        if which == "axial":
            row = min(max(self.cur_y, 0), self.vol.shape[1] - 1)
            return self.vol[:, row, :]
        column = min(max(self.cur_x, 0), self.vol.shape[2] - 1)
        return self.vol[:, :, column]

    def _vertical_offset(self, canvas_h: int, out_h: int) -> int:
        """Centre the view when it fits; when zoomed, keep the current section's
        marker line centred, clamped so the stack never scrolls past its ends."""
        if out_h <= canvas_h or self.cur_z is None or self.vol is None:
            return (canvas_h - out_h) // 2
        nz = max(self.vol.shape[0], 1)
        focus = (self.cur_z + 0.5) / nz * out_h
        offset = int(round(canvas_h / 2.0 - focus))
        return max(canvas_h - out_h, min(0, offset))

    def _render_all(self) -> None:
        self._render_view("axial")
        self._render_view("sagittal")

    def _render_view(self, which: str) -> None:
        view = self.views.get(which)
        if view is None:
            return
        canvas = view["canvas"]
        slice2d = self._slice_for(which)
        if slice2d is None:
            return
        img = Image.fromarray(np.ascontiguousarray(slice2d), "L")
        # Width is the in-plane (coronal) axis and is far coarser than the
        # through-plane step, so stretch it to restore anatomical proportions.
        eff_w = img.width * max(self.inplane_stretch, 1e-6)
        eff_h = float(img.height)
        canvas_w = max(canvas.winfo_width(), 10)
        canvas_h = max(canvas.winfo_height(), 10)
        fit = min(canvas_w / eff_w, canvas_h / eff_h)
        scale = fit * view["zoom"]
        out_w = max(1, int(round(eff_w * scale)))
        out_h = max(1, int(round(eff_h * scale)))
        pil = img.convert("RGB").resize((out_w, out_h), LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        view["photo"] = photo
        offset_x = (canvas_w - out_w) // 2
        offset_y = self._vertical_offset(canvas_h, out_h)
        if view["img_id"] is None:
            view["img_id"] = canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)
        else:
            canvas.itemconfigure(view["img_id"], image=photo)
            canvas.coords(view["img_id"], offset_x, offset_y)
        view["ox"], view["oy"], view["ow"], view["oh"] = offset_x, offset_y, out_w, out_h
        self._draw_line(which)

    def _render_lines(self) -> None:
        self._draw_line("axial")
        self._draw_line("sagittal")

    def _draw_line(self, which: str) -> None:
        view = self.views.get(which)
        if view is None or view["oh"] is None:
            return
        canvas = view["canvas"]
        if self.cur_z is None or self.vol is None or self.vol.shape[0] == 0:
            if view["line_id"] is not None:
                canvas.itemconfigure(view["line_id"], state="hidden")
            return
        nz = self.vol.shape[0]
        line_y = view["oy"] + int((self.cur_z + 0.5) * view["oh"] / max(nz, 1))
        x0 = view["ox"]
        x1 = view["ox"] + view["ow"]
        if view["line_id"] is None:
            view["line_id"] = canvas.create_line(x0, line_y, x1, line_y, fill="#ff3030", width=1)
        else:
            canvas.coords(view["line_id"], x0, line_y, x1, line_y)
            canvas.itemconfigure(view["line_id"], state="normal")
        canvas.tag_raise(view["line_id"])
