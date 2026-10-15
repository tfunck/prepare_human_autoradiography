#!/usr/bin/env python3
"""GUI to QC raw autoradiograph images and their crop-mask overlays.

Reads one of the ``csv/splits/<sub>_<hemi>_<ligand>.csv`` files produced by
``prepare_qc_data.py``. Each row points at a raw PNG and (optionally) a crop
mask PNG. The raw image is shown with its mask painted on top as a
translucent red overlay. If a row has no mask (``has_mask`` is False or the
file is missing) an empty overlay is created so the reviewer can paint one.

Features
  * Left/Right (or Up/Down) arrows move between sections.
  * Rotate the image + overlay in 90 degree steps.
  * Flip the image + overlay left/right or up/down.
  * Paint (left drag) and erase (right drag) on the overlay; a Clear button
    wipes the whole overlay for the current section.
  * The flip/rotation applied to each section is recorded to an output CSV,
    and the edited overlay is written to a mask output directory.

Exactly like ``rotate_gui.py``, the raw image and the saved mask on disk are
kept in their original orientation -- only the recorded flip/rotation
metadata changes. Painting is mapped back through the current transform so
the saved mask always stays aligned with the raw PNG on disk.

Usage
  python qc_gui.py csv/splits/MR1_L_afdx.csv
  python qc_gui.py            # launches a file picker to choose the CSV
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw

from common import (
    apply_transform,
    decode_flip,
    encode_flip,
    normalize_flip,
    normalize_rotation,
    rel_path,
    resolve_path,
)

WORK_MAX = 1600          # longest side of the in-memory working image
MASK_THRESHOLD = 0       # mask foreground = value > threshold
OVERLAY_ALPHA = 110      # 0..255 opacity of the red overlay
MAX_CACHE_ITEMS = 4
SAVE_DEBOUNCE_MS = 400
DEFAULT_BRUSH = 14       # brush radius in on-screen pixels
MIN_BRUSH = 2
MAX_BRUSH = 120

try:  # Pillow >= 9.1 moved the resampling filters onto Image.Resampling.
    NEAREST = Image.Resampling.NEAREST
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    NEAREST = Image.NEAREST
    LANCZOS = Image.LANCZOS

NEW_COLUMNS = (("flip", ""), ("rotation", 0), ("qc_mask", ""), ("qc_reviewed", False))


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_gray_work(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load a PNG as a 2D uint8 array downsampled to WORK_MAX.

    Returns ``(working_array, (native_height, native_width))`` so the edited
    mask can later be written back at the raw image's full resolution.
    """
    with Image.open(path) as img:
        img = img.convert("L")
        native_hw = (img.height, img.width)
        scale = min(WORK_MAX / max(img.width, img.height), 1.0)
        if scale < 1.0:
            size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(size, LANCZOS)
        return np.asarray(img, dtype=np.uint8).copy(), native_hw


def load_mask_work(path: Path, work_hw: tuple[int, int]) -> np.ndarray:
    """Load a mask PNG as a 0/255 uint8 array at the working resolution."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))
    binary = ((arr > MASK_THRESHOLD).astype(np.uint8)) * 255
    resized = Image.fromarray(binary).resize((work_hw[1], work_hw[0]), NEAREST)
    return np.asarray(resized, dtype=np.uint8).copy()


def invert_transform(data: np.ndarray, flip_value: object, rotation_steps: object) -> np.ndarray:
    """Inverse of ``common.apply_transform`` (undo rotation, then flips)."""
    lr_flip, up_flip = decode_flip(flip_value)
    steps = normalize_rotation(rotation_steps)
    out = data
    if steps:
        out = np.rot90(out, k=-steps)
    if up_flip:
        out = np.flipud(out)
    if lr_flip:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def find_root(csv_path: Path, override: str | None) -> Path:
    """Locate the directory that holds the raw/ and mask/ folders."""
    if override:
        return Path(override).expanduser().resolve()
    for parent in [csv_path.parent, *csv_path.parents]:
        if (parent / "raw").is_dir():
            return parent
    return Path.cwd()


def prepare_records(csv_path: Path, out_csv: Path) -> pd.DataFrame:
    """Read the split CSV, add QC columns, and resume from any prior output."""
    records = pd.read_csv(csv_path, keep_default_na=False)
    for column, default in NEW_COLUMNS:
        if column not in records.columns:
            records[column] = default

    if out_csv.exists():
        prior = pd.read_csv(out_csv, keep_default_na=False)
        if "base" in prior.columns:
            prior = prior.drop_duplicates("base", keep="last").set_index("base")
            for row_index, base in records["base"].items():
                if base not in prior.index:
                    continue
                saved = prior.loc[base]
                for column, _ in NEW_COLUMNS:
                    if column in prior.columns:
                        records.at[row_index, column] = saved[column]

    records["flip"] = records["flip"].apply(normalize_flip)
    records["rotation"] = records["rotation"].apply(normalize_rotation).astype(int)
    records["qc_mask"] = records["qc_mask"].fillna("").astype(str)
    records["qc_reviewed"] = records["qc_reviewed"].apply(truthy)
    return records.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class QcGui(tk.Tk):
    def __init__(
        self,
        csv_path: Path | None,
        root_override: str | None = None,
        out_csv_override: str | None = None,
        mask_dir_override: str | None = None,
    ) -> None:
        super().__init__()
        self.title("Autoradiograph QC GUI")
        self.withdraw()  # stay hidden until we know we have a valid CSV to show
        self.aborted = False

        resolved = self.resolve_inputs(csv_path, root_override, out_csv_override, mask_dir_override)
        if resolved is None:
            self.aborted = True
            return
        self.root_dir, self.records, self.out_csv, self.mask_out_dir = resolved
        self.mask_out_dir.mkdir(parents=True, exist_ok=True)

        self.index = 0
        self.cache: OrderedDict[int, tuple[np.ndarray, tuple[int, int]]] = OrderedDict()
        self.base_img: np.ndarray | None = None
        self.base_mask_img: Image.Image | None = None
        self.base_index: np.ndarray | None = None
        self.native_hw: tuple[int, int] = (0, 0)

        self.flip = ""
        self.rotation = 0
        self.erase_mode = False
        self.brush = DEFAULT_BRUSH
        self.overlay_alpha = OVERLAY_ALPHA
        self.contrast_min = 0
        self.contrast_max = 255
        self.mask_dirty = False
        self.mask_source = ""
        self.last_pt: tuple[int, int] | None = None

        self.disp_index: np.ndarray | None = None
        self.disp_shape: tuple[int, int] = (0, 0)
        self.display_scale = 1.0
        self.display_offset: tuple[int, int] = (0, 0)
        self.bg_photo: ImageTk.PhotoImage | None = None
        self.ov_photo: ImageTk.PhotoImage | None = None
        self.bg_id: int | None = None
        self.ov_id: int | None = None
        self.cursor_id: int | None = None
        self.resize_after_id: str | None = None
        self.pending_save_id: str | None = None
        self.mask_save_after_id: str | None = None

        self.position_var = tk.StringVar()
        self.file_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.brush_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.mask_src_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.title("Autoradiograph QC GUI")
        self.geometry("1500x950")
        self.minsize(1000, 700)
        self.deiconify()

        self.build_ui()
        self.bind_all("<Key>", self.on_key)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_current()
        self.after(100, self.focus_force)

    def resolve_inputs(
        self,
        csv_path: Path | None,
        root_override: str | None,
        out_csv_override: str | None,
        mask_dir_override: str | None,
    ) -> tuple[Path, pd.DataFrame, Path, Path] | None:
        """Pick a CSV (file dialog if needed) and resolve all paths/records.

        Returns ``(root_dir, records, out_csv, mask_out_dir)`` or ``None`` if the
        user cancelled or the CSV is unusable (an error dialog is shown).
        """
        if csv_path is None:
            chosen = filedialog.askopenfilename(
                parent=self,
                title="Select a split CSV to review",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not chosen:
                return None
            csv_path = Path(chosen).expanduser().resolve()
        if not csv_path.exists():
            messagebox.showerror("Autoradiograph QC GUI", f"CSV not found:\n{csv_path}", parent=self)
            return None
        root_dir = find_root(csv_path, root_override)
        out_csv = Path(out_csv_override).expanduser().resolve() if out_csv_override else csv_path.with_suffix(".qc.csv")
        mask_out_dir = Path(mask_dir_override).expanduser().resolve() if mask_dir_override else root_dir / "mask_qc"
        try:
            records = prepare_records(csv_path, out_csv)
        except Exception as exc:  # malformed/unreadable CSV -- report instead of crashing
            messagebox.showerror("Autoradiograph QC GUI", f"Could not read CSV:\n{csv_path}\n\n{exc}", parent=self)
            return None
        if records.empty:
            messagebox.showerror("Autoradiograph QC GUI", f"No rows found in:\n{csv_path}", parent=self)
            return None
        return root_dir, records, out_csv, mask_out_dir

    # --- UI construction ---------------------------------------------------- #
    def build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky="ew")

        ttk.Button(toolbar, text="< Prev", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(toolbar, text="Next >", command=lambda: self.move(1)).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Rotate -90", command=lambda: self.rotate(-1)).pack(side="left")
        ttk.Button(toolbar, text="Rotate +90", command=lambda: self.rotate(1)).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Flip H", command=self.flip_h).pack(side="left")
        ttk.Button(toolbar, text="Flip V", command=self.flip_v).pack(side="left", padx=(2, 10))

        self.mode_button = ttk.Button(toolbar, text="Mode: Draw", command=self.toggle_erase)
        self.mode_button.pack(side="left")
        ttk.Button(toolbar, text="Clear mask", command=self.clear_mask).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Brush -", command=lambda: self.change_brush(-2)).pack(side="left")
        ttk.Label(toolbar, textvariable=self.brush_var, width=10).pack(side="left")
        ttk.Button(toolbar, text="Brush +", command=lambda: self.change_brush(2)).pack(side="left", padx=(0, 10))

        ttk.Button(toolbar, text="Save", command=self.save_now).pack(side="left")

        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        controls = ttk.Frame(self, padding=(8, 2))
        controls.grid(row=1, column=0, sticky="ew")
        self.alpha_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=180,
            label="Overlay opacity", command=self.on_alpha_change,
            takefocus=False, highlightthickness=0,
        )
        self.alpha_scale.pack(side="left", padx=(0, 14))
        self.min_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=220,
            label="Contrast min", command=self.on_contrast_change,
            takefocus=False, highlightthickness=0,
        )
        self.min_scale.pack(side="left", padx=(0, 8))
        self.max_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=220,
            label="Contrast max", command=self.on_contrast_change,
            takefocus=False, highlightthickness=0,
        )
        self.max_scale.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Auto", command=self.auto_contrast).pack(side="left", padx=(4, 4))
        ttk.Button(controls, text="Reset", command=self.reset_contrast).pack(side="left")
        self.alpha_scale.set(self.overlay_alpha)
        self.min_scale.set(self.contrast_min)
        self.max_scale.set(self.contrast_max)

        self.canvas = tk.Canvas(self, background="#101010", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.start_stroke(e, self.erase_mode))
        self.canvas.bind("<B1-Motion>", lambda e: self.continue_stroke(e, self.erase_mode))
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.start_stroke(e, True))
        self.canvas.bind("<B3-Motion>", lambda e: self.continue_stroke(e, True))
        self.canvas.bind("<ButtonRelease-3>", self.end_stroke)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.change_brush(2))
        self.canvas.bind("<Button-5>", lambda e: self.change_brush(-2))

        statusbar = ttk.Frame(self, padding=(8, 4))
        statusbar.grid(row=3, column=0, sticky="ew")
        statusbar.columnconfigure(1, weight=1)
        ttk.Label(statusbar, textvariable=self.position_var, width=14).grid(row=0, column=0, sticky="w")
        ttk.Label(statusbar, textvariable=self.file_var).grid(row=0, column=1, sticky="w")
        ttk.Label(statusbar, textvariable=self.mask_src_var).grid(row=0, column=2, sticky="e", padx=(0, 12))
        ttk.Label(statusbar, textvariable=self.state_var).grid(row=0, column=3, sticky="e")

        help_text = (
            "Left/Right: section    r/Rotate buttons: rotate 90    h: flip H    v: flip V    "
            "Left-drag: draw    Right-drag: erase    e: toggle draw/erase    c: clear    "
            "[ ]: brush size    s: save    Esc/q: quit"
        )
        ttk.Label(self, text=help_text, padding=(8, 2)).grid(row=4, column=0, sticky="w")
        self.update_brush_label()
        self.update_mode_label()

    # --- data loading ------------------------------------------------------- #
    def get_base_img(self, index: int) -> tuple[np.ndarray, tuple[int, int]]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        raw_value = self.records.at[index, "raw"]
        path = resolve_path(self.root_dir, raw_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing raw image: {path}")
        data, native_hw = load_gray_work(path)
        self.cache[index] = (data, native_hw)
        if len(self.cache) > MAX_CACHE_ITEMS:
            self.cache.popitem(last=False)
        return data, native_hw

    def get_mask_img(self, index: int, work_hw: tuple[int, int]) -> Image.Image:
        """Return the working mask image, preferring a previously edited one."""
        candidates: list[tuple[str, Path]] = []
        qc_value = str(self.records.at[index, "qc_mask"]).strip()
        if qc_value:
            candidates.append(("edited", resolve_path(self.root_dir, qc_value)))
        mask_value = str(self.records.at[index, "mask"]).strip()
        if mask_value and truthy(self.records.at[index, "has_mask"]):
            candidates.append(("original", resolve_path(self.root_dir, mask_value)))

        for source, path in candidates:
            try:
                if path.exists():
                    self.mask_source = source
                    return Image.fromarray(load_mask_work(path, work_hw), mode="L")
            except Exception:
                continue

        self.mask_source = "empty"
        return Image.new("L", (work_hw[1], work_hw[0]), 0)

    def load_current(self) -> None:
        try:
            base_img, native_hw = self.get_base_img(self.index)
        except Exception as exc:
            messagebox.showerror("QC GUI", str(exc))
            return
        self.base_img = base_img
        self.native_hw = native_hw
        height, width = base_img.shape
        self.base_index = np.arange(height * width, dtype=np.int32).reshape(height, width)
        self.base_mask_img = self.get_mask_img(self.index, (height, width))

        self.flip = normalize_flip(self.records.at[self.index, "flip"])
        self.rotation = normalize_rotation(self.records.at[self.index, "rotation"])
        self.records.at[self.index, "flip"] = self.flip
        self.records.at[self.index, "rotation"] = self.rotation

        self.last_pt = None
        self.mask_dirty = False
        self.disp_index = None
        self.records.at[self.index, "qc_reviewed"] = True  # displaying a section marks it reviewed
        self.render_all()
        self.schedule_csv_save()

    # --- rendering ---------------------------------------------------------- #
    def render_all(self) -> None:
        if self.base_img is None:
            return
        self.render_background()
        self.render_overlay()
        self.update_status()

    def render_background(self) -> None:
        if self.base_img is None or self.base_index is None:
            return
        disp_img = apply_transform(self.base_img, self.flip, self.rotation)
        self.disp_index = apply_transform(self.base_index, self.flip, self.rotation)
        disp_h, disp_w = disp_img.shape
        self.disp_shape = (disp_h, disp_w)

        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        scale = max(min(canvas_w / disp_w, canvas_h / disp_h), 1e-3)
        out_w = max(1, int(disp_w * scale))
        out_h = max(1, int(disp_h * scale))
        self.display_scale = scale
        offset_x = (canvas_w - out_w) // 2
        offset_y = (canvas_h - out_h) // 2
        self.display_offset = (offset_x, offset_y)

        disp_img = self.apply_contrast(disp_img)
        pil = Image.fromarray(disp_img, "L").convert("RGB").resize((out_w, out_h), LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(pil)
        if self.bg_id is None:
            self.bg_id = self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.bg_photo)
        else:
            self.canvas.itemconfigure(self.bg_id, image=self.bg_photo)
            self.canvas.coords(self.bg_id, offset_x, offset_y)

    def render_overlay(self) -> None:
        if self.base_mask_img is None:
            return
        mask_np = np.asarray(self.base_mask_img)
        disp_mask = apply_transform(mask_np, self.flip, self.rotation)
        out_w = max(1, int(self.disp_shape[1] * self.display_scale))
        out_h = max(1, int(self.disp_shape[0] * self.display_scale))
        stencil = Image.fromarray(disp_mask, "L").resize((out_w, out_h), NEAREST)
        rgba = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        rgba.paste(Image.new("RGBA", (out_w, out_h), (255, 40, 40, self.overlay_alpha)), (0, 0), stencil)
        self.ov_photo = ImageTk.PhotoImage(rgba)

        offset_x, offset_y = self.display_offset
        if self.ov_id is None:
            self.ov_id = self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.ov_photo)
        else:
            self.canvas.itemconfigure(self.ov_id, image=self.ov_photo)
            self.canvas.coords(self.ov_id, offset_x, offset_y)
        self.canvas.tag_raise(self.ov_id)
        if self.cursor_id is not None:
            self.canvas.tag_raise(self.cursor_id)

    # --- display controls --------------------------------------------------- #
    def apply_contrast(self, arr: np.ndarray) -> np.ndarray:
        """Window the 0-255 display values between contrast_min and contrast_max."""
        lo, hi = self.contrast_min, self.contrast_max
        if lo <= 0 and hi >= 255:
            return arr
        if hi <= lo:
            hi = lo + 1
        scaled = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def on_alpha_change(self, value: str) -> None:
        self.overlay_alpha = int(float(value))
        self.render_overlay()

    def on_contrast_change(self, _value: str | None = None) -> None:
        self.contrast_min = int(self.min_scale.get())
        self.contrast_max = int(self.max_scale.get())
        self.render_background()

    def auto_contrast(self) -> None:
        """Stretch the window to the current image's actual value range."""
        if self.base_img is None:
            return
        lo = int(self.base_img.min())
        hi = int(self.base_img.max())
        if hi <= lo:
            hi = min(255, lo + 1)
        self.min_scale.set(lo)
        self.max_scale.set(hi)

    def reset_contrast(self) -> None:
        self.min_scale.set(0)
        self.max_scale.set(255)

    # --- painting ----------------------------------------------------------- #
    def canvas_to_base(self, event_x: int, event_y: int) -> tuple[int, int] | None:
        if self.disp_index is None or self.base_img is None:
            return None
        offset_x, offset_y = self.display_offset
        scale = self.display_scale
        disp_x = (event_x - offset_x) / scale
        disp_y = (event_y - offset_y) / scale
        disp_h, disp_w = self.disp_shape
        if disp_x < 0 or disp_y < 0 or disp_x >= disp_w or disp_y >= disp_h:
            return None
        flat = int(self.disp_index[int(disp_y), int(disp_x)])
        base_w = self.base_img.shape[1]
        return flat // base_w, flat % base_w

    def stamp(self, r0: int, c0: int, r1: int, c1: int, value: int) -> None:
        if self.base_mask_img is None:
            return
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        draw = ImageDraw.Draw(self.base_mask_img)
        draw.line([(c0, r0), (c1, r1)], fill=value, width=2 * radius)
        for row, col in ((r0, c0), (r1, c1)):
            draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=value)

    def start_stroke(self, event: tk.Event, erase: bool) -> None:
        self.canvas.focus_set()
        self.last_pt = None
        self.continue_stroke(event, erase)

    def continue_stroke(self, event: tk.Event, erase: bool) -> None:
        point = self.canvas_to_base(event.x, event.y)
        self.update_cursor(event.x, event.y, erase)
        if point is None:
            self.last_pt = None
            return
        value = 0 if erase else 255
        if self.last_pt is None:
            self.stamp(point[0], point[1], point[0], point[1], value)
        else:
            self.stamp(self.last_pt[0], self.last_pt[1], point[0], point[1], value)
        self.last_pt = point
        self.mask_dirty = True
        self.render_overlay()

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_pt = None
        if self.mask_dirty:
            self.records.at[self.index, "qc_reviewed"] = True
            self.schedule_mask_save()
            self.schedule_csv_save()

    def clear_mask(self) -> None:
        if self.base_mask_img is None:
            return
        self.base_mask_img.paste(0, (0, 0, self.base_mask_img.width, self.base_mask_img.height))
        self.mask_dirty = True
        self.mask_source = "empty"
        self.records.at[self.index, "qc_reviewed"] = True
        self.render_overlay()
        self.update_status()
        self.schedule_mask_save()
        self.schedule_csv_save()

    # --- transforms --------------------------------------------------------- #
    def rotate(self, delta: int) -> None:
        self.rotation = normalize_rotation(self.rotation + delta)
        self.records.at[self.index, "rotation"] = self.rotation
        self.render_all()
        self.schedule_csv_save()

    def flip_h(self) -> None:
        lr_flip, up_flip = decode_flip(self.flip)
        self.flip = encode_flip(not lr_flip, up_flip)
        self.records.at[self.index, "flip"] = self.flip
        self.render_all()
        self.schedule_csv_save()

    def flip_v(self) -> None:
        lr_flip, up_flip = decode_flip(self.flip)
        self.flip = encode_flip(lr_flip, not up_flip)
        self.records.at[self.index, "flip"] = self.flip
        self.render_all()
        self.schedule_csv_save()

    # --- brush / mode ------------------------------------------------------- #
    def change_brush(self, delta: int) -> None:
        self.brush = int(min(MAX_BRUSH, max(MIN_BRUSH, self.brush + delta)))
        self.update_brush_label()

    def toggle_erase(self) -> None:
        self.erase_mode = not self.erase_mode
        self.update_mode_label()

    def update_brush_label(self) -> None:
        self.brush_var.set(f"brush: {self.brush}px")

    def update_mode_label(self) -> None:
        mode = "Erase" if self.erase_mode else "Draw"
        self.mode_var.set(mode)
        self.mode_button.configure(text=f"Mode: {mode}")

    def update_cursor(self, x: int, y: int, erase: bool) -> None:
        radius = self.brush
        colour = "#ff4040" if erase else "#ffff40"
        if self.cursor_id is None:
            self.cursor_id = self.canvas.create_oval(0, 0, 0, 0, outline=colour, width=2)
        self.canvas.coords(self.cursor_id, x - radius, y - radius, x + radius, y + radius)
        self.canvas.itemconfigure(self.cursor_id, outline=colour)
        self.canvas.tag_raise(self.cursor_id)

    def on_motion(self, event: tk.Event) -> None:
        self.update_cursor(event.x, event.y, self.erase_mode)

    def on_leave(self, _event: tk.Event) -> None:
        if self.cursor_id is not None:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None

    def on_wheel(self, event: tk.Event) -> None:
        self.change_brush(2 if event.delta > 0 else -2)

    # --- navigation / keys -------------------------------------------------- #
    def move(self, delta: int) -> None:
        self.flush_mask()
        self.index = (self.index + delta) % len(self.records)
        self.load_current()

    def flush_mask(self) -> None:
        if self.mask_save_after_id is not None:
            self.after_cancel(self.mask_save_after_id)
            self.mask_save_after_id = None
        self.save_mask()

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym
        char = event.char.lower() if event.char else ""
        if key in ("Left", "Up"):
            self.move(-1)
        elif key in ("Right", "Down"):
            self.move(1)
        elif char == "r":
            self.rotate(1)
        elif char == "h":
            self.flip_h()
        elif char == "v":
            self.flip_v()
        elif char == "c":
            self.clear_mask()
        elif char == "e":
            self.toggle_erase()
        elif char == "s":
            self.save_now()
        elif char == "[":
            self.change_brush(-2)
        elif char == "]":
            self.change_brush(2)
        elif key == "Escape" or char == "q":
            self.on_close()

    def on_canvas_configure(self, _event: tk.Event) -> None:
        if self.base_img is None:
            return
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        self.resize_after_id = self.after(120, self.render_all)

    # --- saving ------------------------------------------------------------- #
    def mask_out_path(self, index: int) -> Path:
        return self.mask_out_dir / f"{self.records.at[index, 'base']}.png"

    def save_mask(self) -> None:
        if self.base_mask_img is None or not self.mask_dirty:
            return
        out_path = self.mask_out_path(self.index)
        native_h, native_w = self.native_hw
        full = self.base_mask_img.resize((native_w, native_h), NEAREST)
        full.save(out_path)
        self.records.at[self.index, "qc_mask"] = rel_path(self.root_dir, out_path)
        self.records.at[self.index, "qc_reviewed"] = True
        self.mask_dirty = False

    def schedule_mask_save(self) -> None:
        if self.mask_save_after_id is not None:
            self.after_cancel(self.mask_save_after_id)
        self.mask_save_after_id = self.after(SAVE_DEBOUNCE_MS, self.run_mask_save)
        self.status_var.set("Saving...")

    def run_mask_save(self) -> None:
        self.mask_save_after_id = None
        self.save_mask()
        self.status_var.set("Saved")

    def schedule_csv_save(self) -> None:
        if self.pending_save_id is not None:
            self.after_cancel(self.pending_save_id)
        self.pending_save_id = self.after(SAVE_DEBOUNCE_MS, self.run_csv_save)
        self.status_var.set("Saving...")

    def run_csv_save(self) -> None:
        self.pending_save_id = None
        self.write_csv()
        self.status_var.set("Saved")

    def write_csv(self) -> None:
        output = self.records.copy()
        output["flip"] = output["flip"].fillna("").astype(str)
        output["rotation"] = output["rotation"].fillna(0).astype(int)
        output["qc_mask"] = output["qc_mask"].fillna("").astype(str)
        output["qc_reviewed"] = output["qc_reviewed"].astype(bool)
        output.to_csv(self.out_csv, index=False)

    def commit_current(self) -> None:
        self.flush_mask()
        if self.pending_save_id is not None:
            self.after_cancel(self.pending_save_id)
            self.pending_save_id = None
        self.write_csv()

    def save_now(self) -> None:
        self.commit_current()
        self.status_var.set("Saved")

    def on_close(self) -> None:
        self.commit_current()
        self.destroy()

    # --- status ------------------------------------------------------------- #
    def update_status(self) -> None:
        row = self.records.iloc[self.index]
        flip_display = self.flip if self.flip else "none"
        reviewed = "reviewed" if truthy(row["qc_reviewed"]) else "not reviewed"
        self.position_var.set(f"{self.index + 1} / {len(self.records)}")
        self.file_var.set(str(row["base"]))
        self.mask_src_var.set(f"mask: {self.mask_source}  ({reviewed})")
        self.state_var.set(
            f"flip: {flip_display}    rotation: {self.rotation} x 90 ({self.rotation * 90:+d} deg)"
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC raw autoradiograph images and their crop-mask overlays.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="split CSV to review, e.g. csv/splits/MR1_L_afdx.csv (if omitted, a file picker opens)",
    )
    parser.add_argument("--root", default=None, help="directory holding raw/ and mask/ (auto-detected by default)")
    parser.add_argument("--out-csv", default=None, help="output CSV for recorded flips/rotations (default <csv>.qc.csv)")
    parser.add_argument("--mask-out-dir", default=None, help="directory for edited masks (default <root>/mask_qc)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    app = QcGui(csv_path, args.root, args.out_csv, args.mask_out_dir)
    if app.aborted:
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
