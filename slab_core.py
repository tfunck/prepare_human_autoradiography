"""Qt-free core of the slab landmark tool (human autoradiography slab <-> donor MRI).

Geometry of a slab
------------------
Sections are 2-D NIfTI images (BrainBuilder style: first axis x, second axis z)
or PNG/TIFF images, stacked along y. In-plane pixel sizes come from each
section's NIfTI header (``pixel_size`` for PNG/TIFF).

Order along y (``order``):
  section  y = section number x slice thickness (default 0.02 mm)
  stack    as in stack_view.py: sorted by (repeat, ligand, name), where the
           repeat is the 7th '#' field of the name; consecutive sections are one
           slice thickness apart; names without a repeat go last
  auto     section if every section has its own number, else stack (e.g. when
           the section column holds the repeat number, shared by ~20 ligands)

In-plane placement (``SlabStack.centres``):
  mass     as in stack_view.py: each section is shifted so the centre of mass of
           its tissue mask (CSV mask column; intensity-weighted centroid when a
           section has no mask) is at the same point for all sections
  extent   as in BrainBuilder: sections of different sizes are centred in a
           common extent

The *stack frame* is that grid in millimetres:

    x = offset_x + (i + 0.5) * pixel_x      (i: first section axis)
    y = position * thickness                (position: section number or rank)
    z = offset_z + (j + 0.5) * pixel_z      (j: second section axis)

ANTs works in LPS physical space. The stack frame is used as the slab's
physical space; it is mirrored in x when the landmarks show that the slab is
mirror-imaged relative to the MRI (antspyx rigid *and* affine landmark fits
only produce proper rotations, so handedness must match before fitting).

Transforms
----------
``fit_transform`` maps slab (fixed) physical points to MRI (moving) points, so
``antsApplyTransforms`` with the slab as reference resamples the MRI into slab
space. By the number of landmark pairs: none (< rigid_min), rigid, affine, or
nonlinear = affine + B-spline displacement field. As in BrainBuilder, the
nonlinear part is fitted after the affine: the MRI landmarks are first moved
into slab space with the inverse affine (BrainBuilder does this by resampling
the label volume; with centroids it is the same operation without resampling
error), then the B-spline field is fitted between those and the slab
landmarks. The transform list for antsApplyTransforms is [nonlinear, affine].
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

import landmark_core as core

log = logging.getLogger("landmarks")

DEFAULT_THICKNESS = 0.02          # mm between consecutive section numbers
DEFAULT_PIXEL_SIZE = 0.2          # mm, PNG/TIFF sections (no header): 20 um originals downsampled 10x
MAX_ROWS = 800                    # rows (y samples) in the sagittal/axial slab views
THUMB_BUDGET = 250e6              # bytes for the downsampled slab volume
CORONAL_RES = 768                 # longest side of the resampled coronal plane (floating window)
MRI_KEY = "__mri__"

try:
    BOX = Image.Resampling.BOX
except AttributeError:            # Pillow < 9.1
    BOX = Image.BOX


def _ants():
    import ants                   # imported lazily: slow import, only needed for transforms
    return ants


# --------------------------------------------------------------------------- #
# Slab selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlabKey:
    sub: str
    hemisphere: str
    chunk: str

    @property
    def id(self) -> str:
        return f"{self.sub}-{self.hemisphere}-slab{self.chunk}"

    def matches(self, sub, hemisphere, chunk) -> bool:
        return (core.clean_sub(sub) == self.sub and str(hemisphere).strip() == self.hemisphere
                and str(chunk).strip() == self.chunk)


SLAB_REQUIRED = ("sub", "hemisphere", "raw")      # section numbers are optional (see ``order``)


def list_slabs(csv_path: str | Path, overrides: dict | None = None) -> list[SlabKey]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    cols = core.resolve_columns(df.columns, overrides, SLAB_REQUIRED)
    if "chunk" not in cols:
        raise ValueError(f"The CSV needs a slab column (chunk/slab). Columns: {list(df.columns)}")
    keys = []
    for _, r in df.iterrows():
        k = SlabKey(core.clean_sub(r[cols["sub"]]), str(r[cols["hemisphere"]]).strip(), str(r[cols["chunk"]]).strip())
        if k not in keys:
            keys.append(k)
    return keys


def other_slab_labels(df: pd.DataFrame, cols: dict, csv_dir: Path, key: SlabKey) -> dict[int, list[str]]:
    """Labels used by the *other* slabs of the same subject/hemisphere
    (read from the label files listed in the CSV): label -> [slab ids]."""
    out: dict[int, list[str]] = {}
    if "labels" not in cols or "chunk" not in cols:
        return out
    for _, r in df.iterrows():
        sub, hemi, chunk = core.clean_sub(r[cols["sub"]]), str(r[cols["hemisphere"]]).strip(), str(r[cols["chunk"]]).strip()
        if sub != key.sub or hemi != key.hemisphere or chunk == key.chunk:
            continue
        p = core.resolve_path(r[cols["labels"]], csv_dir)
        if p is None or not p.exists():
            continue
        try:
            arr = np.asanyarray(core.nib.load(str(p)).dataobj) if core.is_nifti(p) else np.asarray(Image.open(p))
        except Exception as exc:          # unreadable file: report, don't block
            log.warning("Cannot read %s: %s", p, exc)
            continue
        sid = SlabKey(sub, hemi, chunk).id
        for v in np.unique(np.rint(arr)).astype(np.int64):
            if v > 0 and sid not in out.setdefault(int(v), []):
                out[int(v)].append(sid)
    return out


# --------------------------------------------------------------------------- #
# Section order and centring (as in stack_view.py)
# --------------------------------------------------------------------------- #
ORDER_MODES = ("auto", "section", "stack")
CENTER_MODES = ("auto", "mask", "extent")
CENTRE_CACHE_VERSION = 1


def parse_repeat(base: object) -> int:
    """Repeat number (``#``-field 6) of a section name; unparseable -> 10**6 (sorted last)."""
    parts = str(base).split("#")
    if len(parts) > 6:
        m = re.search(r"\d+", parts[6])
        if m:
            return int(m.group())
    return 10 ** 6


def stack_key(s: core.Section) -> tuple:
    """stack_view.py order: (repeat, ligand, name)."""
    return parse_repeat(s.name), s.acquisition or "", s.name


def resolve_order(sections: Sequence[core.Section], order: str = "auto") -> tuple[str, str]:
    """('section' | 'stack', explanation)."""
    if order not in ORDER_MODES:
        raise ValueError(f"order must be one of {ORDER_MODES}, not {order!r}")
    nums = [s.number for s in sections]
    missing = sum(n is None for n in nums)
    present = [n for n in nums if n is not None]
    shared = len(present) - len(set(present))
    if order == "section":
        if missing:
            raise ValueError(f"{missing} section(s) have no section number; use --order stack to stack them "
                             f"by name like stack_view.py")
        return "section", "y = section number x slice thickness"
    why = []
    if shared:
        why.append(f"{len(present)} sections have only {len(set(present))} distinct section numbers")
    if missing:
        why.append(f"{missing} have none")
    if order == "auto" and not why:
        return "section", "every section has its own number: y = section number x slice thickness"
    unparsed = sum(parse_repeat(s.name) == 10 ** 6 for s in sections)
    note = ("stacked like stack_view.py: ordered by (repeat, ligand, name), one slice thickness apart"
            + (f" ({'; '.join(why)})" if why else ""))
    if unparsed:
        note += f"; {unparsed} section(s) without a repeat number in their name are at the end"
    return "stack", note


class SlabStack(core.SpecimenStack):
    """A slab's sections in stacking order, with optional centres of mass.

    ``positions[i]`` is the y index of section i (y = position x thickness):
    the section number, or the rank in stack_view order. ``centres`` is None
    (centre by extent) or an (n, 2) array of each section's centre of mass in
    its file layout (NIfTI (i, j); PNG/TIFF (row, column))."""

    def __init__(self, sections: Sequence[core.Section], order: str = "auto", **kw):
        mode, note = resolve_order(sections, order)
        super().__init__(sections, **kw)
        self.order, self.order_note = mode, note
        if mode == "stack":
            self.sections = sorted(self.sections, key=stack_key)
            self.positions = np.arange(len(self.sections), dtype=np.int64)
        else:
            self.positions = np.array([s.number for s in self.sections], dtype=np.int64)
        self.centres: np.ndarray | None = None
        self.centre_source: list[str] | None = None

    def describe(self, idx: int, short: bool = False) -> str:
        if self.order != "stack":
            return super().describe(idx, short)
        return f"#{idx + 1}" if short else self.sections[idx].name

    def find(self, text: str) -> tuple[int | None, str]:
        """Section by stack position (as shown, 1..n) or by part of its name."""
        if self.order != "stack":
            return super().find(text)
        t = text.strip().lstrip("#")
        if not t:
            return None, ""
        if t.isdigit():
            k = int(t)
            if 1 <= k <= len(self):
                return k - 1, ""
            k = int(np.clip(k, 1, len(self)))
            return k - 1, f"position {t} not in the stack (1-{len(self)}); showing {k}"
        hits = [i for i, s in enumerate(self.sections) if t.lower() in s.name.lower()]
        if not hits:
            return None, f"no section name contains '{t}'"
        return hits[0], (f"{len(hits)} sections match '{t}'; showing the first" if len(hits) > 1 else "")


def _read_mask(path: str) -> np.ndarray:
    arr, _meta = core.read_image(path)
    return arr > 0


def section_centre(raw: str, mask: str | None) -> tuple[float, float, str, str]:
    """Centre of mass of one section in its file layout, as in stack_view.py:
    the mask's centre of mass (mask > 0) if there is a non-empty mask, else the
    intensity-weighted centroid. Returns (c0, c1, source, note) with source
    'mask', 'intensity' or 'centre' (blank image)."""
    shape = core.read_meta(raw).shape
    note = ""
    if mask:
        try:
            m = _read_mask(mask)
            if m.shape != shape:
                f0, f1 = shape[0] / m.shape[0], shape[1] / m.shape[1]
                if abs(f0 / f1 - 1) > 0.05:
                    raise ValueError(f"mask shape {m.shape} doesn't match the section's {shape}")
            if m.any():
                r, c = np.nonzero(m)
                c0, c1 = r.mean(), c.mean()
                if m.shape != shape:     # mask stored at another resolution: scale its centre
                    c0, c1 = (c0 + 0.5) * f0 - 0.5, (c1 + 0.5) * f1 - 0.5
                return float(c0), float(c1), "mask", ""
            note = "empty mask"
        except Exception as exc:
            note = f"mask not used: {exc}"
    arr, _meta = core.read_image(raw)
    w = np.clip(np.nan_to_num(arr.astype(np.float64)), 0, None)
    total = float(w.sum())
    if total > 0:
        c0 = float((w.sum(axis=1) * np.arange(shape[0])).sum() / total)
        c1 = float((w.sum(axis=0) * np.arange(shape[1])).sum() / total)
        return c0, c1, "intensity", note
    return (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, "centre", note


def _file_sig(path: str | None) -> str:
    if not path:
        return "-"
    try:
        st = os.stat(path)
        return f"{Path(path).resolve()}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{path}:missing"


def compute_centres(stack: SlabStack, cache_dir: Path | None = None, workers: int = 4,
                    progress=None) -> list[str]:
    """Fill ``stack.centres`` / ``stack.centre_source``; returns notes for the
    sections whose mask could not be used. Results are cached per file."""
    secs = stack.sections
    n = len(secs)
    keys = [f"{_file_sig(s.raw)}|{_file_sig(s.mask)}|v{CENTRE_CACHE_VERSION}" for s in secs]
    cache_file = None
    cached: dict = {}
    if cache_dir is not None:
        tag = hashlib.sha1("|".join(sorted(s.raw for s in secs)).encode()).hexdigest()[:12]
        cache_file = Path(cache_dir) / f"centres_{tag}.json"
        try:
            cached = json.loads(cache_file.read_text())
        except Exception:
            cached = {}
    out: list = [None] * n
    todo = []
    for i, k in enumerate(keys):
        if k in cached:
            out[i] = tuple(cached[k])
        else:
            todo.append(i)
    done = n - len(todo)
    if progress:
        progress("Centring sections on their masks", done, n)
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="centre") as pool:
            futs = {i: pool.submit(section_centre, secs[i].raw, secs[i].mask) for i in todo}
            for i in todo:
                try:
                    out[i] = futs[i].result()
                except Exception as exc:           # unreadable image: leave it at the canvas centre
                    meta = stack.meta(i)
                    out[i] = ((meta.shape[0] - 1) / 2.0, (meta.shape[1] - 1) / 2.0, "centre", f"error: {exc}")
                done += 1
                if progress and (done % 16 == 0 or done == n):
                    progress("Centring sections on their masks", done, n)
        if cache_file is not None:
            try:
                cached.update({keys[i]: list(out[i]) for i in todo})
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(cached))
                os.replace(tmp, cache_file)
            except OSError as exc:
                log.debug("Cannot write %s: %s", cache_file, exc)
    stack.centres = np.array([(o[0], o[1]) for o in out], dtype=np.float64)
    stack.centre_source = [o[2] for o in out]
    return [f"{secs[i].name}: {o[3]}" for i, o in enumerate(out) if o[3]]


def centring_summary(stack: SlabStack) -> str:
    if stack.centres is None:
        return "sections centred by image extent (no mask column)"
    src = stack.centre_source or []
    n_mask, n_int = src.count("mask"), src.count("intensity") + src.count("centre")
    text = f"sections centred on their mask's centre of mass ({n_mask} of {len(src)})"
    if n_int:
        text += f"; {n_int} without a usable mask centred on their intensity-weighted centroid"
    return text


# --------------------------------------------------------------------------- #
# Slab geometry
# --------------------------------------------------------------------------- #
class SlabGeometry:
    """Sections in the slab's (u, v) convention: u to the right, v up.

    NIfTI sections: u, v are the file's two axes (x, z), pixel size from the
    header. PNG/TIFF sections: u = column, v = rows counted from the bottom (the
    image is shown as stored, row 0 at the top); pixel size = ``pixel_size``.
    Label arrays always keep the file's own layout; use ``to_uv`` /
    ``native_to_uv`` / ``uv_to_native_flat`` to convert.

    y positions and in-plane placement come from a :class:`SlabStack`
    (``positions``, ``centres``); a plain SpecimenStack uses the section numbers
    and centres the sections by extent."""

    def __init__(self, stack: core.SpecimenStack, thickness: float = DEFAULT_THICKNESS,
                 pixel_size: float = DEFAULT_PIXEL_SIZE):
        n = len(stack)
        metas = [stack.meta(i) for i in range(n)]
        self.thickness = float(thickness)
        self.pixel_size = float(pixel_size)
        self.raster = np.array([m.kind != "nifti" for m in metas], dtype=bool)
        self.shapes = np.array([m.shape if m.kind == "nifti" else (m.shape[1], m.shape[0]) for m in metas],
                               dtype=np.int64)                                    # (n, 2): (nu, nv)
        self.spacing = np.array([m.spacing if m.kind == "nifti" else (pixel_size, pixel_size) for m in metas],
                                dtype=np.float64)                                 # (n, 2) mm
        ext = self.shapes * self.spacing
        centres = getattr(stack, "centres", None)
        if centres is None:
            self.centring = "extent"
            self.extent = ext.max(axis=0)                                         # (Ex, Ez) mm
            self.offset = (self.extent - ext) / 2.0                               # centred, like BrainBuilder
            self.centre = self.extent / 2.0
        else:                       # like stack_view.py: all centres of mass at one point
            self.centring = "mass"
            uv = np.array([self.native_to_uv(i, *centres[i]) for i in range(n)], dtype=np.float64)
            start = -(uv + 0.5) * self.spacing         # section corner when its centre of mass is at 0
            lo, hi = start.min(axis=0), (start + ext).max(axis=0)
            self.offset = start - lo
            self.extent = hi - lo
            self.centre = -lo                          # frame (x, z) of the common centre of mass
        pos = getattr(stack, "positions", None)
        if pos is None:
            pos = [s.number for s in stack.sections]
        self.numbers = np.asarray(pos, dtype=np.int64)
        self.y = self.numbers * self.thickness
        order = np.argsort(self.y, kind="stable")
        self._y_sorted, self._idx_sorted = self.y[order], order

        # Downsampled grid for the sagittal/axial views and the 3-D resampling.
        side = int(np.clip(math.sqrt(THUMB_BUDGET / max(n, 1)), 128, 512))
        self.ts = float(self.extent.max() / side)
        self.ts = max(self.ts, float(self.spacing.min()))
        self.tshape = (max(1, int(math.ceil(self.extent[0] / self.ts))),
                       max(1, int(math.ceil(self.extent[1] / self.ts))))
        span = float(self.y.max() - self.y.min())
        k = max(1, int(math.ceil(round(span / self.thickness) / MAX_ROWS)))
        self.ty = self.thickness * k
        self.nrows = int(round(span / self.ty)) + 1
        self.y0 = float(self.y.min())
        self.row_y = self.y0 + np.arange(self.nrows) * self.ty
        self.row_sec = np.array([self.nearest_section(v) for v in self.row_y], dtype=np.int64)

        # Coronal plane grid used for resampling the MRI in the floating window.
        self.cs = max(float(self.extent.max() / CORONAL_RES), float(self.spacing.min()))
        self.cshape = (max(1, int(math.ceil(self.extent[0] / self.cs))),
                       max(1, int(math.ceil(self.extent[1] / self.cs))))

    # -- coordinates ------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.y)

    def nearest_section(self, y_mm: float) -> int:
        pos = int(np.searchsorted(self._y_sorted, y_mm))
        cands = [p for p in (pos - 1, pos) if 0 <= p < len(self._y_sorted)]
        best = min(cands, key=lambda p: abs(self._y_sorted[p] - y_mm))
        return int(self._idx_sorted[best])

    # -- file layout <-> (u, v) ------------------------------------------------ #
    def to_uv(self, idx: int, arr: np.ndarray | None):
        """View of a section-shaped array (file layout) indexed [u, v]."""
        if arr is None or not self.raster[idx]:
            return arr
        return arr.T[:, ::-1]

    def native_to_uv(self, idx: int, i: float, j: float) -> tuple[float, float]:
        if not self.raster[idx]:
            return i, j
        return j, self.shapes[idx][1] - 1 - i

    def uv_to_native_flat(self, idx: int, u, v) -> np.ndarray:
        u, v = np.asarray(u, np.int64), np.asarray(v, np.int64)
        nu, nv = (int(x) for x in self.shapes[idx])
        if not self.raster[idx]:
            return u * nv + v
        return (nv - 1 - v) * nu + u

    def com_to_frame(self, idx: int, com) -> np.ndarray:
        """Frame position of a label centre of mass given in file layout."""
        return self.section_to_frame(idx, *self.native_to_uv(idx, com[0], com[1]))

    def section_to_frame(self, idx: int, i: float, j: float) -> np.ndarray:
        sx, sz = self.spacing[idx]
        ox, oz = self.offset[idx]
        return np.array([ox + (i + 0.5) * sx, self.y[idx], oz + (j + 0.5) * sz])

    def frame_to_section(self, idx: int, x: float, z: float) -> tuple[float, float]:
        sx, sz = self.spacing[idx]
        ox, oz = self.offset[idx]
        return (x - ox) / sx - 0.5, (z - oz) / sz - 0.5

    def y_to_row(self, y_mm: float) -> float:
        return (y_mm - self.y0) / self.ty

    def row_to_y(self, row: float) -> float:
        return self.y0 + row * self.ty

    @staticmethod
    def to_physical(frame_pts: np.ndarray, mirror: bool) -> np.ndarray:
        p = np.array(frame_pts, dtype=np.float64, copy=True)
        if mirror:
            p[..., 0] *= -1.0
        return p

    to_frame = to_physical    # mirroring x is its own inverse

    # -- ANTs reference grids ------------------------------------------------- #
    def _ref(self, shape, origin_frame, spacing, mirror):
        ants = _ants()
        d = np.eye(3)
        if mirror:
            d[0, 0] = -1.0
        origin = self.to_physical(np.asarray(origin_frame, float), mirror)
        return ants.from_numpy(np.zeros(shape, np.float32), origin=tuple(float(v) for v in origin),
                               spacing=tuple(float(v) for v in spacing), direction=d)

    def coronal_ref(self, y_mm: float, mirror: bool):
        return self._ref((self.cshape[0], 1, self.cshape[1]), (0.5 * self.cs, y_mm, 0.5 * self.cs),
                         (self.cs, self.ty, self.cs), mirror)

    def sagittal_ref(self, x_mm: float, mirror: bool):
        return self._ref((1, self.nrows, self.tshape[1]), (x_mm, self.y0, 0.5 * self.ts),
                         (self.ts, self.ty, self.ts), mirror)

    def axial_ref(self, z_mm: float, mirror: bool):
        return self._ref((self.tshape[0], self.nrows, 1), (0.5 * self.ts, self.y0, z_mm),
                         (self.ts, self.ty, self.ts), mirror)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Frame bounding box (min, max) of the slab."""
        return (np.array([0.0, self.y.min() - self.thickness / 2, 0.0]),
                np.array([self.extent[0], self.y.max() + self.thickness / 2, self.extent[1]]))


# --------------------------------------------------------------------------- #
# Downsampled slab (for the sagittal/axial views)
# --------------------------------------------------------------------------- #
def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "slab_landmark_gui"


class ThumbStack:
    """Each section, contrast-normalised (0.5-99.5th percentile) and downsampled
    onto the common grid, built in a background thread. Cached on disk (keyed by
    file path, size and modification time) so re-opening a slab is fast."""

    def __init__(self, stack: core.SpecimenStack, geom: SlabGeometry, cache_dir: Path | None = None,
                 workers: int = 3):
        self.stack, self.geom = stack, geom
        self.cache_dir = Path(cache_dir) if cache_dir else None
        tw, th = geom.tshape
        self.thumbs = np.zeros((geom.n, tw, th), np.uint8)
        self.done = np.zeros(geom.n, bool)
        self.failed: dict[int, str] = {}
        self._stop = False
        self._priority: list[int] = []
        self._lock = threading.Lock()
        self._busy: set[int] = set()
        self._threads = [threading.Thread(target=self._run, daemon=True, name=f"thumbs{k}")
                         for k in range(max(1, workers))]

    def start(self, first: int = 0):
        self._priority = [first]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop = True

    def prioritise(self, idx: int):
        with self._lock:
            self._priority.append(idx)

    @property
    def progress(self) -> tuple[int, int]:
        return int(self.done.sum()) + len(self.failed), self.geom.n

    def _key(self, idx: int) -> str:
        p = Path(self.stack.sections[idx].raw)
        st = p.stat()
        h = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{self.geom.tshape}|{self.geom.ts:.6f}|"
                         f"{self.geom.offset[idx].tolist()}".encode()).hexdigest()
        return h

    def _compute(self, idx: int) -> np.ndarray:
        arr, _meta = core.read_image(self.stack.sections[idx].raw)
        arr = self.geom.to_uv(idx, arr)
        lo, hi = core.levels_from_sample(core.percentile_sample(arr), 0.5, 99.5)
        img8 = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        g = self.geom
        ex, ez = g.shapes[idx] * g.spacing[idx]
        tx, tz = max(1, int(round(ex / g.ts))), max(1, int(round(ez / g.ts)))
        small = np.asarray(Image.fromarray(img8).resize((tz, tx), BOX))       # rows=x, cols=z
        out = np.zeros(g.tshape, np.uint8)
        a0 = int(round(g.offset[idx][0] / g.ts))
        b0 = int(round(g.offset[idx][1] / g.ts))
        a1, b1 = min(g.tshape[0], a0 + tx), min(g.tshape[1], b0 + tz)
        out[a0:a1, b0:b1] = small[: a1 - a0, : b1 - b0]
        return out

    def _load(self, idx: int) -> np.ndarray:
        path = None
        if self.cache_dir is not None:
            try:
                path = self.cache_dir / f"{self._key(idx)}.npy"
                if path.exists():
                    t = np.load(path)
                    if t.shape == self.geom.tshape:
                        return t
            except Exception:                       # unreadable cache entry: recompute
                path = None
        t = self._compute(idx)
        if path is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
                np.save(tmp, t)
                os.replace(tmp, path)
            except OSError:
                pass
        return t

    def _next(self):
        with self._lock:
            while self._priority:
                i = self._priority.pop()
                if 0 <= i < self.geom.n and not self.done[i] and i not in self.failed and i not in self._busy:
                    self._busy.add(i)
                    return i
            for i in np.flatnonzero(~self.done):
                i = int(i)
                if i not in self.failed and i not in self._busy:
                    self._busy.add(i)
                    return i
        return None

    def _run(self):
        while not self._stop:
            i = self._next()
            if i is None:
                return
            try:
                self.thumbs[i] = self._load(i)
                self.done[i] = True
            except Exception as exc:
                log.warning("Cannot build view for %s: %s", self.stack.sections[i].raw, exc)
                self.failed[i] = str(exc)
            finally:
                with self._lock:
                    self._busy.discard(i)

    # -- planes (arrays indexed [u, v]: u horizontal, v vertical/up) ----------- #
    def sagittal(self, x_mm: float) -> np.ndarray:
        a = int(np.clip(np.floor(x_mm / self.geom.ts), 0, self.geom.tshape[0] - 1))
        return self.thumbs[self.geom.row_sec, a, :]                              # (rows, TH): u=y, v=z

    def axial(self, z_mm: float) -> np.ndarray:
        b = int(np.clip(np.floor(z_mm / self.geom.ts), 0, self.geom.tshape[1] - 1))
        return self.thumbs[self.geom.row_sec, :, b].T                            # (TW, rows): u=x, v=y


# --------------------------------------------------------------------------- #
# Donor MRI
# --------------------------------------------------------------------------- #
class DonorMRI(core.Template):
    """Template + the three anatomical axes and LPS conversion."""

    def __init__(self, image_path, label_path=None):
        super().__init__(image_path, label_path, "auto")
        codes = self.axcodes
        def axis_of(letters, default):
            hits = [i for i, c in enumerate(codes) if c in letters]
            return hits[0] if hits else default
        self.ax_cor = axis_of("AP", 1)
        self.ax_sag = axis_of("LR", 0)
        self.ax_axi = axis_of("SI", 2)
        if len({self.ax_cor, self.ax_sag, self.ax_axi}) != 3:
            self.ax_sag, self.ax_cor, self.ax_axi = 0, 1, 2
        self._ants_img = None
        self._ants_lab = None
        self._lab_version = -1
        self.version = 0                   # bumped by the GUI when labels change

    # plane definitions: (fixed axis, u axis, v axis, (flip_u, flip_v))
    def plane_axes(self, plane: str):
        c = self.axcodes
        if plane == "coronal":
            u, v = self.ax_sag, self.ax_axi
            return self.ax_cor, u, v, (c[u] == "L", c[v] == "I")
        if plane == "sagittal":
            u, v = self.ax_cor, self.ax_axi          # anterior shown on the left
            return self.ax_sag, u, v, (c[u] == "A", c[v] == "I")
        u, v = self.ax_sag, self.ax_cor              # axial: anterior at the top
        return self.ax_axi, u, v, (c[u] == "L", c[v] == "P")

    def plane(self, vol: np.ndarray, plane: str, k: int) -> np.ndarray:
        a, u, v, _ = self.plane_axes(plane)
        return np.transpose(vol, (a, u, v))[int(k)]

    def plane_flat_to_volume(self, plane: str, k: int, flat: np.ndarray) -> np.ndarray:
        a, u, v, _ = self.plane_axes(plane)
        uu, vv = np.unravel_index(np.asarray(flat), (self.shape[u], self.shape[v]))
        idx: list = [None, None, None]
        idx[a], idx[u], idx[v] = np.full(uu.shape, int(k)), uu, vv
        return np.ravel_multi_index(idx, self.shape)

    def voxel_to_lps(self, ijk) -> np.ndarray:
        ijk = np.atleast_2d(np.asarray(ijk, float))
        ras = (self.affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
        return ras * np.array([-1.0, -1.0, 1.0])

    def lps_to_voxel(self, lps) -> np.ndarray:
        lps = np.atleast_2d(np.asarray(lps, float))
        ras = lps * np.array([-1.0, -1.0, 1.0])
        return (np.linalg.inv(self.affine) @ np.c_[ras, np.ones(len(ras))].T).T[:, :3]

    def ants_geometry(self):
        """origin/spacing/direction of the MRI grid in ANTs (LPS) terms."""
        flip = np.diag([-1.0, -1.0, 1.0])
        m = flip @ self.affine[:3, :3]
        spacing = np.linalg.norm(m, axis=0)
        direction = m / spacing
        origin = flip @ self.affine[:3, 3]
        return tuple(origin), tuple(spacing), direction

    def ants_image(self):
        if self._ants_img is None:
            o, s, d = self.ants_geometry()
            self._ants_img = _ants().from_numpy(self.image, origin=o, spacing=s, direction=d)
        return self._ants_img

    def ants_labels(self):
        if self._ants_lab is None or self._lab_version != self.version:
            o, s, d = self.ants_geometry()
            self._ants_lab = _ants().from_numpy(self.labels.astype(np.float32), origin=o, spacing=s, direction=d)
            self._lab_version = self.version
        return self._ants_lab


# --------------------------------------------------------------------------- #
# Landmark pairs and transform fitting
# --------------------------------------------------------------------------- #
@dataclass
class TransformSettings:
    rigid_min: int = 3
    affine_min: int = 12
    nonlinear_min: int = 20
    cp_spacing: float = 10.0          # final B-spline control-point spacing (mm)
    levels: int = 4                   # B-spline fitting levels (mesh doubles per level)
    handedness: str = "auto"          # auto | normal | mirrored
    last_mirror: bool = False         # auto: decision kept while landmarks can't tell

    def kind_for(self, n: int) -> str:
        if n < self.rigid_min:
            return "none"
        if n < self.affine_min:
            return "rigid"
        if n < self.nonlinear_min:
            return "affine"
        return "nonlinear"


@dataclass
class Pairs:
    labels: list[int]
    slab_frame: np.ndarray            # (n, 3) stack-frame mm
    mri_lps: np.ndarray               # (n, 3) MRI physical (LPS) mm
    sections: list[int]
    excluded: dict[int, str] = field(default_factory=dict)


def landmark_pairs(stack: core.SpecimenStack, geom: SlabGeometry, mri: DonorMRI,
                   conflicts: dict[int, list[str]] | None = None) -> Pairs:
    """Labels usable for fitting: exactly one blob in one section of the slab,
    one blob in the MRI, and not used by another slab."""
    labels, sp, mp, secs, excluded = [], [], [], [], {}
    for lab, places in sorted(stack.locations().items()):
        if lab not in mri.stats:
            excluded[lab] = "not in MRI"
            continue
        if len(places) > 1:
            excluded[lab] = f"in {len(places)} sections"
            continue
        idx, st = places[0]
        if st.n_components > 1:
            excluded[lab] = "several blobs in section"
            continue
        if mri.stats[lab].n_components > 1:
            excluded[lab] = "several blobs in MRI"
            continue
        if conflicts and lab in conflicts:
            excluded[lab] = "also in " + ", ".join(conflicts[lab])
            continue
        labels.append(lab)
        sp.append(geom.com_to_frame(idx, st.com))
        mp.append(mri.voxel_to_lps(mri.stats[lab].com)[0])
        secs.append(idx)
    return Pairs(labels, np.array(sp).reshape(-1, 3), np.array(mp).reshape(-1, 3), secs, excluded)


def handedness_from_points(frame_pts: np.ndarray, mri_pts: np.ndarray) -> bool | None:
    """True if the slab frame is mirror-imaged relative to the MRI (least-squares
    affine has negative determinant); None if the points can't tell (< 4 or flat)."""
    if len(frame_pts) < 4:
        return None
    x = frame_pts - frame_pts.mean(0)
    sv = np.linalg.svd(x, compute_uv=False)
    if sv[0] == 0 or sv[-1] / sv[0] < 0.01:
        return None
    b, *_ = np.linalg.lstsq(np.c_[x, np.ones(len(x))], mri_pts - mri_pts.mean(0), rcond=None)
    return bool(np.linalg.det(b[:3].T) < 0)


@dataclass
class FitResult:
    kind: str                          # none | rigid | affine | nonlinear
    n_pairs: int
    mirror: bool = False
    files: list[str] = field(default_factory=list)       # transform list for antsApplyTransforms
    residuals: dict[int, float] = field(default_factory=dict)
    rms: float = float("nan")
    message: str = ""
    mesh: tuple = ()
    levels: int = 0
    eff_spacing: tuple = ()
    handedness_note: str = ""
    seconds: float = 0.0
    fit_id: int = 0
    _affine: object = None
    _affine_inv: object = None
    _nl: object = None
    mri_bbox: tuple | None = None       # (lo, hi) MRI voxel indices covering the slab

    @property
    def ok(self) -> bool:
        return self.kind != "none" and bool(self.files)

    def summary(self) -> str:
        if not self.ok:
            return self.message or "no transform"
        s = f"{self.kind} · {self.n_pairs} landmarks · RMS {self.rms:.2f} mm"
        if self.kind == "nonlinear" and self.eff_spacing:
            s += " · control points " + "×".join(f"{v:.1f}" for v in self.eff_spacing) + " mm"
        if self.mirror:
            s += " · slab mirrored (x) to match MRI handedness"
        if self.handedness_note:
            s += " · " + self.handedness_note
        return s

    # point mapping (slab physical -> MRI physical) with the ANTs transform objects
    def to_mri(self, phys: np.ndarray) -> np.ndarray:
        ants = _ants()
        out = []
        for p in np.atleast_2d(phys):
            q = list(map(float, p))
            if self._nl is not None:
                q = ants.apply_ants_transform_to_point(self._nl, q)
            q = ants.apply_ants_transform_to_point(self._affine, q)
            out.append(q)
        return np.array(out)

    def to_slab(self, mri_lps: np.ndarray, iters: int = 30) -> np.ndarray:
        """Inverse mapping MRI physical -> slab physical (fixed-point iteration
        for the nonlinear part)."""
        ants = _ants()
        out = []
        for p in np.atleast_2d(mri_lps):
            q = np.array(ants.apply_ants_transform_to_point(self._affine_inv, list(map(float, p))))
            x = q.copy()
            if self._nl is not None:
                for _ in range(iters):
                    fx = np.array(ants.apply_ants_transform_to_point(self._nl, list(map(float, x))))
                    step = q - fx
                    x = x + step
                    if np.linalg.norm(step) < 1e-4:
                        break
            out.append(x)
        return np.array(out)


def _bspline_domain(geom: SlabGeometry, settings: TransformSettings, mirror: bool):
    lo, hi = geom.bounds()
    margin = max(settings.cp_spacing / 2.0, 2.0)
    lo, hi = lo - margin, hi + margin
    size = hi - lo
    dv = float(np.clip(settings.cp_spacing / 4.0, 0.25, 2.0))
    dv = max(dv, float(size.max() / 160.0))
    shape = tuple(int(math.ceil(s / dv)) + 1 for s in size)
    # The finest level should have the requested spacing on every axis; each level
    # doubles the mesh, so the thinnest axis (the slab thickness) limits the levels.
    final = np.maximum(1, np.round(size / settings.cp_spacing)).astype(int)
    levels = int(np.clip(min(settings.levels, 1 + int(math.floor(math.log2(final.min())))), 1, 8))
    init = np.maximum(1, np.round(final / 2 ** (levels - 1))).astype(int)
    eff = tuple(float(s / (m * 2 ** (levels - 1))) for s, m in zip(size, init))
    d = np.eye(3)
    if mirror:
        d[0, 0] = -1.0
    origin = SlabGeometry.to_physical(lo, mirror)
    return dict(origin=tuple(origin), spacing=(dv, dv, dv), size=shape, direction=d), init, levels, eff


_fit_counter = [0]


def fit_transform(pairs: Pairs, geom: SlabGeometry, mri: DonorMRI, settings: TransformSettings,
                  workdir: str | Path) -> FitResult:
    ants = _ants()
    t0 = time.time()
    _fit_counter[0] += 1
    fid = _fit_counter[0]
    n = len(pairs.labels)
    kind = settings.kind_for(n)
    res = FitResult(kind, n, fit_id=fid)
    if kind == "none":
        res.message = f"no transform: {n} landmark pair(s), need {settings.rigid_min}"
        return res
    if settings.handedness == "mirrored":
        mirror = True
    elif settings.handedness == "normal":
        mirror = False
    else:
        found = handedness_from_points(pairs.slab_frame, pairs.mri_lps)
        if found is None:
            mirror = bool(settings.last_mirror)
            res.handedness_note = ("handedness can't be determined from these landmarks (needs 4 not in one "
                                   "plane); using " + ("mirrored" if mirror else "as stored"))
        else:
            mirror = found
    res.mirror = mirror
    fixed = SlabGeometry.to_physical(pairs.slab_frame, mirror)
    moving = pairs.mri_lps
    A = ants.fit_transform_to_paired_points(moving, fixed, transform_type="rigid" if kind == "rigid" else "affine")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    a_path = str(workdir / f"fit{fid:05d}_affine.mat")
    ants.write_transform(A, a_path)
    res._affine, res._affine_inv = A, ants.invert_ants_transform(A)
    files = [a_path]
    if kind == "nonlinear":
        q = np.array([ants.apply_ants_transform_to_point(res._affine_inv, list(map(float, p))) for p in moving])
        dom, init, levels, eff = _bspline_domain(geom, settings, mirror)
        fld = ants.fit_bspline_displacement_field(
            displacement_origins=fixed, displacements=q - fixed, origin=dom["origin"], spacing=dom["spacing"],
            size=dom["size"], direction=dom["direction"], number_of_fitting_levels=levels,
            mesh_size=[int(v) for v in init], spline_order=3, enforce_stationary_boundary=False)
        nl_path = str(workdir / f"fit{fid:05d}_nonlinear.nii.gz")
        ants.image_write(fld, nl_path)
        res._nl = ants.transform_from_displacement_field(fld)
        res.mesh, res.eff_spacing, res.levels = tuple(int(v) for v in init), eff, levels
        files = [nl_path, a_path]                 # antsApplyTransforms order: nonlinear first, then affine
    res.files = files
    mapped = res.to_mri(fixed)
    err = np.linalg.norm(mapped - moving, axis=1)
    res.residuals = {lab: float(e) for lab, e in zip(pairs.labels, err)}
    res.rms = float(np.sqrt(np.mean(err ** 2)))
    # MRI region the slab maps to (to crop the MRI before resampling planes)
    lo, hi = geom.bounds()
    grid = np.stack(np.meshgrid(*[np.linspace(a, b, 5) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    vox = mri.lps_to_voxel(res.to_mri(SlabGeometry.to_physical(grid, mirror)))
    pad = 6
    vlo = np.maximum(0, np.floor(vox.min(0)).astype(int) - pad)
    vhi = np.minimum(np.array(mri.shape) - 1, np.ceil(vox.max(0)).astype(int) + pad)
    res.mri_bbox = (tuple(int(v) for v in vlo), tuple(int(v) for v in vhi)) if np.all(vhi > vlo) else None
    res.seconds = time.time() - t0
    return res


def crop(img, bbox):
    if bbox is None:
        return img
    return _ants().crop_indices(img, list(bbox[0]), [b + 1 for b in bbox[1]])


def resample_plane(res: FitResult, moving, ref, interpolator: str = "linear") -> np.ndarray:
    """antsApplyTransforms of ``moving`` (MRI image/labels) onto a slab plane."""
    ants = _ants()
    out = ants.apply_transforms(ref, moving, res.files, interpolator=interpolator,
                                whichtoinvert=[False] * len(res.files))
    return np.squeeze(out.numpy())


# --------------------------------------------------------------------------- #
# Validation, outputs
# --------------------------------------------------------------------------- #
def validate_slab(stack: core.SpecimenStack, mri: DonorMRI, conflicts: dict[int, list[str]],
                  fit: FitResult | None = None, geom: SlabGeometry | None = None) -> list[core.Issue]:
    issues = [i for i in core.validate([stack], mri)
              if not (i.severity == "info" and "not identified on this hemisphere" in i.message)]
    for i in issues:   # wording: the template is the donor MRI here
        i.message = i.message.replace("3D template", "MRI").replace("Template:", "MRI:")
    for lab in sorted(stack.all_labels() & set(conflicts)):
        places = stack.locations()[lab]
        issues.append(core.Issue("error", f"{stack.id}: label {lab} is also used in "
                                          f"{', '.join(conflicts[lab])}; a landmark can only be in one slab",
                                 label=lab, specimen=stack.id, section_idx=places[0][0]))
    if geom is not None and getattr(stack, "order", "section") == "section":
        nums, counts = np.unique(geom.numbers, return_counts=True)
        if (counts > 1).any():
            issues.append(core.Issue("warning", f"{stack.id}: section number(s) used by more than one image: "
                                                f"{nums[counts > 1][:10].tolist()} (they share a y position)"))
    mine = stack.all_labels()
    others = {lab for lab in conflicts}
    n_unplaced = len(set(mri.stats) - mine - others)
    issues.append(core.Issue("info", f"{stack.id}: {len(mine)} labels placed; {n_unplaced} MRI label(s) not in "
                                     f"this or another slab"))
    if fit is not None:
        issues.append(core.Issue("info", f"Transform: {fit.summary()}"))
        for lab, e in sorted(fit.residuals.items(), key=lambda kv: -kv[1])[:5]:
            if e > 2.0:
                issues.append(core.Issue("warning", f"Label {lab}: fit error {e:.1f} mm (check both positions)",
                                         label=lab, specimen=stack.id,
                                         section_idx=stack.locations()[lab][0][0],
                                         template_slice=mri.com_slice(lab)))
    return issues


class SlabOutputNames(core.OutputNames):
    """Per-slab MRI landmark volume: sub-<sub>_hemi-<h>_chunk-<c>_landmarks_<tag>.nii.gz"""
    key: SlabKey | None = None

    def hemi(self, stack) -> Path:
        k = self.key
        return self.out_dir / f"sub-{k.sub}_hemi-{k.hemisphere}_chunk-{k.chunk}_landmarks{self._t}.nii.gz"


def slab_output_names(out_dir, key: SlabKey, mri_labels_input=None, tag=None,
                      label_suffix=core.LABEL_SUFFIX) -> SlabOutputNames:
    base = core.output_names(out_dir, mri_labels_input, tag, label_suffix)
    if not mri_labels_input:
        base.template_stem = "mri_landmarks"
    n = SlabOutputNames(base.out_dir, base.tag, base.label_suffix, base.template_stem)
    n.key = key
    return n


def fit_table(pairs: Pairs, fit: FitResult | None, stack: core.SpecimenStack, mri: DonorMRI) -> pd.DataFrame:
    rows = []
    for lab, sf, mp, idx in zip(pairs.labels, pairs.slab_frame, pairs.mri_lps, pairs.sections):
        s = stack.sections[idx]
        ras = mp * np.array([-1, -1, 1])
        rows.append(dict(label=lab, section=s.number, position=idx + 1, name=s.name, file=Path(s.raw).name,
                         slab_x_mm=sf[0], slab_y_mm=sf[1], slab_z_mm=sf[2],
                         mri_x_ras=ras[0], mri_y_ras=ras[1], mri_z_ras=ras[2],
                         fit_error_mm=(fit.residuals.get(lab, np.nan) if fit is not None else np.nan),
                         transform=(fit.kind if fit is not None else "none")))
    for lab, why in sorted(pairs.excluded.items()):
        rows.append(dict(label=lab, excluded=why))
    return pd.DataFrame(rows)


def new_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="slab_landmarks_"))
