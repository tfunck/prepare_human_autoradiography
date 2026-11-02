#!/usr/bin/env python3
r"""Slab landmark GUI: one autoradiograph slab <-> the donor MRI.

The main window shows the slab (coronal sections, 0.02 mm apart) and the donor
MRI, each in coronal, sagittal and axial views drawn at their physical aspect.
As in stack_view.py, sections are stacked in (repeat, ligand, name) order and
shifted so the centres of mass of their masks (CSV "mask" column) line up; see
--order and --center (a CSV whose sections each have their own number is
stacked at section number x slice thickness instead).

A floating window shows the MRI resampled into slab space with
antsApplyTransforms, using a transform fitted to the landmark pairs: none
(< 3 pairs), rigid, affine, or affine + B-spline.

    python slab_gui.py --csv sections.csv --mri donor_T1.nii.gz \
        --mri-labels mri_landmarks.nii.gz --sub MR1 --hemisphere L --chunk 6 \
        --out-dir v1.1

Mouse
  Paint mode (P):    left click/drag = paint the current label, right click/drag = erase
                     (slab: coronal section view; MRI: any view)
  Navigate mode (N): left click = move the cursor there, left-drag = pan, right-drag = zoom
  any mode           wheel = zoom, middle-drag = pan, double-click = move the cursor there,
                     Ctrl+click = pick the label under the cursor,
                     Shift+click (Paint) = move the current label here (removes it elsewhere)
  Clicking the slab's sagittal/axial views or the floating window always moves the cursor.

Keys
  P / N (Esc)   Paint / Navigate      Left/Right, Up/Down   previous/next slice (pane under the mouse)
  [ and ]       brush size            F                     fit the views under the mouse
  Ctrl+G        go to label           Ctrl+Z  undo          Ctrl+S  save
  Ctrl+M        show/hide the floating "MRI in slab space" window (also View menu)

Nothing is written unless an annotation changes. Edited labels and a new CSV go
to --out-dir (created on the first save); a folder that already holds files is
only written to after you confirm, or with --overwrite.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PIL import Image
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import landmark_core as core
import landmark_gui as LG
import slab_core as sc

Qt = QtCore.Qt
log = logging.getLogger("landmarks")
LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
SLAB_LABEL_DTYPE = np.uint16
MAX_LABEL = int(np.iinfo(SLAB_LABEL_DTYPE).max)
YELLOW = "#ffd400"
CROSS_PEN = pg.mkPen((80, 200, 255, 140), width=1, style=Qt.PenStyle.DashLine)
AUTORAD_LUT = pg.colormap.get("inferno").getLookupTable(nPts=256, alpha=False)


def _exec(w):
    return w.exec() if hasattr(w, "exec") else w.exec_()


# --------------------------------------------------------------------------- #
# Background work (ANTs calls run on one worker thread; latest request wins)
# --------------------------------------------------------------------------- #
class Async(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ants")
        self.jobs: dict = {}
        self.latest: dict = {}
        self.seq = 0
        self.closed = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._poll)

    def submit(self, key, fn, callback):
        if self.closed:
            return
        self.seq += 1
        seq = self.seq
        self.latest[key] = seq

        def run():
            if self.latest.get(key) != seq:
                return _SKIPPED
            return fn()
        self.jobs[(key, seq)] = (self.pool.submit(run), callback)
        if not self.timer.isActive():
            self.timer.start()

    def busy(self, key=None) -> bool:
        return any(k == key or key is None for (k, _s) in self.jobs)

    def _poll(self):
        for (key, seq), (fut, cb) in list(self.jobs.items()):
            if not fut.done():
                continue
            del self.jobs[(key, seq)]
            if self.latest.get(key) != seq:
                continue
            try:
                out = fut.result()
            except Exception as exc:
                log.error("%s failed: %s", key, exc)
                log.debug(traceback.format_exc())
                cb(None, exc)
                continue
            if out is not _SKIPPED:
                cb(out, None)
        if not self.jobs:
            self.timer.stop()

    def wait(self, timeout=60.0):
        """Run pending callbacks synchronously (tests/saving)."""
        t0 = time.time()
        while self.jobs and time.time() - t0 < timeout:
            QtWidgets.QApplication.processEvents()
            self._poll()
            time.sleep(0.01)


_SKIPPED = object()


# --------------------------------------------------------------------------- #
# One 2-D view of a plane
# --------------------------------------------------------------------------- #
class PaintImage(pg.ImageItem):
    def __init__(self, view):
        super().__init__(autoDownsample=True)
        self.view = view

    def _painting(self):
        return self.view.main.mode == "paint" and self.view.paintable

    def hoverEvent(self, ev):
        if ev.isExit():
            return
        ev.acceptClicks(LEFT)
        if self._painting():
            ev.acceptClicks(RIGHT)
            ev.acceptDrags(LEFT)
            ev.acceptDrags(RIGHT)

    def mouseClickEvent(self, ev):
        b = ev.button()
        if b not in (LEFT, RIGHT) or (b == RIGHT and not self._painting()):
            ev.ignore()
            return
        ev.accept()
        self.view.on_click(ev.pos(), b, ev.modifiers(), ev.double())

    def mouseDragEvent(self, ev):
        b = ev.button()
        if b not in (LEFT, RIGHT) or not self._painting():
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self.view.stroke_begin(ev.buttonDownPos(), erase=(b == RIGHT))
        self.view.stroke_to(ev.pos())
        if ev.isFinish():
            self.view.stroke_end()


class _GV(pg.GraphicsView):
    def __init__(self, view):
        super().__init__()
        self.view = view

    def leaveEvent(self, ev):
        self.view.on_leave()
        super().leaveEvent(ev)


class OrthoView(QtWidgets.QFrame):
    """A plane array indexed [u, v] (u to the right, v up), drawn at physical
    size, with optional label overlay, second-image overlay, crosshair, number
    tags and markers. Optional flips mirror u and/or v on screen."""

    def __init__(self, pane, name, paintable=False):
        super().__init__()
        self.pane, self.main, self.name, self.paintable = pane, pane.main, name, paintable
        self.shape_uv = (1, 1)
        self.spacing = (1.0, 1.0)
        self.flips = (False, False)
        self.lab_uv = None              # labels plane [u, v] (read-only reference)
        self.lab_raster = None
        self.rgba = None
        self._texts: list = []
        self._last = None
        self._has_image = False
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.gv = _GV(self)
        self.gv.setBackground("k")
        self.vb = pg.ViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.gv.setCentralItem(self.vb)
        self.img = PaintImage(self)
        self.over = pg.ImageItem()
        self.over.setZValue(5)
        self.over.setLookupTable(AUTORAD_LUT)
        self.over.hide()
        self.lab = pg.ImageItem()
        self.lab.setZValue(10)
        for it in (self.img, self.over, self.lab):
            self.vb.addItem(it)
        self.vline = pg.InfiniteLine(angle=90, pen=CROSS_PEN, movable=False)
        self.hline = pg.InfiniteLine(angle=0, pen=CROSS_PEN, movable=False)
        for ln in (self.vline, self.hline):
            ln.setZValue(15)
            self.vb.addItem(ln, ignoreBounds=True)
        self.rings = pg.ScatterPlotItem(size=16, pxMode=True)
        self.rings.setZValue(18)
        self.vb.addItem(self.rings, ignoreBounds=True)
        self.goto = pg.ScatterPlotItem(size=30, pxMode=True, brush=pg.mkBrush(None))
        self.goto.setZValue(22)
        self.vb.addItem(self.goto, ignoreBounds=True)
        self.brush_item = QtWidgets.QGraphicsEllipseItem()
        bp = pg.mkPen("#ffffff", width=1)
        bp.setCosmetic(True)
        self.brush_item.setPen(bp)
        self.brush_item.setZValue(16)
        self.brush_item.hide()
        self.vb.addItem(self.brush_item, ignoreBounds=True)
        self.hover = pg.TextItem("", anchor=(-0.2, 1.2), fill=pg.mkBrush(0, 0, 0, 200))
        self.hover.setZValue(30)
        self.hover.hide()
        self.vb.addItem(self.hover, ignoreBounds=True)
        self.gv.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.caption = QtWidgets.QLabel(name, self.gv)
        self.caption.setStyleSheet("color:#bbb; background: rgba(0,0,0,150); padding:1px 4px")
        self.caption.move(4, 4)
        self.note_lbl = QtWidgets.QLabel(self.gv)
        self.note_lbl.hide()
        lay.addWidget(self.gv)

    # -- geometry -------------------------------------------------------------- #
    @property
    def raster_shape(self):
        return self.shape_uv[1], self.shape_uv[0]

    def raster(self, arr):
        t = np.asarray(arr).T[::-1]
        if self.flips[0]:
            t = t[:, ::-1]
        if self.flips[1]:
            t = t[::-1]
        return t

    def rc_to_uv(self, r, c):
        nu, nv = self.shape_uv
        u = (nu - 1 - c) if self.flips[0] else c
        v = r if self.flips[1] else (nv - 1 - r)
        return u, v

    def uv_to_rc(self, u, v):
        nu, nv = self.shape_uv
        c = (nu - 1 - u) if self.flips[0] else u
        r = v if self.flips[1] else (nv - 1 - v)
        return r, c

    def uv_to_view(self, u, v):
        r, c = self.uv_to_rc(u, v)
        return QtCore.QPointF((c + 0.5) * self.spacing[0], (r + 0.5) * self.spacing[1])

    def item_to_uv(self, pos):
        """continuous (u, v) for an ImageItem-local position (x=col, y=row)."""
        nu, nv = self.shape_uv
        c, r = pos.x() - 0.5, pos.y() - 0.5
        u = (nu - 1 - c) if self.flips[0] else c
        v = r if self.flips[1] else (nv - 1 - r)
        return u, v

    def rect(self):
        nu, nv = self.shape_uv
        return QtCore.QRectF(0, 0, nu * self.spacing[0], nv * self.spacing[1])

    # -- content ----------------------------------------------------------------- #
    def show_plane(self, arr_uv, spacing, flips, levels, labels_uv=None):
        arr_uv = np.asarray(arr_uv)
        if arr_uv.ndim != 2 or 0 in arr_uv.shape:
            self.clear()
            return
        first = (not self._has_image) or tuple(arr_uv.shape) != self.shape_uv or tuple(spacing) != self.spacing
        self.shape_uv, self.spacing, self.flips = tuple(arr_uv.shape), tuple(float(s) for s in spacing), tuple(flips)
        self.img.setImage(np.ascontiguousarray(self.raster(arr_uv)), autoLevels=False, levels=levels)
        self.img.setRect(self.rect())
        self._has_image = True
        self.set_labels(labels_uv)
        if first:
            self.vb.autoRange(padding=0.02)

    def clear(self):
        self.img.clear()
        self.lab.clear()
        self.over.clear()
        self._has_image = False
        self.lab_uv = self.lab_raster = self.rgba = None

    def set_levels(self, levels):
        if self._has_image:
            self.img.setLevels(levels)

    def set_overlay(self, arr_uv, alpha):
        if arr_uv is None or alpha <= 0:
            self.over.hide()
            return
        self.over.setImage(np.ascontiguousarray(self.raster(arr_uv)), autoLevels=False, levels=(0, 255))
        self.over.setRect(self.rect())
        self.over.setOpacity(alpha)
        self.over.show()

    def set_labels(self, labels_uv):
        self.hover.hide()
        self.lab_uv = labels_uv
        if labels_uv is None or not np.any(labels_uv):
            self.lab_raster = self.rgba = None
            self.lab.clear()
            return
        self.lab_raster = np.ascontiguousarray(self.raster(labels_uv))
        self.rgba = self._colorize(self.lab_raster)
        self.lab.setImage(self.rgba, autoLevels=False, levels=(0, 255))
        self.lab.setRect(self.rect())

    def _colorize(self, lab):
        out = np.zeros(lab.shape + (4,), np.uint8)
        nz = np.flatnonzero(lab)
        if nz.size:
            out.reshape(-1, 4)[nz] = self.main.lut[LG.color_index(lab.ravel()[nz])]
        return out

    def refresh_label_block(self, r0, r1, c0, c1, labels_uv):
        """Re-colour one display rectangle after painting (labels_uv: full plane)."""
        self.lab_uv = labels_uv
        if self.rgba is None:
            if labels_uv is None:
                return
            self.lab_raster = np.zeros(self.raster_shape, labels_uv.dtype)
            self.rgba = np.zeros(self.raster_shape + (4,), np.uint8)
            self.lab.setImage(self.rgba, autoLevels=False, levels=(0, 255))
            self.lab.setRect(self.rect())
        nu, nv = self.shape_uv
        u0, u1 = ((nu - c1, nu - c0) if self.flips[0] else (c0, c1))
        v0, v1 = ((r0, r1) if self.flips[1] else (nv - r1, nv - r0))
        block = self.raster(labels_uv[u0:u1, v0:v1])
        self.lab_raster[r0:r1, c0:c1] = block
        self.rgba[r0:r1, c0:c1] = self._colorize(block)
        self.lab.updateImage()

    def set_cross(self, u, v):
        if not self._has_image or u is None:
            self.vline.hide()
            self.hline.hide()
            return
        p = self.uv_to_view(u, v)
        self.vline.setPos(p.x())
        self.hline.setPos(p.y())
        self.vline.show()
        self.hline.show()

    def set_tags(self, items):
        """items: [(label, u, v)]"""
        for t in self._texts:
            self.vb.removeItem(t)
        self._texts = []
        if not self.main.show_numbers:
            return
        font = QtGui.QFont()
        font.setBold(True)
        for lab, u, v in items:
            t = pg.TextItem(str(lab), color=LG.label_qcolor(lab), anchor=(0.5, 1.35), fill=pg.mkBrush(0, 0, 0, 150))
            t.setFont(font)
            t.setZValue(25)
            self.vb.addItem(t, ignoreBounds=True)
            t.setPos(self.uv_to_view(u, v))
            self._texts.append(t)

    def set_rings(self, items):
        """items: [(label, u, v, faded)] -> coloured rings (landmarks near this plane)."""
        spots = []
        for lab, u, v, faded in items:
            col = LG.label_qcolor(lab)
            if faded:
                col.setAlpha(110)
            p = self.uv_to_view(u, v)
            spots.append(dict(pos=(p.x(), p.y()), pen=pg.mkPen(col, width=2), brush=pg.mkBrush(None), symbol="o"))
        self.rings.setData(spots)

    def set_goto(self, u=None, v=None, predicted=False):
        if u is None or not self._has_image:
            self.goto.setData([])
            return
        p = self.uv_to_view(u, v)
        pen = pg.mkPen(YELLOW, width=2, style=Qt.PenStyle.DashLine if predicted else Qt.PenStyle.SolidLine)
        self.goto.setData([dict(pos=(p.x(), p.y()), pen=pen, brush=pg.mkBrush(None), symbol="o")])

    def note(self, text, kind=""):
        if not text:
            self.note_lbl.hide()
            return
        self.note_lbl.setText(text)
        self.note_lbl.setWordWrap(True)
        self.note_lbl.setStyleSheet(f"color: {LG.NOTE_COLORS.get(kind, '#ddd')}; background: rgba(0,0,0,175);"
                                    "padding: 2px 6px; border-radius: 3px")
        w = min(max(120, self.gv.width() - 12), self.note_lbl.fontMetrics().boundingRect(text).width() + 18)
        self.note_lbl.setFixedWidth(w)
        self.note_lbl.setFixedHeight(self.note_lbl.heightForWidth(w))
        self.note_lbl.move(4, 26)
        self.note_lbl.show()
        self.note_lbl.raise_()

    def fit_view(self):
        if self._has_image:
            self.vb.autoRange(padding=0.02)

    # -- mouse ------------------------------------------------------------------- #
    def on_leave(self):
        self.hover.hide()
        self.brush_item.hide()

    def _inside(self, r, c):
        nr, nc = self.raster_shape
        return 0 <= r < nr and 0 <= c < nc

    def on_mouse_moved(self, scene_pos):
        if not self._has_image:
            return
        self.main.active_pane = self.pane
        if not self.vb.sceneBoundingRect().contains(scene_pos):
            self.on_leave()
            return
        p = self.img.mapFromScene(scene_pos)
        c, r = int(math.floor(p.x())), int(math.floor(p.y()))
        self._update_brush(p.x(), p.y())
        if not self._inside(r, c):
            self.hover.hide()
            self.main.set_hover_status("")
            return
        tol = int(np.clip(math.ceil(6 * max(self.vb.viewPixelSize()[0] / self.spacing[0],
                                          self.vb.viewPixelSize()[1] / self.spacing[1])), 1, 80))
        lab = core.label_near(self.lab_raster, r, c, tol) if self.lab_raster is not None else 0
        if lab:
            col = LG.label_qcolor(lab).name()
            self.hover.setHtml(f"<span style='font-size:13pt; font-weight:bold; color:{col}'>{lab}</span>")
            self.hover.setPos(self.vb.mapSceneToView(scene_pos))
            self.hover.show()
        else:
            self.hover.hide()
        u, v = self.rc_to_uv(r, c)
        val = float(self.img.image[r, c]) if self.img.image is not None else float("nan")
        self.main.set_hover_status(self.pane.describe_point(self, u, v, val, lab))

    def _update_brush(self, x, y):
        if not (self.main.mode == "paint" and self.paintable) or not self._inside(int(y), int(x)):
            self.brush_item.hide()
            return
        rad = self.main.brush_radius(self)
        rect = QtCore.QRectF(int(x) - rad, int(y) - rad, 2 * rad + 1, 2 * rad + 1)
        self.brush_item.setRect(self.img.mapRectToView(rect))
        self.brush_item.show()

    def on_click(self, pos, button, mods, double):
        r, c = int(math.floor(pos.y())), int(math.floor(pos.x()))
        if not self._inside(r, c):
            return
        self.main.active_pane = self.pane
        if mods & Qt.KeyboardModifier.ControlModifier:
            self.main.pick_label(self, r, c)
            return
        painting = self.main.mode == "paint" and self.paintable
        if double or not painting:
            if button == LEFT:
                self.pane.set_cursor_from(self, *self.item_to_uv(pos))
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and button == LEFT:
            self.main.place_label(self, r, c)
            return
        self.main.begin_stroke(self, erase=(button == RIGHT))
        self.main.paint_segment(self, (pos.x(), pos.y()), (pos.x(), pos.y()))
        self.main.end_stroke()

    def stroke_begin(self, pos, erase):
        self.main.begin_stroke(self, erase=erase)
        self._last = (pos.x(), pos.y())

    def stroke_to(self, pos):
        cur = (pos.x(), pos.y())
        if self._last is None:
            self._last = cur
        self.main.paint_segment(self, self._last, cur)
        self._last = cur
        self._update_brush(*cur)

    def stroke_end(self):
        self._last = None
        self.main.end_stroke()


# --------------------------------------------------------------------------- #
# Panes (three views + controls)
# --------------------------------------------------------------------------- #
def _slider(rng_max):
    s = QtWidgets.QSlider(Qt.Orientation.Horizontal)
    s.setRange(0, max(0, rng_max))
    s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return s


def _tool(text, tip=""):
    b = QtWidgets.QToolButton()
    b.setText(text)
    if tip:
        b.setToolTip(tip)
    return b


class OrthoPane(QtWidgets.QFrame):
    def __init__(self, main, title):
        super().__init__()
        self.main = main
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(4, 4, 4, 4)
        self.lay.setSpacing(3)
        head = QtWidgets.QHBoxLayout()
        self.title_lbl = QtWidgets.QLabel(f"<b>{title}</b>")
        self.info_lbl = QtWidgets.QLabel()
        self.info_lbl.setStyleSheet("color:#9a9a9a")
        fit = _tool("Fit", "Fit all three views (F)")
        fit.clicked.connect(self.fit_views)
        head.addWidget(self.title_lbl)
        head.addSpacing(8)
        head.addWidget(self.info_lbl, 1)
        head.addWidget(fit)
        self.lay.addLayout(head)
        self.views: dict[str, OrthoView] = {}

    def build_views(self, paintable: dict):
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        right = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        for name in ("coronal", "sagittal", "axial"):
            self.views[name] = OrthoView(self, name, paintable.get(name, False))
        split.addWidget(self.views["coronal"])
        right.addWidget(self.views["sagittal"])
        right.addWidget(self.views["axial"])
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([480, 300])
        self.lay.addWidget(split, 1)

    def fit_views(self):
        for v in self.views.values():
            v.fit_view()

    def clear_goto(self):
        for v in self.views.values():
            v.set_goto()

    def contrast_row(self, on_change):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Contrast"))
        self.lo_slider, self.hi_slider = _slider(500), _slider(500)
        self.hi_slider.setRange(500, 1000)
        self.lo_slider.setValue(5)
        self.hi_slider.setValue(995)
        for s in (self.lo_slider, self.hi_slider):
            s.valueChanged.connect(on_change)
            row.addWidget(s, 1)
        auto = _tool("Auto", "Reset contrast to the 0.5-99.5 percentile range")
        auto.clicked.connect(lambda: (self.lo_slider.setValue(5), self.hi_slider.setValue(995)))
        row.addWidget(auto)
        return row

    @property
    def pct(self):
        return self.lo_slider.value() / 10, self.hi_slider.value() / 10


class SlabPane(OrthoPane):
    """Coronal = the current section at full resolution (painting happens here);
    sagittal/axial = the downsampled slab with physical spacing along y."""

    def __init__(self, main, stack, geom, thumbs, flip_lr):
        super().__init__(main, f"Slab · {stack.id}" + ("-flipped-lr" if flip_lr else ""))
        self.stack, self.geom, self.thumbs, self.flip_lr = stack, geom, thumbs, flip_lr
        self.idx = 0
        self.x, self.z = (float(v) for v in geom.centre)
        self.cur = None                      # current section view data
        self.build_views({"coronal": True})
        nav = QtWidgets.QHBoxLayout()
        self.minus, self.plus = _tool("−", "Previous section"), _tool("+", "Next section")
        self.sec_slider = _slider(geom.n - 1)
        self.goto = QtWidgets.QLineEdit()
        if getattr(stack, "order", "section") == "stack":
            self.goto.setPlaceholderText("position / name")
            self.goto.setFixedWidth(120)
            self.goto.setToolTip("Type a position in the stack (1-%d) or part of a section name "
                                 "(e.g. dpmg#6555#01) and press Enter" % geom.n)
        else:
            self.goto.setPlaceholderText("section #")
            self.goto.setFixedWidth(84)
            self.goto.setToolTip("Type a section number and press Enter")
        nav.addWidget(QtWidgets.QLabel("Section"))
        nav.addWidget(self.minus)
        nav.addWidget(self.sec_slider, 1)
        nav.addWidget(self.plus)
        nav.addWidget(self.goto)
        self.lay.addLayout(nav)
        xz = QtWidgets.QHBoxLayout()
        self.x_slider, self.z_slider = _slider(geom.tshape[0] - 1), _slider(geom.tshape[1] - 1)
        self.x_lbl, self.z_lbl = QtWidgets.QLabel(), QtWidgets.QLabel()
        for lbl in (self.x_lbl, self.z_lbl):
            lbl.setMinimumWidth(70)
        xz.addWidget(QtWidgets.QLabel("Sagittal x"))
        xz.addWidget(self.x_slider, 1)
        xz.addWidget(self.x_lbl)
        xz.addWidget(QtWidgets.QLabel("Axial z"))
        xz.addWidget(self.z_slider, 1)
        xz.addWidget(self.z_lbl)
        self.lay.addLayout(xz)
        self.lay.addLayout(self.contrast_row(self._on_contrast))
        self.build_lbl = QtWidgets.QLabel()
        self.build_lbl.setStyleSheet("color:#9a9a9a")
        self.lay.addWidget(self.build_lbl)
        self._sec_timer = QtCore.QTimer(self)
        self._sec_timer.setSingleShot(True)
        self._sec_timer.setInterval(80)
        self._sec_timer.timeout.connect(lambda: self.set_section(self.sec_slider.value()))
        self.sec_slider.valueChanged.connect(lambda v: (self.info_lbl.setText(self.describe_section(v)),
                                                         self._sec_timer.start()))
        self.minus.clicked.connect(lambda: self.step(-1))
        self.plus.clicked.connect(lambda: self.step(1))
        self.goto.returnPressed.connect(self._on_goto)
        self.x_slider.valueChanged.connect(lambda v: self.set_xz(x=(v + 0.5) * geom.ts))
        self.z_slider.valueChanged.connect(lambda v: self.set_xz(z=(v + 0.5) * geom.ts))

    # -- state ------------------------------------------------------------------ #
    def describe_section(self, idx):
        return f"{self.stack.describe(idx)} · y = {self.geom.y[idx]:.2f} mm · {idx + 1}/{self.geom.n}"

    def step(self, d):
        self.set_section(self.idx + d)

    def _on_goto(self):
        text = self.goto.text().strip()
        if not text:
            return
        idx, msg = self.stack.find(text)
        if idx is None:
            self.views["coronal"].note(msg, "error")
            return
        self.set_section(idx, force=True)
        if msg:
            self.views["coronal"].note(msg, "warn")
        self.goto.clear()

    def set_section(self, idx, force=False):
        idx = int(np.clip(idx, 0, self.geom.n - 1))
        if idx == self.idx and not force and self.cur is not None:
            return
        self.clear_goto()
        self.main.warp.pane.clear_goto()
        self.idx = idx
        self.thumbs.prioritise(idx)
        self.sec_slider.blockSignals(True)
        self.sec_slider.setValue(idx)
        self.sec_slider.blockSignals(False)
        self.views["coronal"].note("")
        try:
            self.render_coronal()
        except Exception as exc:
            log.exception("Cannot display section %s", idx)
            self.views["coronal"].note(f"cannot display: {exc}", "error")
        self.render_thumb_views()
        self.main.on_slab_cursor()

    def set_xz(self, x=None, z=None):
        g = self.geom
        if x is not None or z is not None:
            self.views["sagittal"].set_goto()
            self.views["axial"].set_goto()
            self.main.warp.pane.clear_goto()
        if x is not None:
            self.x = float(np.clip(x, 0, g.extent[0]))
        if z is not None:
            self.z = float(np.clip(z, 0, g.extent[1]))
        for s, val in ((self.x_slider, self.x), (self.z_slider, self.z)):
            s.blockSignals(True)
            s.setValue(int(np.clip(val / g.ts, 0, s.maximum())))
            s.blockSignals(False)
        self.x_lbl.setText(f"{self.x:.1f} mm")
        self.z_lbl.setText(f"{self.z:.1f} mm")
        self.render_thumb_views()
        self.update_cross()
        self.main.on_slab_cursor()

    def set_cursor_from(self, view, u, v):
        g = self.geom
        if view.name == "coronal":
            p = g.section_to_frame(self.idx, u, v)
            self.set_xz(x=p[0], z=p[2])
        elif view.name == "sagittal":
            self.set_section(g.nearest_section(g.row_to_y(u)))
            self.set_xz(z=(v + 0.5) * g.ts)
        else:
            self.set_section(g.nearest_section(g.row_to_y(v)))
            self.set_xz(x=(u + 0.5) * g.ts)

    # -- rendering ------------------------------------------------------------------ #
    def section_levels(self):
        return core.levels_from_sample(self.cur["sample"], *self.pct)

    def _on_contrast(self, _v=None):
        if self.cur is not None:
            self.views["coronal"].set_levels(self.section_levels())

    def render_coronal(self):
        arr, meta, sample = self.stack.image(self.idx)
        self.cur = dict(arr=arr, meta=meta, sample=sample)
        sx, sz = self.geom.spacing[self.idx]
        self.views["coronal"].show_plane(self.geom.to_uv(self.idx, arr), (sx, sz), (self.flip_lr, False),
                                         self.section_levels(), self.section_labels_uv(self.idx))
        self.info_lbl.setText(self.describe_section(self.idx))
        for d in (1, -1, 2):
            self.stack.prefetch(self.idx + d)
        self.update_annotations()

    def render_thumb_views(self):
        g = self.geom
        self.views["sagittal"].show_plane(self.thumbs.sagittal(self.x), (g.ty, g.ts), (False, False), (0, 255))
        self.views["axial"].show_plane(self.thumbs.axial(self.z), (g.ts, g.ty), (self.flip_lr, False), (0, 255))
        done, n = self.thumbs.progress
        self.build_lbl.setText("" if done >= n else f"Building the slab's sagittal/axial views: {done}/{n} sections")
        self.update_annotations()

    def landmarks(self):
        """[(label, frame xyz, section idx)] for every label placed in this slab."""
        out = []
        for lab, places in self.stack.locations().items():
            for idx, st in places:
                out.append((lab, self.geom.com_to_frame(idx, st.com), idx))
        return out

    def update_cross(self):
        g = self.geom
        u, v = g.frame_to_section(self.idx, self.x, self.z)
        self.views["coronal"].set_cross(u, v)
        row = g.y_to_row(g.y[self.idx])
        self.views["sagittal"].set_cross(row, self.z / g.ts - 0.5)
        self.views["axial"].set_cross(self.x / g.ts - 0.5, row)

    def update_annotations(self):
        g = self.geom
        st = self.stack.stats.get(self.idx, {})
        self.views["coronal"].set_tags([(lab, *self.geom.native_to_uv(self.idx, s.com[0], s.com[1]))
                                        for lab, s in st.items()])
        tol = max(1.0, 3 * g.ts)
        sag, axi = [], []
        for lab, p, idx in self.landmarks():
            row = g.y_to_row(p[1])
            if abs(p[0] - self.x) <= tol:
                sag.append((lab, row, p[2] / g.ts - 0.5, abs(p[0] - self.x) > g.ts))
            if abs(p[2] - self.z) <= tol:
                axi.append((lab, p[0] / g.ts - 0.5, row, abs(p[2] - self.z) > g.ts))
        self.views["sagittal"].set_rings(sag)
        self.views["axial"].set_rings(axi)
        self.views["sagittal"].set_tags([(lab, u, v) for lab, u, v, _f in sag])
        self.views["axial"].set_tags([(lab, u, v) for lab, u, v, _f in axi])
        labs = sorted(st)
        txt = ", ".join(map(str, labs[:10])) + (" …" if len(labs) > 10 else "")
        self.title_lbl.setText(f"<b>Slab · {self.stack.id}{'-flipped-lr' if self.flip_lr else ''}</b>"
                               + (f"  <span style='color:#bbb'>[{txt}]</span>" if labs else ""))
        self.update_cross()

    # -- painting interface ------------------------------------------------------------ #
    def target(self, view, create=True):
        return self.stack.get_labels(self.idx, create=create)

    def to_target_flat(self, view, u, v):
        return self.geom.uv_to_native_flat(self.idx, u, v)

    def target_key(self, view):
        return ("slab", self.idx)

    def section_labels_uv(self, idx):
        return self.geom.to_uv(idx, self.stack.labels.get(idx))

    def labels_uv(self, view):
        return self.section_labels_uv(self.idx)

    def describe_point(self, view, u, v, val, lab):
        g = self.geom
        if view.name == "coronal":
            p = g.section_to_frame(self.idx, u, v)
            if g.raster[self.idx]:
                where = f"{self.stack.describe(self.idx)} · row {int(g.shapes[self.idx][1]) - 1 - v}, column {u}"
            else:
                where = f"{self.stack.describe(self.idx)} · pixel ({u}, {v})"
        elif view.name == "sagittal":
            y = g.row_to_y(u)
            p = np.array([self.x, y, (v + 0.5) * g.ts])
            where = f"near {self.stack.describe(g.nearest_section(y))}"
        else:
            y = g.row_to_y(v)
            p = np.array([(u + 0.5) * g.ts, y, self.z])
            where = f"near {self.stack.describe(g.nearest_section(y))}"
        return (f"slab · {where} · ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) mm · value {val:.4g}"
                + (f" · label {lab}" if lab else ""))

    def state(self):
        return dict(idx=self.idx, raw=self.stack.sections[self.idx].raw, x=self.x, z=self.z,
                    lo=self.lo_slider.value(), hi=self.hi_slider.value())


class MRIPane(OrthoPane):
    def __init__(self, main, mri: sc.DonorMRI):
        super().__init__(main, f"Donor MRI · {''.join(mri.axcodes)}")
        self.mri = mri
        self.cursor = [n // 2 for n in mri.shape]
        self.build_views({"coronal": True, "sagittal": True, "axial": True})
        row = QtWidgets.QHBoxLayout()
        self.sliders, self.slider_lbls = {}, {}
        for name in ("coronal", "sagittal", "axial"):
            a = mri.plane_axes(name)[0]
            s = _slider(mri.shape[a] - 1)
            s.valueChanged.connect(lambda v, n=name: self.set_plane(n, v))
            lbl = QtWidgets.QLabel()
            lbl.setMinimumWidth(48)
            row.addWidget(QtWidgets.QLabel(name.capitalize()))
            row.addWidget(s, 1)
            row.addWidget(lbl)
            self.sliders[name], self.slider_lbls[name] = s, lbl
        self.lay.addLayout(row)
        self.lay.addLayout(self.contrast_row(self._on_contrast))

    def levels(self):
        return core.levels_from_sample(self.mri.sample, *self.pct)

    def _on_contrast(self, _v=None):
        for v in self.views.values():
            v.set_levels(self.levels())

    def set_plane(self, name, k):
        a = self.mri.plane_axes(name)[0]
        k = int(np.clip(k, 0, self.mri.shape[a] - 1))
        if k != self.cursor[a]:
            self.views[name].set_goto()
        self.cursor[a] = k
        self.render()

    def step(self, d):
        self.set_plane("coronal", self.cursor[self.mri.ax_cor] + d)

    def set_cursor(self, ijk):
        new = [int(np.clip(round(v), 0, n - 1)) for v, n in zip(ijk, self.mri.shape)]
        if new != self.cursor:
            self.clear_goto()
        self.cursor = new
        self.render()

    def set_cursor_from(self, view, u, v):
        a, au, av, _ = self.mri.plane_axes(view.name)
        c = list(self.cursor)
        c[au], c[av] = int(round(u)), int(round(v))
        self.set_cursor(c)

    def render(self, names=("coronal", "sagittal", "axial")):
        m = self.mri
        for name in names:
            a, au, av, flips = m.plane_axes(name)
            k = self.cursor[a]
            self.views[name].show_plane(m.plane(m.image, name, k), (m.zooms[au], m.zooms[av]), flips,
                                        self.levels(), m.plane(m.labels, name, k))
            s = self.sliders[name]
            s.blockSignals(True)
            s.setValue(k)
            s.blockSignals(False)
            self.slider_lbls[name].setText(f"{'ijk'[a]}={k}")
        self.update_annotations()

    def update_annotations(self):
        m = self.mri
        for name, view in self.views.items():
            a, au, av, _ = m.plane_axes(name)
            st = core.compute_stats(np.ascontiguousarray(m.plane(m.labels, name, self.cursor[a])))
            view.set_tags([(lab, s.com[0], s.com[1]) for lab, s in st.items()])
            view.set_cross(self.cursor[au], self.cursor[av])
        self.info_lbl.setText("voxel (" + ", ".join(map(str, self.cursor)) + ")")

    def target(self, view, create=True):
        return self.mri.labels

    def to_target_flat(self, view, u, v):
        a = self.mri.plane_axes(view.name)[0]
        flat = np.asarray(u, np.int64) * view.shape_uv[1] + np.asarray(v, np.int64)
        return self.mri.plane_flat_to_volume(view.name, self.cursor[a], flat)

    def target_key(self, view):
        return (sc.MRI_KEY,)

    def labels_uv(self, view):
        a = self.mri.plane_axes(view.name)[0]
        return self.mri.plane(self.mri.labels, view.name, self.cursor[a])

    def describe_point(self, view, u, v, val, lab):
        a, au, av, _ = self.mri.plane_axes(view.name)
        ijk = list(self.cursor)
        ijk[au], ijk[av] = int(u), int(v)
        ras = self.mri.world(ijk)
        return (f"MRI · voxel ({ijk[0]}, {ijk[1]}, {ijk[2]}) · RAS ({ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f}) mm"
                f" · value {val:.4g}" + (f" · label {lab}" if lab else ""))

    def state(self):
        return dict(cursor=self.cursor, lo=self.lo_slider.value(), hi=self.hi_slider.value())


class WarpWindow(QtWidgets.QWidget):
    """The MRI resampled into slab space (follows the slab cursor)."""

    def __init__(self, main):
        # A tool window owned by the main window: the window manager keeps it
        # above the main window instead of letting the main window cover it.
        super().__init__(main, Qt.WindowType.Tool)
        mac_keep = getattr(Qt.WidgetAttribute, "WA_MacAlwaysShowToolWindow", None)
        if mac_keep is not None:
            self.setAttribute(mac_keep, True)      # macOS: don't hide it when the app loses focus
        self.main = main
        self.setWindowTitle("MRI in slab space")
        self.resize(900, 640)
        self._placed = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.status = QtWidgets.QLabel("no transform yet")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.pane = _WarpPane(main, self)
        lay.addWidget(self.pane, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Autoradiograph overlay"))
        self.alpha = _slider(100)
        self.alpha.setValue(0)
        self.alpha.valueChanged.connect(lambda _v: self.pane.render_overlays())
        row.addWidget(self.alpha, 1)
        self.labels_cb = QtWidgets.QCheckBox("MRI labels")
        self.labels_cb.setChecked(True)
        self.labels_cb.toggled.connect(lambda _v: self.pane.render_all())
        row.addWidget(self.labels_cb)
        lay.addLayout(row)

    def place(self):
        """Put the window over the lower-right part of the main window, on screen."""
        main = self.main.frameGeometry()
        screen = (self.main.screen().availableGeometry() if hasattr(self.main, "screen") and self.main.screen()
                  else QtWidgets.QApplication.desktop().availableGeometry(self.main))
        w = min(self.width(), screen.width() - 40)
        h = min(self.height(), screen.height() - 80)
        x = main.right() - w - 30
        y = main.bottom() - h - 60
        x = max(screen.left() + 10, min(x, screen.right() - w - 10))
        y = max(screen.top() + 30, min(y, screen.bottom() - h - 10))
        self.setGeometry(x, y, w, h)
        self._placed = True

    def bring_up(self):
        if not self._placed:
            self.place()
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, ev):
        self.main.warp_action.setChecked(False)
        ev.accept()


class _WarpPane(OrthoPane):
    def __init__(self, main, window):
        super().__init__(main, "MRI → slab")
        self.window = window
        self.build_views({})
        self.planes: dict = {}               # name -> (img_uv, lab_uv) for the current request
        self.lay.addLayout(self.contrast_row(lambda _v: self.render_all()))

    @property
    def slab(self):
        return self.main.slab

    def set_cursor_from(self, view, u, v):
        g = self.slab.geom
        if view.name == "coronal":
            self.slab.set_xz(x=(u + 0.5) * g.cs, z=(v + 0.5) * g.cs)
        else:
            self.slab.set_cursor_from(self.slab.views[view.name], u, v)

    def step(self, d):
        self.slab.step(d)

    def describe_point(self, view, u, v, val, lab):
        return f"MRI resampled into slab space · value {val:.4g}" + (f" · MRI label {lab}" if lab else "")

    def geometry_for(self, name):
        g, s = self.slab.geom, self.slab
        if name == "coronal":
            return (g.cs, g.cs), (s.flip_lr, False)
        if name == "sagittal":
            return (g.ty, g.ts), (False, False)
        return (g.ts, g.ty), (s.flip_lr, False)

    def set_plane(self, name, img_uv, lab_uv):
        self.planes[name] = (img_uv, lab_uv)
        self.render_all((name,))

    def render_all(self, names=("coronal", "sagittal", "axial")):
        mri = self.main.mri
        levels = core.levels_from_sample(mri.sample, *self.pct)
        for name in names:
            if name not in self.planes:
                continue
            img, lab = self.planes[name]
            spacing, flips = self.geometry_for(name)
            self.views[name].show_plane(img, spacing, flips, levels,
                                        lab if self.window.labels_cb.isChecked() else None)
            if lab is not None and self.window.labels_cb.isChecked():
                st = core.compute_stats(lab.astype(np.int64))
                self.views[name].set_tags([(k, s.com[0], s.com[1]) for k, s in st.items()])
            else:
                self.views[name].set_tags([])
        self.render_overlays(names)
        self.update_annotations()

    def render_overlays(self, names=("coronal", "sagittal", "axial")):
        a = self.window.alpha.value() / 100.0
        s = self.slab
        for name in names:
            if name not in self.planes:
                continue
            if name == "coronal":
                ov = s.coronal_on_grid() if a > 0 else None
            elif name == "sagittal":
                ov = s.thumbs.sagittal(s.x)
            else:
                ov = s.thumbs.axial(s.z)
            self.views[name].set_overlay(ov, a)

    def update_annotations(self):
        s, g = self.slab, self.slab.geom
        row = g.y_to_row(g.y[s.idx])
        cor, sag, axi = [], [], []
        tol = max(1.0, 3 * g.ts)
        for lab, p, idx in s.landmarks():
            if idx == s.idx:
                cor.append((lab, p[0] / g.cs - 0.5, p[2] / g.cs - 0.5, False))
            r = g.y_to_row(p[1])
            if abs(p[0] - s.x) <= tol:
                sag.append((lab, r, p[2] / g.ts - 0.5, abs(p[0] - s.x) > g.ts))
            if abs(p[2] - s.z) <= tol:
                axi.append((lab, p[0] / g.ts - 0.5, r, abs(p[2] - s.z) > g.ts))
        for name, items in (("coronal", cor), ("sagittal", sag), ("axial", axi)):
            self.views[name].set_rings(items)
        self.views["coronal"].set_cross(s.x / g.cs - 0.5, s.z / g.cs - 0.5)
        self.views["sagittal"].set_cross(row, s.z / g.ts - 0.5)
        self.views["axial"].set_cross(s.x / g.ts - 0.5, row)


def _coronal_on_grid(self):
    """Current section, contrast-normalised, on the floating window's coronal grid."""
    if self.cur is None:
        return None
    key = (self.idx, self.lo_slider.value(), self.hi_slider.value())
    if getattr(self, "_cog_key", None) == key:
        return self._cog
    g = self.geom
    lo, hi = self.section_levels()
    img8 = (np.clip((self.geom.to_uv(self.idx, self.cur["arr"]) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    ex, ez = g.shapes[self.idx] * g.spacing[self.idx]
    tx, tz = max(1, int(round(ex / g.cs))), max(1, int(round(ez / g.cs)))
    small = np.asarray(Image.fromarray(img8).resize((tz, tx), sc.BOX))
    out = np.zeros(g.cshape, np.uint8)
    a0, b0 = int(round(g.offset[self.idx][0] / g.cs)), int(round(g.offset[self.idx][1] / g.cs))
    a1, b1 = min(g.cshape[0], a0 + tx), min(g.cshape[1], b0 + tz)
    out[a0:a1, b0:b1] = small[: a1 - a0, : b1 - b0]
    self._cog_key, self._cog = key, out
    return out


SlabPane.coronal_on_grid = _coronal_on_grid


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ds: dict, out_dir: Path, settings: sc.TransformSettings, session_info: dict, state: dict,
                 autosave_min=5.0, overwrite=False, version_tag=None, label_suffix=core.LABEL_SUFFIX,
                 flip_lr=False, label_format="match"):
        super().__init__()
        self.label_format = label_format
        self.ds = ds
        self.key: sc.SlabKey = ds["key"]
        self.stack: core.SpecimenStack = ds["stack"]
        self.mri: sc.DonorMRI = ds["mri"]
        self.geom: sc.SlabGeometry = ds["geom"]
        self.thumbs: sc.ThumbStack = ds["thumbs"]
        self.conflicts: dict = ds["conflicts"]
        self.settings = settings
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.session_info = session_info
        self.overwrite, self.version_tag, self.label_suffix = overwrite, version_tag, label_suffix
        self._out_owned = False
        self.mode = "navigate"
        self.undo_stack: list = []
        self._stroke = None
        self._saved_top = None
        self.dirty = False
        self.fit: sc.FitResult | None = None
        self.pairs: sc.Pairs | None = None
        self.workdir = sc.new_workdir()
        self.show_numbers = bool(state.get("show_numbers", True))
        self.alpha = int(state.get("overlay_alpha", 190))
        self.lut = self._make_lut(self.alpha)
        self.issues: list = []
        self._load_warnings = ([core.Issue("warning", w) for w in ds.get("warnings", [])]
                               + [core.Issue("info", n) for n in ds.get("notes", [])])
        self._icons: dict = {}
        self._lab_crop_cache = None
        self.async_ = Async(self)
        self.resize(1800, 1050)
        self._update_title()

        self._build_toolbars(state)
        self.slab = SlabPane(self, self.stack, self.geom, self.thumbs, flip_lr)
        self.mri_pane = MRIPane(self, self.mri)
        self.panes = [self.slab, self.mri_pane]
        self.active_pane = self.slab
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.slab)
        split.addWidget(self.mri_pane)
        split.setSizes([900, 700])
        self.setCentralWidget(split)
        self.warp = WarpWindow(self)
        self._build_dock(state)
        self._build_menu()
        self._build_shortcuts()
        self.hover_lbl = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.hover_lbl)

        self._fit_timer = QtCore.QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(300)
        self._fit_timer.timeout.connect(self.run_fit)
        self._warp_timer = QtCore.QTimer(self)
        self._warp_timer.setSingleShot(True)
        self._warp_timer.setInterval(120)
        self._warp_timer.timeout.connect(self.request_warp)
        self._table_timer = QtCore.QTimer(self)
        self._table_timer.setSingleShot(True)
        self._table_timer.setInterval(150)
        self._table_timer.timeout.connect(self.refresh_table)
        self._thumb_timer = QtCore.QTimer(self)
        self._thumb_timer.setInterval(1000)
        self._thumb_timer.timeout.connect(self._thumb_tick)
        self._thumb_timer.start()

        ss, ms = state.get("slab", {}), state.get("mri", {})
        for pane, st in ((self.slab, ss), (self.mri_pane, ms)):
            if "lo" in st:
                pane.lo_slider.setValue(int(st["lo"]))
                pane.hi_slider.setValue(int(st["hi"]))
        first = min(self.stack.stats, default=self.geom.n // 2)
        self.slab.x = float(ss.get("x", self.slab.x))
        self.slab.z = float(ss.get("z", self.slab.z))
        idx = int(ss.get("idx", first))
        if ss.get("raw"):              # by file: the stacking order may have changed since
            idx = next((i for i, sec in enumerate(self.stack.sections) if sec.raw == ss["raw"]), idx)
        self.slab.set_section(idx, force=True)
        self.slab.set_xz()
        if "cursor" in ms:
            self.mri_pane.set_cursor(ms["cursor"])
        else:
            lab = min(self.mri.stats, default=None)
            self.mri_pane.set_cursor(np.rint(self.mri.stats[lab].com) if lab is not None else self.mri_pane.cursor)
        self.refresh_table()
        self.populate_issues(sc.validate_slab(self.stack, self.mri, self.conflicts, None, self.geom), switch=False)
        self.set_mode("navigate")
        self.statusBar().showMessage("P paint (left paint, right erase) · N navigate · double-click moves the cursor · "
                                     "Ctrl+click picks a label · Ctrl+Z undo · Ctrl+S save", 15000)
        self._autosave = QtCore.QTimer(self)
        if autosave_min and autosave_min > 0:
            self._autosave.setInterval(int(autosave_min * 60_000))
            self._autosave.timeout.connect(self._on_autosave)
            self._autosave.start()
        self.schedule_fit(now=True)
        # The floating window is shown once the main window is on screen (showEvent);
        # shown earlier, the main window would open on top of it and hide it.
        self._warp_on_start = bool(state.get("warp_visible", True))
        self._shown_once = False

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._shown_once:
            self._shown_once = True
            if self._warp_on_start:
                QtCore.QTimer.singleShot(0, lambda: self.warp_action.setChecked(True))

    # -- UI ----------------------------------------------------------------------- #
    def _update_title(self):
        self.setWindowTitle(f"Slab landmarks — {self.key.id} — saves to {self.out_dir}[*]")

    def _make_lut(self, alpha):
        lut = LG.PALETTE.copy()
        lut[1:, 3] = alpha
        return lut

    def _build_toolbars(self, state):
        tb = self.addToolBar("Edit")
        tb.setMovable(False)
        group = (QtGui.QActionGroup if hasattr(QtGui, "QActionGroup") else QtWidgets.QActionGroup)(self)
        self.mode_actions = {}
        for key, text, tip in (("navigate", "Navigate", "Left click moves the cursor; drag pans (N or Esc)"),
                               ("paint", "Paint", "Left = paint the current label, right = erase (P)")):
            act = LG.QAction(text, self)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _c=False, k=key: self.set_mode(k))
            group.addAction(act)
            tb.addAction(act)
            self.mode_actions[key] = act
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Label "))
        self.label_spin = QtWidgets.QSpinBox()
        self.label_spin.setRange(1, MAX_LABEL)
        self.label_spin.setValue(int(state.get("current_label", 1)))
        self.label_spin.setMinimumWidth(80)
        tb.addWidget(self.label_spin)
        self.swatch = QtWidgets.QLabel()
        self.swatch.setFixedSize(18, 18)
        tb.addWidget(self.swatch)
        self.label_spin.valueChanged.connect(self._update_swatch)
        self._update_swatch()
        nf = _tool("Next free", "One more than any label in use (this slab, other slabs, MRI)")
        nf.clicked.connect(self.next_free_label)
        tb.addWidget(nf)
        self.protect_cb = QtWidgets.QCheckBox("Don't overwrite other labels")
        self.protect_cb.setChecked(bool(state.get("protect", True)))
        tb.addWidget(self.protect_cb)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Brush: slab "))
        self.brush_slab = QtWidgets.QSpinBox()
        self.brush_slab.setRange(0, 500)
        default = max(2, int(round(0.3 / float(self.geom.spacing.min()))))
        self.brush_slab.setValue(int(state.get("brush_slab", default)))
        self.brush_slab.setSuffix(" px")
        self.brush_slab.valueChanged.connect(self._brush_tip)
        tb.addWidget(self.brush_slab)
        tb.addWidget(QtWidgets.QLabel(" MRI "))
        self.brush_mri = QtWidgets.QSpinBox()
        self.brush_mri.setRange(0, 50)
        self.brush_mri.setValue(int(state.get("brush_mri", 1)))
        self.brush_mri.setSuffix(" vox")
        tb.addWidget(self.brush_mri)
        self._brush_tip()
        tb.addSeparator()
        u = _tool("Undo", "Ctrl+Z")
        u.clicked.connect(self.undo)
        tb.addWidget(u)

        self.addToolBarBreak()
        tb2 = self.addToolBar("Go")
        tb2.setMovable(False)
        tb2.addWidget(QtWidgets.QLabel(" Go to label "))
        self.goto_label = QtWidgets.QLineEdit()
        self.goto_label.setFixedWidth(80)
        self.goto_label.setPlaceholderText("e.g. 12")
        self.goto_label.returnPressed.connect(self._on_goto_label)
        tb2.addWidget(self.goto_label)
        go = _tool("Go")
        go.clicked.connect(self._on_goto_label)
        tb2.addWidget(go)
        tb2.addSeparator()
        tb2.addWidget(QtWidgets.QLabel(" Label opacity "))
        self.alpha_slider = _slider(255)
        self.alpha_slider.setValue(self.alpha)
        self.alpha_slider.setFixedWidth(110)
        self.alpha_slider.valueChanged.connect(self._on_alpha)
        tb2.addWidget(self.alpha_slider)
        self.numbers_cb = QtWidgets.QCheckBox("Show label numbers")
        self.numbers_cb.setChecked(self.show_numbers)
        self.numbers_cb.toggled.connect(self._on_numbers)
        tb2.addWidget(self.numbers_cb)
        tb2.addSeparator()
        for text, slot, tip in (("Validate", self.validate_now, "Check uniqueness and the slab/MRI correspondence"),
                                ("Save", self.save, "Save all outputs (Ctrl+S)")):
            b = _tool(text, tip)
            b.clicked.connect(slot)
            tb2.addWidget(b)
        tb2.addSeparator()
        self.warp_action = LG.QAction("MRI in slab space", self)
        self.warp_action.setCheckable(True)
        self.warp_action.setToolTip("Show/hide the floating window with the MRI resampled into slab space (Ctrl+M)")
        self.warp_action.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self.warp_action.toggled.connect(self._toggle_warp)
        tb2.addAction(self.warp_action)

    def _build_dock(self, state):
        dock = QtWidgets.QDockWidget("Landmarks", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.tabs = QtWidgets.QTabWidget()
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._on_table_click)
        self.issues_list = QtWidgets.QListWidget()
        self.issues_list.itemClicked.connect(self._on_issue)
        self.issues_list.itemActivated.connect(self._on_issue)
        self.tabs.addTab(self.table, "Labels")
        self.tabs.addTab(self.issues_list, "Issues")
        self.tabs.addTab(self._transform_tab(), "Transform")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(420)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _transform_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        s = self.settings
        def spin(lo, hi, val, tip, suffix=""):
            b = QtWidgets.QSpinBox()
            b.setRange(lo, hi)
            b.setValue(int(val))
            b.setSuffix(suffix)
            b.setToolTip(tip)
            b.valueChanged.connect(self._on_settings)
            return b
        self.rigid_min = spin(3, 999, s.rigid_min, "Fewer pairs than this: no transform")
        self.affine_min = spin(3, 999, s.affine_min, "From this many pairs: affine")
        self.nl_min = spin(3, 9999, s.nonlinear_min, "From this many pairs: affine + B-spline")
        self.cp_spacing = QtWidgets.QDoubleSpinBox()
        self.cp_spacing.setRange(0.5, 500)
        self.cp_spacing.setDecimals(1)
        self.cp_spacing.setSuffix(" mm")
        self.cp_spacing.setValue(s.cp_spacing)
        self.cp_spacing.setToolTip("B-spline control-point spacing at the finest level (slab space)")
        self.cp_spacing.valueChanged.connect(self._on_settings)
        self.levels = spin(1, 8, s.levels, "Maximum number of B-spline fitting levels (mesh doubles per level)")
        self.hand = QtWidgets.QComboBox()
        self.hand.addItems(["auto", "normal", "mirrored"])
        self.hand.setCurrentText(s.handedness)
        self.hand.setToolTip("Whether the slab is mirror-imaged relative to the MRI. auto = from the landmarks "
                             "(needs 4 not in one plane). antspyx rigid/affine fits cannot mirror.")
        self.hand.currentTextChanged.connect(self._on_settings)
        form.addRow("Rigid from", self.rigid_min)
        form.addRow("Affine from", self.affine_min)
        form.addRow("Nonlinear from", self.nl_min)
        form.addRow("Control-point spacing", self.cp_spacing)
        form.addRow("B-spline levels (max)", self.levels)
        form.addRow("Slab handedness", self.hand)
        self.fit_status = QtWidgets.QLabel("…")
        self.fit_status.setWordWrap(True)
        form.addRow(self.fit_status)
        return w

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, scut in (("Save", self.save, "Ctrl+S"), ("Quit", self.close, "Ctrl+Q")):
            a = LG.QAction(text, self)
            a.setShortcut(QtGui.QKeySequence(scut))
            a.triggered.connect(slot)
            m.addAction(a)
        e = self.menuBar().addMenu("&Edit")
        a = LG.QAction("Undo", self)
        a.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        a.triggered.connect(self.undo)
        e.addAction(a)
        v = self.menuBar().addMenu("&View")
        v.addAction(self.warp_action)
        a = LG.QAction("Bring the MRI-in-slab-space window to the front", self)
        a.triggered.connect(lambda: (self.warp_action.setChecked(True), self.warp.bring_up()))
        v.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = LG.QAction("Mouse and keys", self)
        text = "Mouse" + __doc__.split("Mouse", 1)[1].split("Nothing is written")[0]
        a.triggered.connect(lambda: QtWidgets.QMessageBox.information(self, "Mouse and keys", text))
        h.addAction(a)

    def _build_shortcuts(self):
        def scut(keys, slot):
            s = LG.QShortcut(QtGui.QKeySequence(keys), self)
            s.activated.connect(slot)
        scut("P", lambda: self.set_mode("paint"))
        scut("N", lambda: self.set_mode("navigate"))
        scut("Escape", lambda: self.set_mode("navigate"))
        for k in ("Left", "Up", "PgUp"):
            scut(k, lambda: self.active_pane.step(-1))
        for k in ("Right", "Down", "PgDown"):
            scut(k, lambda: self.active_pane.step(1))
        scut("[", lambda: self._brush_step(-1))
        scut("]", lambda: self._brush_step(1))
        scut("F", lambda: self.active_pane.fit_views())
        scut("Ctrl+G", lambda: (self.goto_label.setFocus(), self.goto_label.selectAll()))

    # -- small handlers ---------------------------------------------------------- #
    def _update_swatch(self):
        self.swatch.setStyleSheet(f"background: {LG.label_qcolor(self.label_spin.value()).name()};"
                                  "border: 1px solid #444; border-radius: 3px")

    def _brush_tip(self, *_):
        self.brush_slab.setToolTip(f"≈ {self.brush_slab.value() * float(self.geom.spacing.min()):.2f} mm radius")

    def set_mode(self, mode):
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        for pane in self.panes:
            for v in pane.views.values():
                v.gv.setCursor(Qt.CursorShape.CrossCursor if (mode == "paint" and v.paintable)
                               else Qt.CursorShape.ArrowCursor)
                if mode != "paint":
                    v.brush_item.hide()

    def brush_radius(self, view):
        return self.brush_slab.value() if view.pane is self.slab else self.brush_mri.value()

    def _brush_step(self, d):
        box = self.brush_slab if self.active_pane is self.slab else self.brush_mri
        box.setValue(box.value() + d)

    def _on_alpha(self, v):
        self.alpha = int(v)
        self.lut = self._make_lut(self.alpha)
        self.refresh_overlays()

    def _on_numbers(self, on):
        self.show_numbers = bool(on)
        self.slab.update_annotations()
        self.mri_pane.update_annotations()
        self.warp.pane.render_all()

    def _toggle_warp(self, on):
        if on:
            self.warp.bring_up()
            self.request_warp()
        else:
            self.warp.hide()

    def _thumb_tick(self):
        done, n = self.thumbs.progress
        if not hasattr(self, "_thumb_done") or self._thumb_done != done:
            self._thumb_done = done
            self.slab.render_thumb_views()
            if self.warp.isVisible():
                self.warp.pane.render_overlays(("sagittal", "axial"))
        if done >= n:
            self._thumb_timer.stop()

    def set_hover_status(self, text):
        self.hover_lbl.setText(text)

    def flash(self, text, kind="", ms=8000):
        self.statusBar().setStyleSheet(f"color: {LG.NOTE_COLORS.get(kind)}" if kind else "")
        self.statusBar().showMessage(text, ms)

    def refresh_overlays(self):
        self.slab.views["coronal"].set_labels(self.slab.section_labels_uv(self.slab.idx))
        self.mri_pane.render()
        self.warp.pane.render_all()

    def used_labels(self):
        return set(self.mri.stats) | self.stack.all_labels() | set(self.conflicts)

    def next_free_label(self):
        used = self.used_labels()
        self.label_spin.setValue(min(MAX_LABEL, (max(used) + 1) if used else 1))
        self.flash(f"Current label set to {self.label_spin.value()} (unused in this slab, other slabs and the MRI)")

    def on_slab_cursor(self):
        if self.warp.isVisible():
            self.warp.pane.update_annotations()
            self._warp_timer.start()

    # -- painting --------------------------------------------------------------- #
    def begin_stroke(self, view, erase=False, desc=None, kind="paint"):
        if self._stroke is None:
            what = "erase" if erase else f"paint {self.label_spin.value()}"
            self._stroke = LG.Stroke(view, desc or f"{what} on {view.pane.__class__.__name__}", kind)
            self._stroke.erase = erase

    def paint_segment(self, view, p0, p1):
        if self._stroke is None or not view._has_image:
            return
        erase = getattr(self._stroke, "erase", False)
        value = 0 if erase else int(self.label_spin.value())
        protect = self.protect_cb.isChecked() and value != 0
        rad = self.brush_radius(view)
        nr, nc = view.raster_shape
        target = view.pane.target(view, create=True)
        key = view.pane.target_key(view)
        (c0, r0), (c1, r1) = p0, p1
        dist = math.hypot(c1 - c0, r1 - r0)
        n = max(1, int(math.ceil(dist / max(0.5, rad * 0.5))))
        box = [nr, 0, nc, 0]
        blocked = 0
        for t in (np.linspace(0.0, 1.0, n + 1) if dist > 0 else (0.0,)):
            cc = int(math.floor(c0 + (c1 - c0) * t))
            rr = int(math.floor(r0 + (r1 - r0) * t))
            if not (0 <= rr < nr and 0 <= cc < nc):
                continue
            ra, rb, ca, cb = max(0, rr - rad), min(nr, rr + rad + 1), max(0, cc - rad), min(nc, cc + rad + 1)
            yy, xx = np.ogrid[ra:rb, ca:cb]
            mask = (yy - rr) ** 2 + (xx - cc) ** 2 <= rad * rad
            rows, cols = np.nonzero(mask)
            u, v = view.rc_to_uv(rows + ra, cols + ca)
            tflat = view.pane.to_target_flat(view, u, v)
            cur = np.take(target, tflat)
            sel = cur != value
            if protect:
                blocked += int(np.count_nonzero(sel & (cur != 0)))
                sel &= cur == 0
            if not sel.any():
                continue
            np.put(target, tflat[sel], value)
            self._stroke.add(key, tflat[sel], cur[sel], value)
            box = [min(box[0], ra), max(box[1], rb), min(box[2], ca), max(box[3], cb)]
        if box[1] > box[0]:
            view.refresh_label_block(*box, view.pane.labels_uv(view))
        if blocked:
            self.flash("Other labels under the brush were left untouched "
                       "(untick \"Don't overwrite other labels\" to paint over them)", "warn", 4000)

    def end_stroke(self):
        rec, self._stroke = self._stroke, None
        if rec is None or not rec.entries:
            return
        self.undo_stack.append(rec)
        del self.undo_stack[:-300]
        self._apply_changes(rec.by_key(), own=rec.pane if rec.kind == "paint" else None)
        self.update_dirty()

    def _target_for_key(self, key):
        if key[0] == sc.MRI_KEY:
            return self.mri.labels
        return self.stack.get_labels(key[1], create=True)

    def _apply_changes(self, by_key, own=None):
        """Refresh statistics and views after edits; ``own`` is a slab view that
        already shows the edit (painted incrementally)."""
        for key, (flats, labs) in by_key.items():
            flat = np.concatenate(flats)
            if key[0] == sc.MRI_KEY:
                self.mri.refresh_labels(labs, core.bbox_from_flat(flat, self.mri.shape))
                self.mri.version += 1
            else:
                self.stack.refresh(key[1])
        if ("slab", self.slab.idx) in by_key and own is not self.slab.views["coronal"]:
            self.slab.views["coronal"].set_labels(self.slab.section_labels_uv(self.slab.idx))
        if any(k[0] == "slab" for k in by_key):
            self.slab.update_annotations()
        if (sc.MRI_KEY,) in by_key:
            self.mri_pane.render()
        self._table_timer.start()
        self.schedule_fit()

    def undo(self):
        if self._stroke is not None:
            return
        if not self.undo_stack:
            self.flash("Nothing to undo")
            return
        rec = self.undo_stack.pop()
        by_key: dict = {}
        for key, flat, old, _value in reversed(rec.entries):
            tgt = self._target_for_key(key)
            before = np.take(tgt, flat)
            np.put(tgt, flat, old)
            flats, labs = by_key.setdefault(key, ([], set()))
            flats.append(flat)
            labs.update(int(x) for x in np.unique(before))
            labs.update(int(x) for x in np.unique(old))
        self._apply_changes(by_key)
        self.update_dirty()
        self.flash(f"Undone: {rec.desc}")

    def place_label(self, view, r, c):
        lab = int(self.label_spin.value())
        u, v = view.rc_to_uv(np.array([r]), np.array([c]))
        target = view.pane.target(view, create=True)
        occupant = int(np.take(target, view.pane.to_target_flat(view, u, v))[0])
        if occupant not in (0, lab):
            self.flash(f"That spot is already label {occupant}; erase it first or click elsewhere", "warn")
            return
        self.begin_stroke(view, desc=f"move label {lab}", kind="place")
        moved = []
        if view.pane is self.mri_pane:
            flat = self.mri.remove_label(lab)
            if flat.size:
                self._stroke.add((sc.MRI_KEY,), flat, np.full(flat.size, lab), 0)
                moved.append("MRI")
        else:
            for idx, flat in self.stack.remove_label(lab):
                self._stroke.add(("slab", idx), flat, np.full(flat.size, lab), 0)
                moved.append(self.stack.describe(idx, short=True))
        self.paint_segment(view, (c + 0.5, r + 0.5), (c + 0.5, r + 0.5))
        self.end_stroke()
        if view.pane is self.slab:
            self.slab.views["coronal"].set_labels(self.slab.section_labels_uv(self.slab.idx))
        self.flash(f"Label {lab} placed" + (f" (removed from {', '.join(moved)})" if moved else ""))

    def pick_label(self, view, r, c):
        if view.lab_raster is None:
            return
        lab = core.label_near(view.lab_raster, r, c, 6)
        if lab:
            self.label_spin.setValue(int(lab))
            self.flash(f"Current label: {lab}")

    # -- transform ------------------------------------------------------------- #
    def _on_settings(self, *_):
        s = self.settings
        s.rigid_min, s.affine_min, s.nonlinear_min = self.rigid_min.value(), self.affine_min.value(), self.nl_min.value()
        s.cp_spacing, s.levels, s.handedness = self.cp_spacing.value(), self.levels.value(), self.hand.currentText()
        self.schedule_fit()

    def schedule_fit(self, now=False):
        self.fit_status.setText("fitting…")
        if now:
            self.run_fit()
        else:
            self._fit_timer.start()

    def run_fit(self):
        self.pairs = sc.landmark_pairs(self.stack, self.geom, self.mri, self.conflicts)
        pairs, geom, mri = self.pairs, self.geom, self.mri
        settings = sc.TransformSettings(**vars(self.settings))
        self.async_.submit("fit", lambda: sc.fit_transform(pairs, geom, mri, settings, self.workdir), self._on_fit)

    def _on_fit(self, res, exc):
        if exc is not None:
            self.fit = None
            self.fit_status.setText(f"<span style='color:#d9534f'>Fit failed: {exc}</span>")
            return
        self.fit = res
        if res.ok and self.settings.handedness == "auto" and not res.handedness_note:
            self.settings.last_mirror = res.mirror
        excl = ""
        if self.pairs and self.pairs.excluded:
            excl = "<br>Not used: " + ", ".join(f"{k} ({v})" for k, v in sorted(self.pairs.excluded.items()))
        txt = res.summary()
        if res.ok:
            txt += f"<br>fit {res.seconds:.2f} s; transforms: " + " ← ".join(Path(f).name for f in res.files)
        self.fit_status.setText(txt + excl)
        self.warp.status.setText(txt)
        self._table_timer.start()
        self._lab_crop_cache = None
        self.request_warp()

    def _labels_crop(self, bbox):
        """Copy of the MRI labels inside bbox (taken on the GUI thread)."""
        key = (bbox, self.mri.version)
        if self._lab_crop_cache is not None and self._lab_crop_cache[0] == key:
            return self._lab_crop_cache[1]
        lo, hi = bbox
        arr = self.mri.labels[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].astype(np.float32)
        origin = tuple(self.mri.voxel_to_lps(lo)[0])
        _o, spacing, direction = self.mri.ants_geometry()
        out = (arr, origin, spacing, direction)
        self._lab_crop_cache = (key, out)
        return out

    def request_warp(self):
        if not self.warp.isVisible():
            return
        fit = self.fit
        if fit is None or not fit.ok:
            for v in self.warp.pane.views.values():
                v.clear()
                v.note(fit.message if fit is not None else "no transform yet", "muted")
            return
        for v in self.warp.pane.views.values():
            v.note("")
        s, g = self.slab, self.geom
        refs = {"coronal": g.coronal_ref(g.y[s.idx], fit.mirror), "sagittal": g.sagittal_ref(s.x, fit.mirror),
                "axial": g.axial_ref(s.z, fit.mirror)}
        bbox = fit.mri_bbox
        want_labels = self.warp.labels_cb.isChecked()
        lab_crop = self._labels_crop(bbox) if (want_labels and bbox is not None) else None
        mri = self.mri

        def work():
            ants = sc._ants()
            cache = getattr(fit, "_img_crop", None)
            if cache is None:
                cache = sc.crop(mri.ants_image(), bbox)
                fit._img_crop = cache
            lab_img = None
            if want_labels:
                if lab_crop is not None:
                    arr, o, sp, d = lab_crop
                    lab_img = ants.from_numpy(arr, origin=o, spacing=sp, direction=d)
                else:
                    lab_img = mri.ants_labels()
            out = {}
            for name, ref in refs.items():
                img = sc.resample_plane(fit, cache, ref, "linear")
                lab = (np.rint(sc.resample_plane(fit, lab_img, ref, "nearestNeighbor")).astype(np.int64)
                       if lab_img is not None else None)
                out[name] = (img, lab)
            return out

        def done(out, exc):
            if exc is not None:
                self.warp.status.setText(f"<span style='color:#d9534f'>Resampling failed: {exc}</span>")
                return
            for name, (img, lab) in out.items():
                self.warp.pane.set_plane(name, img, lab)

        self.async_.submit("warp", work, done)

    # -- go to label --------------------------------------------------------------- #
    def _on_goto_label(self):
        text = self.goto_label.text().strip()
        if not text:
            return
        try:
            lab = int(float(text))
        except ValueError:
            self.flash(f"'{text}' is not a label number", "error")
            return
        self.go_to_label(lab)

    def _clear_goto(self):
        for pane in self.panes + [self.warp.pane]:
            for v in pane.views.values():
                v.set_goto()

    def go_to_label(self, lab):
        st = self.mri.stats.get(lab)
        if st is None:
            self.flash(f"Label {lab} does not exist in the MRI.", "error")
            self.goto_label.setStyleSheet("background: #5a1f1f")
            QtCore.QTimer.singleShot(1200, lambda: self.goto_label.setStyleSheet(""))
            return
        self._clear_goto()
        self.label_spin.setValue(lab)
        com = np.rint(st.com).astype(int)
        self.mri_pane.set_cursor(com)
        for name, view in self.mri_pane.views.items():
            a, au, av, _ = self.mri.plane_axes(name)
            view.set_goto(st.com[au], st.com[av])
        self.mri_pane.views["coronal"].note(f"label {lab}: centre of mass voxel ({com[0]}, {com[1]}, {com[2]})", "ok")
        places = self.stack.locations().get(lab, [])
        cv = self.slab.views["coronal"]
        if places:
            idx, s = places[0]
            p = self.geom.com_to_frame(idx, s.com)
            self._show_slab_point(p, idx, predicted=False)
            if len(places) > 1:
                cv.note(f"label {lab} is in {len(places)} sections: "
                        + ", ".join(self.stack.describe(i, short=True) for i, _ in places), "error")
            else:
                cv.note(f"label {lab}", "ok")
        elif lab in self.conflicts:
            cv.note(f"label {lab} belongs to {', '.join(self.conflicts[lab])}", "muted")
        elif self.fit is not None and self.fit.ok:
            phys = self.fit.to_slab(self.mri.voxel_to_lps(st.com))[0]
            p = sc.SlabGeometry.to_frame(phys, self.fit.mirror)
            lo, hi = self.geom.bounds()
            if lo[1] - 1.0 <= p[1] <= hi[1] + 1.0:
                idx = self.geom.nearest_section(p[1])
                self._show_slab_point(p, idx, predicted=True)
                cv.note(f"label {lab} not on this slab · predicted position from the {self.fit.kind} transform "
                        f"(dashed)", "warn")
            else:
                cv.note(f"label {lab} not on this slab · predicted y = {p[1]:.1f} mm is outside the slab "
                        f"({lo[1]:.1f}–{hi[1]:.1f} mm)", "muted")
        else:
            cv.note(f"label {lab} not on this slab", "muted")
        self.flash(f"Label {lab}: MRI voxel ({com[0]}, {com[1]}, {com[2]})"
                   + (f"; slab {self.stack.describe(places[0][0], short=True)}" if places else ""))
        self._select_table_row(lab)

    def _show_slab_point(self, p, idx, predicted):
        g, s = self.geom, self.slab
        s.set_section(idx, force=True)
        s.set_xz(x=p[0], z=p[2])
        u, v = g.frame_to_section(idx, p[0], p[2])
        s.views["coronal"].set_goto(u, v, predicted)
        row = g.y_to_row(p[1])
        s.views["sagittal"].set_goto(row, p[2] / g.ts - 0.5, predicted)
        s.views["axial"].set_goto(p[0] / g.ts - 0.5, row, predicted)
        w = self.warp.pane.views
        w["coronal"].set_goto(p[0] / g.cs - 0.5, p[2] / g.cs - 0.5, predicted)
        w["sagittal"].set_goto(row, p[2] / g.ts - 0.5, predicted)
        w["axial"].set_goto(p[0] / g.ts - 0.5, row, predicted)

    # -- tables ------------------------------------------------------------------- #
    def _icon(self, lab):
        ic = self._icons.get(lab)
        if ic is None:
            pm = QtGui.QPixmap(10, 10)
            pm.fill(LG.label_qcolor(lab))
            ic = self._icons[lab] = QtGui.QIcon(pm)
        return ic

    def refresh_table(self):
        locs = self.stack.locations()
        labels = sorted(set(self.mri.stats) | set(locs) | set(self.conflicts))
        res = self.fit.residuals if (self.fit is not None and self.fit.ok) else {}
        excluded = self.pairs.excluded if self.pairs else {}
        headers = ["Label", "MRI", "This slab", "Other slab", "Fit error"]
        t = self.table
        t.setUpdatesEnabled(False)
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(labels))
        err_bg, err_fg, muted = QtGui.QColor("#f2c4c4"), QtGui.QColor("#5c0000"), QtGui.QColor("#8a8a8a")
        n_err = 0
        for row, lab in enumerate(labels):
            it = QtWidgets.QTableWidgetItem()
            it.setData(Qt.ItemDataRole.DisplayRole, lab)
            it.setData(Qt.ItemDataRole.UserRole, lab)
            it.setIcon(self._icon(lab))
            t.setItem(row, 0, it)
            cells = []
            ms = self.mri.stats.get(lab)
            if ms is None:
                cells.append(("missing", True))
            elif ms.n_components > 1:
                cells.append((f"{ms.n_components} blobs", True))
            else:
                cells.append((f"{'ijk'[self.mri.ax_cor]}={int(round(ms.com[self.mri.ax_cor]))}", False))
            places = locs.get(lab, [])
            if not places:
                cells.append(("–", None))
            elif len(places) > 1:
                cells.append(("×%d " % len(places) + ",".join(self.stack.describe(i, True) for i, _ in places), True))
            else:
                bad = places[0][1].n_components > 1 or lab in self.conflicts or ms is None
                cells.append((self.stack.describe(places[0][0], short=True), bad))
            cells.append((", ".join(self.conflicts.get(lab, [])) or "–", bool(places and lab in self.conflicts)))
            if lab in res:
                cells.append((f"{res[lab]:.2f} mm", res[lab] > 2.0))
            else:
                cells.append((excluded.get(lab, "–") if places else "–", None))
            for col, (text, bad) in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, lab)
                if bad:
                    item.setBackground(err_bg)
                    item.setForeground(err_fg)
                    n_err += 1
                elif bad is None:
                    item.setForeground(muted)
                t.setItem(row, col, item)
        t.resizeColumnsToContents()
        t.setUpdatesEnabled(True)
        self.tabs.setTabText(0, f"Labels ({len(labels)})" + (f" · {n_err} problems" if n_err else ""))

    def _select_table_row(self, lab):
        for row in range(self.table.rowCount()):
            it = self.table.item(row, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) == lab:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                self.table.scrollToItem(it)
                return

    def _on_table_click(self, row, _col):
        it = self.table.item(row, 0)
        if it is not None:
            self.go_to_label(int(it.data(Qt.ItemDataRole.UserRole)))

    def validate_now(self):
        self.populate_issues(sc.validate_slab(self.stack, self.mri, self.conflicts, self.fit, self.geom))

    def populate_issues(self, issues, switch=True):
        self.issues = list(self._load_warnings) + list(issues)
        self.issues_list.clear()
        order = {"error": 0, "warning": 1, "info": 2}
        colors = {"error": "#e06c6c", "warning": "#e0a030", "info": "#9a9a9a"}
        for i, iss in sorted(enumerate(self.issues), key=lambda x: (order[x[1].severity], x[0])):
            item = QtWidgets.QListWidgetItem(f"{iss.severity.upper()}: {iss.message}")
            item.setForeground(QtGui.QColor(colors[iss.severity]))
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip(iss.message)
            self.issues_list.addItem(item)
        n_err = sum(i.severity == "error" for i in issues)
        self.tabs.setTabText(1, f"Issues ({n_err} errors)" if n_err else "Issues (none)")
        if switch:
            self.tabs.setCurrentIndex(1)
            self.flash(f"Validation: {n_err} error(s)" if n_err else
                       "Validation passed: labels unique and every slab label has an MRI counterpart",
                       "error" if n_err else "ok")

    def _on_issue(self, item):
        iss = self.issues[int(item.data(Qt.ItemDataRole.UserRole))]
        if iss.label is not None and iss.label in self.mri.stats:
            self.go_to_label(iss.label)
        elif iss.section_idx is not None:
            self.slab.set_section(iss.section_idx, force=True)
        if iss.section_idx is not None and iss.label in self.stack.all_labels():
            idx = iss.section_idx
            self.slab.set_section(idx, force=True)
            self.slab.views["coronal"].note(iss.message, "error" if iss.severity == "error" else "warn")

    # -- saving -------------------------------------------------------------------- #
    def update_dirty(self):
        top = self.undo_stack[-1].id if self.undo_stack else None
        self.dirty = top != self._saved_top
        self.setWindowModified(self.dirty)

    def names(self):
        return sc.slab_output_names(self.out_dir, self.key, self.mri.label_path, self.version_tag, self.label_suffix)

    def gui_state(self):
        return dict(slab=self.slab.state(), mri=self.mri_pane.state(), current_label=int(self.label_spin.value()),
                    brush_slab=self.brush_slab.value(), brush_mri=self.brush_mri.value(),
                    overlay_alpha=self.alpha, show_numbers=self.show_numbers, protect=self.protect_cb.isChecked(),
                    transform=vars(self.settings), warp_visible=self.warp.isVisible())

    def _ensure_writable(self, interactive):
        while not self._out_owned:
            contents = core.folder_contents(self.out_dir)
            csv = self.names().csv()
            if (not contents and not csv.exists()) or self.overwrite:
                return True
            if not interactive:
                return False
            prev = core.previous_session(self.out_dir)
            lines = []
            if contents:
                what = (f"a session saved on {prev.get('saved')}" if prev else
                        f"{len(contents)} item(s), e.g. {', '.join(contents[:3])}")
                lines.append(f"{self.out_dir}\nalready contains {what}.")
            if csv.exists():
                lines.append(f"{csv}\nalready exists.")
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "Output already exists",
                                        "\n\n".join(lines) + "\n\nSaving would overwrite these files. Choose a "
                                        "new folder for the new labels, or overwrite.", parent=self)
            choose = box.addButton("Choose another folder…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            over = box.addButton("Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(choose)
            _exec(box)
            clicked = box.clickedButton()
            if clicked is over:
                return True
            if clicked is not choose:
                return False
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the new labels",
                                                                str(self.out_dir.parent))
            if not folder:
                return False
            self.out_dir = Path(folder).resolve()
            self._update_title()
        return True

    def save(self, quiet=False):
        if self._stroke is not None:
            return False
        if not self.dirty:
            if not quiet:
                self.flash("No changes to the annotations — nothing written.")
            return True
        if not self._ensure_writable(interactive=not quiet):
            if quiet:
                self.flash(f"Autosave skipped: {self.out_dir} already contains files. Press Save to choose another "
                           "folder or overwrite.", "warn", 15000)
            return False
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            names = self.names()
            fit = self.fit
            rep = core.save_session(
                names, self.ds["df"], self.ds["columns"], [self.stack], self.mri, self.label_format,
                input_csv_dir=Path(self.ds["csv_path"]).parent, session_info=self.session_info,
                gui_state=self.gui_state(),
                validator=lambda: sc.validate_slab(self.stack, self.mri, self.conflicts, fit, self.geom))
            pairs = sc.landmark_pairs(self.stack, self.geom, self.mri, self.conflicts)
            tmp = core.atomic_target(names.aux("landmark_fit.csv"))
            sc.fit_table(pairs, fit, self.stack, self.mri).to_csv(tmp, index=False)
            core._replace(tmp, names.aux("landmark_fit.csv"))
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            log.exception("Save failed")
            QtWidgets.QMessageBox.critical(self, "Save failed", f"{exc}\n\nNothing was lost from memory; fix the "
                                           "problem and save again.")
            return False
        QtWidgets.QApplication.restoreOverrideCursor()
        self._out_owned = True
        self._saved_top = self.undo_stack[-1].id if self.undo_stack else None
        self.update_dirty()
        self.populate_issues(rep.issues, switch=bool(rep.n_errors) and not quiet)
        msg = rep.summary().replace("hemisphere 3D labels for", "slab MRI labels for")
        self.flash(("Autosaved. " if quiet else "") + msg, "warn" if rep.n_errors else "ok", 12000)
        return True

    def _on_autosave(self):
        if self.dirty and self._stroke is None:
            self.save(quiet=True)

    def closeEvent(self, ev):
        if self.dirty:
            ans = QtWidgets.QMessageBox.question(
                self, "Unsaved changes", "Save changes before closing?",
                QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel)
            if ans == QtWidgets.QMessageBox.StandardButton.Cancel:
                ev.ignore()
                return
            if ans == QtWidgets.QMessageBox.StandardButton.Save and not self.save():
                ev.ignore()
                return
        if self._out_owned:   # remember view positions in a folder this session already wrote to
            try:
                core.write_text_atomic(self.names().aux(core.STATE_JSON), json.dumps(self.gui_state(), indent=2))
            except OSError:
                pass
        for t in (self._fit_timer, self._warp_timer, self._table_timer, self._thumb_timer, self._autosave):
            t.stop()
        self.thumbs.stop()
        self.async_.closed = True
        self.stack._pool.shutdown(wait=False, cancel_futures=True)
        self.async_.pool.shutdown(wait=True, cancel_futures=True)
        self.warp.close()
        self.release_ants()
        ev.accept()

    def release_ants(self):
        """Drop ANTs objects before interpreter exit (avoids nanobind leak warnings)."""
        import gc
        self.fit = None
        self.async_.jobs.clear()
        self.mri._ants_img = self.mri._ants_lab = None
        gc.collect()


# --------------------------------------------------------------------------- #
# Start-up
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="section CSV: sub, hemisphere, chunk (slab), section, raw (2-D NIfTI), labels")
    ap.add_argument("--mri", help="donor MRI (NIfTI)")
    ap.add_argument("--mri-labels", help="3-D landmark labels on the MRI (NIfTI); created empty if missing")
    ap.add_argument("--sub", help="subject of the slab (default: ask if the CSV has several slabs)")
    ap.add_argument("--hemisphere", help="hemisphere of the slab")
    ap.add_argument("--chunk", help="slab (chunk) number")
    ap.add_argument("--slice-thickness", type=float, default=sc.DEFAULT_THICKNESS,
                    help="mm between consecutive sections (default: %(default)s)")
    ap.add_argument("--order", choices=sc.ORDER_MODES, default=None,
                    help="stacking order: 'stack' = like stack_view.py, by (repeat, ligand, name) with "
                         "consecutive sections one slice thickness apart; 'section' = y from the section "
                         "numbers; 'auto' (default) = section if every section has its own number, else stack")
    ap.add_argument("--center", "--centre", dest="center", choices=sc.CENTER_MODES, default=None,
                    help="in-plane placement: 'mask' = like stack_view.py, align the centre of mass of each "
                         "section's mask (CSV mask column; intensity centroid if there is none); 'extent' = "
                         "centre the image extents (BrainBuilder); 'auto' (default) = mask if the CSV has a "
                         "mask column")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help=f"in-plane pixel size in mm for PNG/TIFF sections, which have no header "
                         f"(default {sc.DEFAULT_PIXEL_SIZE}: 20 um originals downsampled 10x). NIfTI sections "
                         f"use their header.")
    ap.add_argument("--label-format", default="match", choices=core.LABEL_FORMATS,
                    help="file format of saved section labels; 'match' = same as each section "
                         "(PNG -> 16-bit PNG, NIfTI -> .nii.gz)")
    ap.add_argument("--out-dir", required=True,
                    help="new folder for edited labels and the new CSV; created on the first save with changes")
    ap.add_argument("--overwrite", action="store_true", help="allow saving into an --out-dir that already has files")
    ap.add_argument("--version-tag", help="suffix for output names (default: --out-dir name if it looks like v1.4)")
    ap.add_argument("--label-suffix", default=core.LABEL_SUFFIX, help="2-D label files: <image><suffix>.nii.gz")
    ap.add_argument("--flip-lr", action="store_true",
                    help="show the slab's sections mirrored left-right (display only; files keep their orientation)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="load the labels saved in --out-dir")
    g.add_argument("--fresh", action="store_true", help="load --csv/--mri-labels even if --out-dir has a session")
    t = ap.add_argument_group("transform")
    t.add_argument("--rigid-min", type=int, default=3, help="pairs needed for a rigid transform (default 3)")
    t.add_argument("--affine-min", type=int, default=12, help="pairs needed for affine (default 12)")
    t.add_argument("--nonlinear-min", type=int, default=20, help="pairs needed for affine + B-spline (default 20)")
    t.add_argument("--cp-spacing", type=float, default=10.0, help="B-spline control-point spacing, mm (default 10)")
    t.add_argument("--levels", type=int, default=4, help="max B-spline fitting levels (default 4)")
    t.add_argument("--handedness", choices=("auto", "normal", "mirrored"), default="auto")
    ap.add_argument("--autosave-min", type=float, default=5.0, help="autosave interval in minutes (0 = off)")
    ap.add_argument("--cache-dir", default=None, help="cache for the downsampled slab (default ~/.cache/slab_landmark_gui)")
    ap.add_argument("--no-cache", action="store_true", help="don't cache the downsampled slab")
    for role in ("sub", "hemi", "chunk", "section", "raw", "labels", "mask", "base"):
        ap.add_argument(f"--col-{role}", help=f"CSV column holding the {role} (default: auto-detect)")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def resolve_session(args, ask) -> dict:
    out_dir = Path(args.out_dir).expanduser().resolve()
    overrides = {"sub": args.col_sub, "hemisphere": args.col_hemi, "chunk": args.col_chunk,
                 "section": args.col_section, "raw": args.col_raw, "labels": args.col_labels,
                 "mask": args.col_mask, "base": args.col_base}
    overrides = {k: v for k, v in overrides.items() if v}
    cfg = dict(out_dir=out_dir, csv=args.csv, mri=args.mri, mri_labels=args.mri_labels, overrides=overrides,
               thickness=args.slice_thickness, slab=None, resumed=False, flip_lr=args.flip_lr,
               order=args.order or "auto", center=args.center or "auto",
               pixel_size=args.pixel_size if args.pixel_size is not None else sc.DEFAULT_PIXEL_SIZE)
    prev = core.previous_session(out_dir)
    if prev is None:
        if args.resume:
            raise SystemExit(f"--resume: no saved session in {out_dir}")
        return cfg
    passed_prev = args.csv and Path(args.csv).expanduser().resolve() == Path(prev["csv_path"])
    choice = "resume" if (args.resume or passed_prev) else ("fresh" if args.fresh else None)
    if choice is None:
        choice = ask(f"{out_dir} already contains a saved session (last saved {prev.get('saved', '?')}).\n\n"
                     "Resume it (load its labels)? 'Start from inputs' loads the CSV and labels you passed instead.")
    if choice == "cancel":
        raise SystemExit(0)
    if choice == "resume":
        cfg.update(csv=prev["csv_path"], mri_labels=prev["template_labels_path"], resumed=True)
        cfg["mri"] = args.mri or prev.get("template")
        if prev.get("slab"):
            cfg["slab"] = sc.SlabKey(**prev["slab"])
        cfg["thickness"] = prev.get("slice_thickness", args.slice_thickness)
        if args.pixel_size is None and prev.get("pixel_size"):
            cfg["pixel_size"] = prev["pixel_size"]
        cfg["flip_lr"] = args.flip_lr or prev.get("flip_lr", False)
        cfg["order"] = args.order or prev.get("order", "auto")
        cfg["center"] = args.center or prev.get("center", "auto")
        if not overrides and prev.get("columns"):
            cols = prev["columns"]
            cfg["overrides"] = {k: cols[k] for k in ("sub", "hemisphere", "chunk", "section", "raw", "labels",
                                                     "mask", "base") if k in cols}
    return cfg


def choose_slab(args, cfg, slabs, pick) -> sc.SlabKey:
    if cfg.get("slab") is not None:
        return cfg["slab"]
    cands = [k for k in slabs
             if (args.sub is None or k.sub == core.clean_sub(args.sub))
             and (args.hemisphere is None or k.hemisphere == args.hemisphere)
             and (args.chunk is None or k.chunk == str(args.chunk))]
    if not cands:
        raise ValueError(f"No slab matches --sub {args.sub} --hemisphere {args.hemisphere} --chunk {args.chunk}; "
                         f"available: {', '.join(k.id for k in slabs)}")
    if len(cands) == 1:
        return cands[0]
    k = pick([c.id for c in cands])
    if k is None:
        raise SystemExit(0)
    return cands[k]


def load_slab(cfg, key: sc.SlabKey, cache_dir, progress=None) -> dict:
    df, cols, sections, warnings = core.read_sections(cfg["csv"], cfg["overrides"], row_filter=key.matches,
                                                      section_required=False)
    if "chunk" not in cols:
        raise ValueError("The CSV needs a slab column (chunk/slab)")
    stack = sc.SlabStack(sections, order=cfg.get("order", "auto"), label_dtype=SLAB_LABEL_DTYPE)
    stack.id = key.id
    notes = [f"{stack.id}: {stack.order_note}"]
    center = cfg.get("center", "auto")
    if center == "mask" or (center == "auto" and "mask" in cols):
        bad = sc.compute_centres(stack, cache_dir, progress=progress)
        if bad:
            warnings.append(f"{len(bad)} section(s) whose mask could not be used: " + "; ".join(bad[:3])
                            + (" …" if len(bad) > 3 else ""))
    notes.append(f"{stack.id}: {sc.centring_summary(stack)}")
    stack.load_labels((lambda i, n: progress("Loading slab labels", i, n)) if progress else None)
    warnings += stack.load_warnings
    if progress:
        progress("Loading the MRI", 1, 1)
    mri = sc.DonorMRI(cfg["mri"], cfg["mri_labels"])
    warnings += mri.warnings
    geom = sc.SlabGeometry(stack, cfg["thickness"], cfg.get("pixel_size", sc.DEFAULT_PIXEL_SIZE))
    if geom.raster.any():
        warnings.append(f"{int(geom.raster.sum())} PNG/TIFF section(s): pixel size {geom.pixel_size} mm "
                        f"(--pixel-size); shown as stored, row 0 at the top")
    thumbs = sc.ThumbStack(stack, geom, cache_dir)
    conflicts = sc.other_slab_labels(df, cols, Path(cfg["csv"]).expanduser().resolve().parent, key)
    return dict(key=key, df=df, columns=cols, stack=stack, mri=mri, geom=geom, thumbs=thumbs, conflicts=conflicts,
                warnings=warnings, notes=notes, csv_path=str(Path(cfg["csv"]).expanduser().resolve()))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Slab landmarks")

    def ask(text):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, "Saved session found", text)
        resume = box.addButton("Resume", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        fresh = box.addButton("Start from inputs", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(resume)
        _exec(box)
        c = box.clickedButton()
        return "resume" if c is resume else ("fresh" if c is fresh else "cancel")

    def pick(items):
        item, ok = QtWidgets.QInputDialog.getItem(None, "Choose a slab", "The CSV has several slabs:", items, 0, False)
        return items.index(item) if ok else None

    try:
        cfg = resolve_session(args, ask)
        if not cfg["csv"] or not cfg["mri"]:
            raise ValueError("--csv and --mri are required (or --resume with a saved session)")
        key = choose_slab(args, cfg, sc.list_slabs(cfg["csv"], cfg["overrides"]), pick)
        prog = QtWidgets.QProgressDialog("Loading…", None, 0, 1)
        prog.setWindowTitle("Slab landmarks")
        prog.setMinimumDuration(300)

        def progress(msg, i, n):
            prog.setMaximum(max(n, 1))
            prog.setValue(i)
            prog.setLabelText(msg)
            app.processEvents()

        cache = None if args.no_cache else (Path(args.cache_dir) if args.cache_dir else sc.default_cache_dir())
        ds = load_slab(cfg, key, cache, progress)
        prog.close()
    except SystemExit:
        raise
    except Exception as exc:
        log.debug(traceback.format_exc())
        QtWidgets.QMessageBox.critical(None, "Cannot start", str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1

    state = core.load_state(cfg["out_dir"]) if cfg["resumed"] else {}
    ts = state.get("transform", {})
    settings = sc.TransformSettings(
        rigid_min=ts.get("rigid_min", args.rigid_min), affine_min=ts.get("affine_min", args.affine_min),
        nonlinear_min=ts.get("nonlinear_min", args.nonlinear_min), cp_spacing=ts.get("cp_spacing", args.cp_spacing),
        levels=ts.get("levels", args.levels), handedness=ts.get("handedness", args.handedness),
        last_mirror=ts.get("last_mirror", False))
    session_info = dict(csv_input=str(Path(args.csv).resolve()) if args.csv else cfg["csv"],
                        template_labels_input=cfg["mri_labels"], resumed=cfg["resumed"],
                        slab=dict(sub=key.sub, hemisphere=key.hemisphere, chunk=key.chunk),
                        slice_thickness=cfg["thickness"], flip_lr=bool(cfg["flip_lr"]),
                        pixel_size=cfg.get("pixel_size"), order=cfg.get("order", "auto"),
                        center=cfg.get("center", "auto"))
    ds["thumbs"].start(min(ds["stack"].stats, default=0))
    win = MainWindow(ds, cfg["out_dir"], settings, session_info, state, args.autosave_min,
                     overwrite=args.overwrite, version_tag=args.version_tag, label_suffix=args.label_suffix,
                     flip_lr=bool(cfg["flip_lr"]), label_format=args.label_format)
    win.show()
    return _exec(app)


if __name__ == "__main__":
    sys.exit(main())
