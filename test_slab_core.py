"""Tests for the slab core against a synthetic slab with a known mapping
(run: python -m pytest test_slab_core.py -q)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import landmark_core as core
import make_synthetic_slab as ms
import slab_core as sc

HERE = Path(__file__).resolve().parent
KEY = sc.SlabKey("MR9", "L", "6")


def _make(tmp_path_factory, name, *extra):
    out = tmp_path_factory.mktemp(name)
    subprocess.run([sys.executable, str(HERE / "make_synthetic_slab.py"), str(out), *extra], check=True,
                   capture_output=True)
    return out


@pytest.fixture(scope="session")
def slab(tmp_path_factory):
    return _make(tmp_path_factory, "slab", "--preplaced", "22")


@pytest.fixture(scope="session")
def mirrored(tmp_path_factory):
    return _make(tmp_path_factory, "mirrored", "--mirror", "--conflict")


def load(d):
    df, cols, sections, _ = core.read_sections(d / "sections.csv", row_filter=KEY.matches)
    st = core.SpecimenStack(sections, label_dtype=np.uint16)
    st.id = KEY.id
    st.load_labels()
    mri = sc.DonorMRI(str(d / "mri.nii.gz"), str(d / "labels_in" / "mri_landmarks_v1.0.nii.gz"))
    geom = sc.SlabGeometry(st)
    return df, cols, st, mri, geom, dict(np.load(d / "truth.npz"))


def test_geometry_uses_header_pixel_size_and_thickness(slab):
    _df, _c, st, _m, g, _t = load(slab)
    assert np.allclose(g.spacing, 0.15)
    assert np.allclose(g.y, g.numbers * 0.02)
    assert np.allclose(g.extent, [84.0, 66.0], atol=1e-4)
    small = [i for i in range(g.n) if tuple(g.shapes[i]) != (560, 440)]
    assert small and np.allclose(g.offset[small[0]], [40 * 0.15 / 2, 30 * 0.15 / 2])
    assert g.row_y[0] == g.y.min() and abs(g.row_y[-1] - g.y.max()) < g.ty


def test_mri_ants_geometry_matches_ants_reader(slab):
    ants = sc._ants()
    _df, _c, _s, mri, _g, _t = load(slab)
    img = ants.image_read(str(slab / "mri.nii.gz"))
    o, s, d = mri.ants_geometry()
    assert np.allclose(img.origin, o) and np.allclose(img.spacing, s) and np.allclose(img.direction, d)
    ijk = [20, 30, 40]
    assert np.allclose(ants.transform_index_to_physical_point(img, ijk), mri.voxel_to_lps(ijk)[0], atol=1e-4)


def test_pairs_and_other_slabs(slab):
    df, cols, st, mri, g, truth = load(slab)
    conflicts = sc.other_slab_labels(df, cols, slab, KEY)
    assert set(conflicts) == {30, 31}
    pairs = sc.landmark_pairs(st, g, mri, conflicts)
    assert pairs.labels == list(range(1, 23))
    err = np.linalg.norm(ms.true_map(pairs.slab_frame, truth) - pairs.mri_lps, axis=1)
    assert err.max() < 0.5                                  # only voxel rounding of the MRI landmarks


@pytest.mark.parametrize("n,kind", [(2, "none"), (3, "rigid"), (11, "rigid"), (12, "affine"), (19, "affine"),
                                    (20, "nonlinear")])
def test_transform_kind_by_count(slab, tmp_path, n, kind):
    _df, _c, st, mri, g, truth = load(slab)
    p = sc.landmark_pairs(st, g, mri)
    sub = sc.Pairs(p.labels[:n], p.slab_frame[:n], p.mri_lps[:n], p.sections[:n])
    res = sc.fit_transform(sub, g, mri, sc.TransformSettings(), tmp_path)
    assert res.kind == kind
    if kind != "none":
        pts = np.random.default_rng(0).uniform(g.bounds()[0] + [6, 0, 6], g.bounds()[1] - [6, 0, 6], (200, 3))
        err = np.linalg.norm(res.to_mri(sc.SlabGeometry.to_physical(pts, res.mirror)) - ms.true_map(pts, truth), axis=1)
        assert np.median(err) < (3.0 if kind == "rigid" and n == 3 else 1.0)
    if kind == "nonlinear":
        assert [Path(f).name.split("_")[1] for f in res.files] == ["nonlinear.nii.gz", "affine.mat"]


def test_nonlinear_starts_from_affine_and_composes_in_ants_order(slab, tmp_path):
    """antsApplyTransforms with [nonlinear, affine] must put an MRI point at its slab landmark."""
    ants = sc._ants()
    _df, _c, st, mri, g, _t = load(slab)
    res = sc.fit_transform(sc.landmark_pairs(st, g, mri), g, mri, sc.TransformSettings(), tmp_path)
    lab = res.residuals and next(iter(res.residuals))
    pairs = sc.landmark_pairs(st, g, mri)
    i = pairs.labels.index(lab)
    vox = np.round(mri.stats[lab].com).astype(int)
    img = np.zeros(mri.shape, np.float32)
    img[tuple(vox)] = 1000
    from scipy import ndimage
    img = ndimage.gaussian_filter(img, 1.5)
    o, s, d = mri.ants_geometry()
    moving = ants.from_numpy(img, origin=o, spacing=s, direction=d)
    p = pairs.slab_frame[i]
    ref = g._ref((40, 1, 40), (p[0] - 20 * 0.1, g.y[pairs.sections[i]], p[2] - 20 * 0.1), (0.1, 0.02, 0.1), res.mirror)
    out = sc.resample_plane(res, moving, ref)
    peak = np.unravel_index(np.argmax(out), out.shape)
    assert abs(peak[0] - 20) <= 6 and abs(peak[1] - 20) <= 6     # within 0.6 mm (MRI voxel 0.8 mm)


def test_mirrored_slab_detected(mirrored, tmp_path):
    df, cols, st, mri, g, truth = load(mirrored)
    conflicts = sc.other_slab_labels(df, cols, mirrored, KEY)
    pairs = sc.landmark_pairs(st, g, mri, conflicts)
    assert 5 in pairs.excluded
    assert sc.handedness_from_points(pairs.slab_frame, pairs.mri_lps) is True
    res = sc.fit_transform(pairs, g, mri, sc.TransformSettings(), tmp_path)
    assert res.mirror
    pts = np.random.default_rng(0).uniform(g.bounds()[0] + [6, 0, 6], g.bounds()[1] - [6, 0, 6], (200, 3))
    err = np.linalg.norm(res.to_mri(sc.SlabGeometry.to_physical(pts, res.mirror)) - ms.true_map(pts, truth), axis=1)
    assert np.median(err) < 1.0
    errs = [i for i in sc.validate_slab(st, mri, conflicts, res, g) if i.severity == "error"]
    assert any("also used in MR9-L-slab7" in i.message for i in errs)


def test_three_points_keep_previous_handedness(mirrored, tmp_path):
    _df, _c, st, mri, g, _t = load(mirrored)
    p = sc.landmark_pairs(st, g, mri)
    sub = sc.Pairs(p.labels[:3], p.slab_frame[:3], p.mri_lps[:3], p.sections[:3])
    res = sc.fit_transform(sub, g, mri, sc.TransformSettings(last_mirror=True), tmp_path)
    assert res.mirror and "can't be determined" in res.handedness_note


def test_inverse_mapping(slab, tmp_path):
    _df, _c, st, mri, g, _t = load(slab)
    res = sc.fit_transform(sc.landmark_pairs(st, g, mri), g, mri, sc.TransformSettings(), tmp_path)
    pts = np.random.default_rng(3).uniform(g.bounds()[0] + [6, 0, 6], g.bounds()[1] - [6, 0, 6], (20, 3))
    phys = sc.SlabGeometry.to_physical(pts, res.mirror)
    assert np.allclose(res.to_slab(res.to_mri(phys)), phys, atol=0.01)


def test_control_point_spacing_honoured(slab, tmp_path):
    _df, _c, st, mri, g, _t = load(slab)
    for spacing in (5.0, 10.0, 20.0):
        res = sc.fit_transform(sc.landmark_pairs(st, g, mri), g, mri, sc.TransformSettings(cp_spacing=spacing),
                               tmp_path)
        assert all(0.6 * spacing <= e <= 1.6 * spacing for e in res.eff_spacing), res.eff_spacing


def test_outputs_named_per_slab(tmp_path):
    n = sc.slab_output_names(tmp_path / "v1.1", KEY, "/x/v1.0/mri_landmarks_v1.0.nii.gz")
    stack = core.SpecimenStack.__new__(core.SpecimenStack)
    assert n.hemi(stack).name == "sub-MR9_hemi-L_chunk-6_landmarks_v1.1.nii.gz"
    assert n.template().name == "mri_landmarks_v1.1.nii.gz"


# -- PNG sections ------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def slab_png(tmp_path_factory):
    return _make(tmp_path_factory, "slab_png", "--preplaced", "22", "--png")


def load_png(d, pixel_size=0.15):
    df, cols, sections, _ = core.read_sections(d / "sections.csv", row_filter=KEY.matches)
    st = core.SpecimenStack(sections, label_dtype=np.uint16)
    st.id = KEY.id
    st.load_labels()
    mri = sc.DonorMRI(str(d / "mri.nii.gz"), str(d / "labels_in" / "mri_landmarks_v1.0.nii.gz"))
    return df, cols, st, mri, sc.SlabGeometry(st, pixel_size=pixel_size), dict(np.load(d / "truth.npz"))


def test_png_geometry_matches_nifti(slab, slab_png):
    _df, _c, st_n, _m, g_n, _t = load(slab)
    _df, _c, st_p, _m, g_p, _t = load_png(slab_png)
    assert g_p.raster.all() and not g_n.raster.any()
    assert np.array_equal(g_p.shapes, g_n.shapes) and np.allclose(g_p.spacing, g_n.spacing)
    assert np.allclose(g_p.extent, g_n.extent) and np.allclose(g_p.offset, g_n.offset)
    for i in (0, 4, 11):          # incl. the cropped sections
        a_n, _ = core.read_image(st_n.sections[i].raw)
        a_p, _ = core.read_image(st_p.sections[i].raw)
        assert np.abs(g_p.to_uv(i, a_p) - np.clip(a_n, 0, 255).astype(np.uint8)).max() <= 1   # same pixels as (u, v)


def test_png_landmarks_and_fit_match_truth(slab, slab_png, tmp_path):
    *_, st_n, mri_n, g_n, truth = load(slab)
    *_, st_p, mri_p, g_p, _ = load_png(slab_png)
    p_n = sc.landmark_pairs(st_n, g_n, mri_n)
    p_p = sc.landmark_pairs(st_p, g_p, mri_p)
    assert p_p.labels == p_n.labels
    assert np.allclose(p_p.slab_frame, p_n.slab_frame)          # identical slab positions from PNG and NIfTI
    assert np.linalg.norm(ms.true_map(p_p.slab_frame, truth) - p_p.mri_lps, axis=1).max() < 0.5
    res = sc.fit_transform(p_p, g_p, mri_p, sc.TransformSettings(), tmp_path)
    pts = np.random.default_rng(0).uniform(g_p.bounds()[0] + [6, 0, 6], g_p.bounds()[1] - [6, 0, 6], (200, 3))
    err = np.linalg.norm(res.to_mri(sc.SlabGeometry.to_physical(pts, res.mirror)) - ms.true_map(pts, truth), axis=1)
    assert res.kind == "nonlinear" and np.median(err) < 1.0


def test_png_uv_flat_round_trip(slab_png):
    *_, st, _m, g, _t = load_png(slab_png)
    i = 3
    nu, nv = g.shapes[i]
    arr = np.arange(nu * nv).reshape(nv, nu)                     # file layout (rows, cols)
    uv = g.to_uv(i, arr)
    u, v = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    assert np.array_equal(arr.ravel()[g.uv_to_native_flat(i, u, v)], uv)
    r, c = 7, 11
    assert g.native_to_uv(i, r, c) == (c, nv - 1 - r) and uv[c, nv - 1 - r] == arr[r, c]


# -- stacking like stack_view.py (master.csv-style CSV) ---------------------------- #
@pytest.fixture(scope="session")
def stacked(tmp_path_factory):
    return _make(tmp_path_factory, "stacked", "--stacked", "--preplaced", "22")


def load_stacked(d, cache_dir=None, csv="sections.csv"):
    df, cols, sections, warnings = core.read_sections(d / csv, row_filter=KEY.matches, section_required=False)
    st = sc.SlabStack(sections, label_dtype=np.uint16)
    st.id = KEY.id
    sc.compute_centres(st, cache_dir)
    st.load_labels()
    mri = sc.DonorMRI(str(d / "mri.nii.gz"), str(d / "labels_in" / "mri_landmarks_v1.0.nii.gz"))
    return df, cols, st, mri, sc.SlabGeometry(st, pixel_size=ms.S_PIX), dict(np.load(d / "truth.npz"))


def test_parse_repeat_and_order_modes():
    assert sc.parse_repeat("OL#HG#MR1S6#L#dpmg#6555#01#L") == 1
    assert sc.parse_repeat("OL#HG#MR1S6#L#pire#9507#UB#L") == 10 ** 6
    assert sc.parse_repeat("no_hashes") == 10 ** 6
    mk = lambda name, num, lig: core.Section(0, "x", "x", "L", num, f"/r/{name}.png", acquisition=lig)
    a = [mk("A#a#a#a#musc#1#02#L", 2, "musc"), mk("A#a#a#a#ampa#2#02#L", 2, "ampa"),
         mk("A#a#a#a#musc#3#UB#L", None, "musc"), mk("A#a#a#a#zz#4#01#L", 1, "zz")]
    assert sc.resolve_order(a)[0] == "stack"                         # shared and missing numbers
    st = sc.SlabStack(a)
    assert [s.name.split("#")[4] for s in st.sections] == ["zz", "ampa", "musc", "musc"]
    assert st.sections[-1].number is None and st.positions.tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="no section number"):
        sc.resolve_order(a, "section")
    b = [mk("s1", 10, None), mk("s2", 12, None)]
    assert sc.resolve_order(b)[0] == "section" and sc.SlabStack(b).positions.tolist() == [10, 12]
    assert sc.resolve_order(b, "stack")[0] == "stack"


def test_stacked_order_matches_stack_view(stacked):
    df, cols, st, _m, g, _t = load_stacked(stacked)
    assert cols["mask"] == "mask" and cols["base"] == "base" and cols["has_mask"] == "has_mask"
    assert st.order == "stack" and "stack_view" in st.order_note and "at the end" in st.order_note
    assert [int(df.loc[s.row, "true_rank"]) for s in st.sections] == list(range(len(st)))
    assert np.allclose(g.y, np.arange(len(st)) * 0.02)               # consecutive, one thickness apart
    assert st.sections[-1].number is None                           # "UB" sections (blank section cell) last


def test_mask_centring_aligns_masks(stacked):
    df, _c, st, _m, g, truth = load_stacked(stacked)
    assert g.centring == "mass" and st.centre_source.count("mask") == int(df.query("chunk == '6'")
                                                                           ["has_mask"].eq("True").sum())
    for i, s in enumerate(st.sections):
        ext = g.shapes[i] * g.spacing[i]
        assert (g.offset[i] >= -1e-9).all() and (g.offset[i] + ext <= g.extent + 1e-9).all()
        if st.centre_source[i] != "mask":
            continue
        d = df.loc[s.row, ["true_dx", "true_dz"]].astype(float).to_numpy()
        u, v = (truth["c_mask"] - d) / ms.S_PIX - 0.5                # true mask centre in this image
        p = g.section_to_frame(i, u, v)
        assert np.abs(p[[0, 2]] - g.centre).max() < 0.05, (i, s.name)   # incl. the two half-resolution masks


def test_stacked_landmarks_and_fit_match_truth(stacked, tmp_path):
    *_, st, mri, g, truth = load_stacked(stacked)
    pairs = sc.landmark_pairs(st, g, mri)
    assert pairs.labels == list(range(1, 23))
    shift = np.array([g.centre[0] - truth["c_mask"][0], 0.0, g.centre[1] - truth["c_mask"][1]])
    assert np.linalg.norm(ms.true_map(pairs.slab_frame - shift, truth) - pairs.mri_lps, axis=1).max() < 0.5
    res = sc.fit_transform(pairs, g, mri, sc.TransformSettings(), tmp_path)
    lo, hi = g.bounds()
    pts = np.random.default_rng(0).uniform(lo + [15, 0, 15], hi - [15, 0, 15], (300, 3))
    err = np.linalg.norm(res.to_mri(sc.SlabGeometry.to_physical(pts, res.mirror)) - ms.true_map(pts - shift, truth),
                         axis=1)
    assert res.kind == "nonlinear" and np.median(err) < 1.0


def test_master_like_csv_without_section_or_labels_columns(stacked):
    df = core.pd.read_csv(stacked / "sections.csv", dtype=str, keep_default_na=False)
    df.drop(columns=["section", "labels"]).to_csv(stacked / "bare.csv", index=False)
    assert KEY in sc.list_slabs(stacked / "bare.csv")
    with pytest.raises(ValueError, match="section"):
        core.read_sections(stacked / "bare.csv")                     # the macaque tool still needs them
    _df, cols, st, *_ = load_stacked(stacked, csv="bare.csv")
    assert "section" not in cols and st.order == "stack" and not st.all_labels()


def test_find_by_position_and_name(stacked):
    *_, st, _m, _g, _t = load_stacked(stacked)
    assert st.find("5") == (4, "") and st.find("#5") == (4, "")
    idx, msg = st.find("99999")
    assert idx == len(st) - 1 and "not in the stack" in msg
    name = st.sections[17].name
    assert st.find(name.split("#", 4)[4]) == (17, "")               # e.g. "dpmg#9123#03#L"
    assert st.find("nothing-like-this")[0] is None
    assert st.describe(17) == name and st.describe(17, short=True) == "#18"


def test_intensity_fallback_and_bad_masks(tmp_path):
    from PIL import Image
    img = np.zeros((60, 80), np.uint8)
    rr, cc = np.mgrid[0:60, 0:80]
    img[(rr - 20) ** 2 + (cc - 50) ** 2 < 36] = 200                  # blob centred at row 20, col 50
    Image.fromarray(img).save(tmp_path / "raw.png")
    r, c, src, note = sc.section_centre(str(tmp_path / "raw.png"), None)
    assert src == "intensity" and abs(r - 20) < 0.05 and abs(c - 50) < 0.05 and not note
    Image.fromarray(np.zeros((60, 80), np.uint8)).save(tmp_path / "empty.png")
    assert sc.section_centre(str(tmp_path / "raw.png"), str(tmp_path / "empty.png"))[2:] == ("intensity", "empty mask")
    Image.fromarray(np.full((60, 60), 255, np.uint8)).save(tmp_path / "square.png")
    src, note = sc.section_centre(str(tmp_path / "raw.png"), str(tmp_path / "square.png"))[2:]
    assert src == "intensity" and "doesn't match" in note
    m = np.zeros((30, 40), np.uint8)
    m[5:11, 20:30] = 255                                             # half resolution: rows 10-21, cols 40-59
    Image.fromarray(m).save(tmp_path / "half.png")
    r, c, src, _ = sc.section_centre(str(tmp_path / "raw.png"), str(tmp_path / "half.png"))
    assert src == "mask" and abs(r - 15.5) < 1e-9 and abs(c - 49.5) < 1e-9


def test_centres_cached(stacked, tmp_path, monkeypatch):
    *_, st, _m, _g, _t = load_stacked(stacked, cache_dir=tmp_path)
    first = st.centres.copy()
    monkeypatch.setattr(sc, "section_centre", lambda *a: (_ for _ in ()).throw(AssertionError("not cached")))
    *_, st2, _m, _g, _t = load_stacked(stacked, cache_dir=tmp_path)
    assert np.array_equal(st2.centres, first)


def test_auto_order_keeps_numbered_slabs(slab):
    df, cols, sections, _ = core.read_sections(slab / "sections.csv", row_filter=KEY.matches, section_required=False)
    st = sc.SlabStack(sections)
    assert st.order == "section" and st.positions.tolist() == [s.number for s in st.sections]
    g = sc.SlabGeometry(st)
    assert g.centring == "extent" and np.allclose(g.y, g.numbers * 0.02)


def test_new_csv_rebases_path_columns(stacked):
    df = core.pd.read_csv(stacked / "sections.csv", dtype=str, keep_default_na=False)
    cols = core.resolve_columns(df.columns)
    assert set(core._path_columns(df, cols)) == {"raw", "labels", "mask", "qc"}
