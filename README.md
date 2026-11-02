# Landmark labelling GUIs

Two tools for labelling corresponding landmark points on coronal
autoradiographs and a 3-D MRI, so BrainBuilder can initialise the 2-D/3-D
alignment from them:

- **`landmark_gui.py`** — several hemisphere stacks against a 3-D template
  (macaque). Described in the first part of this README.
- **`slab_gui.py`** — one human autoradiography slab against the donor MRI,
  with a live landmark transform. See [Slab tool](#slab-tool-human-slabs--donor-mri).

Each hemisphere stack gets its own pane, and one pane shows the coronal slices
of the template with its existing 3-D landmarks. Label values are shared
between all panes: **the same number is the same anatomical point** in every
hemisphere and in the template.

![layout](docs/layout.png)

## Files

| file | what it is |
|---|---|
| `landmark_gui.py` | the GUI |
| `landmark_qc.py` | QC figures (also run from the GUI's **QC figures…** button) |
| `landmark_core.py` | reading/writing, label statistics, validation, saving (no Qt) |
| `make_section_csv.py` | builds the input CSV from autoradiograph file names |
| `repair_landmark_csv.py` | fixes CSVs written before the path fix (see [Repairing an older CSV](#repairing-an-older-csv)) |
| `make_synthetic_data.py` | small fake dataset to try everything |
| `test_landmark_core.py` | tests (`python -m pytest test_landmark_core.py -q`) |
| `slab_gui.py` | slab tool: slab + donor MRI + floating "MRI in slab space" window |
| `slab_core.py` | slab geometry, landmark pairs, ANTs transform fitting/resampling (no Qt) |
| `make_synthetic_slab.py` | synthetic donor MRI + slab with a known mapping |
| `test_slab_core.py` | slab tests (`python -m pytest test_slab_core.py -q`) |

## Install

Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

## Input CSV

One row per section. `make_section_csv.py` builds it from file names like
`mr#rh#11530#L#musc#5768#37#L_crop.nii.gz`. In that example the subject is
`11530`, the hemisphere `L`, the ligand `musc` and the section number `37`.
Existing labels are matched as `<image name>_landmark.nii.gz` in the labels
folder:

```bash
python make_section_csv.py --root /data/macaque --images preprocessed --labels v1.3 \
    --out /data/macaque/landmark_sections_v1.3.csv
```

Column names are matched case-insensitively. Any of the names below work, or
pass `--col-sub`, `--col-hemi`, `--col-section`, `--col-raw` or `--col-labels`.

| role | accepted names |
|---|---|
| subject | `sub`, `subject`, `specimen`, … |
| hemisphere | `hemisphere`, `hemi`, `side` |
| section number | `section`, `sample`, `section_number`, … |
| autoradiograph path | `raw`, `autoradiograph`, `image`, `filepath`, … |
| section label path | `labels`, `label`, `landmark`, `landmarks`, `section_labels`, … (optional; empty cell = no labels yet) |
| ligand (optional, shown in the pane) | `ligand`, `acquisition` |
| chunk (optional, sorts sections) | `chunk`, `slab` |

- **Specimen IDs** are `<subject>-<hemisphere>`, e.g. `11530-L`. A `sub-`
  prefix on the subject is dropped.
- **Relative paths** are resolved against the CSV's folder.
- **Formats.** Autoradiographs can be 2-D NIfTI or PNG/TIFF.
- **Existing label images** must have the same shape as their autoradiograph.
  A mismatch stops the program at start-up rather than silently dropping
  labels.
- **Label files that load empty** (no labels, or negative values from a wrong
  data type) are listed in the Issues tab at start-up, along with label files
  that can't be found.

## Run

From the folder that holds `preprocessed/` and `v1.3/`:

```bash
python landmark_gui.py \
    --csv landmark_sections_v1.3.csv \
    --template MEBRAINS_T1.nii.gz \
    --template-labels v1.3/MEBRAINS_L_landmarks_v1.3.nii.gz \
    --out-dir v1.4 \
    --specimens 11530-L 11539-L 11539-R 11543-L \
    --flip-lr 11539-R
```

`--specimens` picks the stacks and their pane order (default: every specimen in
the CSV, in order of appearance). `--flip-lr` shows the listed specimens'
sections mirrored left-right, so a right hemisphere reads like the left ones;
its pane and QC panels are titled e.g. `11539-R-flipped-lr`. This is display
only: clicks map back through the mirror, and label files keep the original
orientation of the autoradiographs. The choice is remembered for `--resume` and
the QC script. The coronal axis of the template is read from
its affine; override it with `--coronal-axis 0|1|2`. If `--template-labels`
doesn't exist yet, you start with an empty label volume.

### Where things are saved

- **No edits, no output.** If you don't change any annotation, nothing is
  written and no folder is created. Editing and then undoing back to the loaded
  state also counts as no change. Browsing, *Go to label* and *Validate* never
  write.
- **The output folder is created on the first save that has edits.**
- **A new CSV is written with them, next to the output folder.** It goes in
  the project folder, beside `landmark_sections_v1.3.csv`, and every path in
  it is relative to that folder: `preprocessed/…` and `v1.4/…_landmark.nii.gz`.
- **Every section with labels is written to the new folder**, changed or not,
  so `v1.4/` is a complete landmark folder for BrainBuilder. A blank label
  cell means that section had no labels when saved.
- **Existing folders are never overwritten silently.** If `--out-dir` already
  contains files, or the new CSV already exists, the first save asks:
  **Choose another folder…**, **Overwrite**, or **Cancel**. `--overwrite`
  skips the question. Autosave never writes into such a folder by itself.
- **After the first save, the folder belongs to the session.** Later saves
  (manual, or autosave every 5 minutes, see `--autosave-min`) go to the same
  folder without asking.
- **Inputs are never modified,** unless you point `--out-dir` at their folder
  and choose to overwrite.
- **Deletion.** Only files this tool wrote itself are ever deleted, e.g. a
  section's label file after you erased all its labels. Any other file is left
  in place and listed in `validation_report.txt`.

**Continuing later.** Either load the saved labels and write a new version
folder:

```bash
python landmark_gui.py --csv landmark_sections_v1.4.csv \
    --template MEBRAINS_T1.nii.gz --template-labels v1.4/MEBRAINS_L_landmarks_v1.4.nii.gz --out-dir v1.5
```

or keep working in the same folder with `--out-dir v1.4 --resume --overwrite`.
If `--out-dir` holds a saved session and you didn't pass its CSV, the GUI asks
whether to load it.

## Using it

**Each pane** has:
- a slider, `−`/`+` buttons (one section back/forward), and a box to type a
  section number. In stacks with several chunks, `chunk:section` also works.
  If the number isn't in the stack, the pane jumps to the nearest section and
  tells you.
- **Contrast** sliders (percentiles of the section's intensities, so they adapt
  across ligands), plus **Auto** to reset them.

Hovering over a label shows its number; the status bar shows the pixel/voxel,
its intensity and world coordinates (template). Every label also carries a
number tag (toggle **Show label numbers**).

**Go to label** (toolbar box, or click a row in the **Labels** table):
- moves the template to the coronal slice through the label's centre of mass
  (rounded to the nearest integer) and rings it;
- moves every hemisphere that has the label to that section and rings it;
- hemispheres without the label stay where they are and say so;
- if the label isn't in the 3-D template, it says so and does nothing else;
- it also makes that label the current painting label.

Optional **Estimate section where missing** jumps hemispheres that lack the
label to the section predicted by a linear fit of section number against
template coronal position, using the landmarks they already share. It needs at
least two shared landmarks and a single chunk.

### Mouse

| action | effect |
|---|---|
| wheel, right-drag | zoom |
| left-drag (Navigate mode), middle-drag (any mode) | pan |
| left click/drag in **Paint** mode | paint the current label (brush radius in px for sections, voxels for the template; template painting is on the current coronal slice) |
| left click/drag in **Erase** mode | erase any label under the brush |
| **Shift+click** (Paint) | *move* the current label here: it is removed from every other section of this hemisphere (or elsewhere in the template), so it stays unique |
| **Ctrl+click** | make the label under the cursor the current label |

With **Don't overwrite other labels** ticked (the default), painting never
changes another label's pixels.

### Keys

| key | action |
|---|---|
| `P` / `E` / `N` (`Esc`) | Paint / Erase / Navigate mode |
| `←`/`↑`, `→`/`↓` | previous / next section in the pane under the mouse |
| `[` / `]` | brush smaller / larger |
| `F` | fit the pane under the mouse |
| `Ctrl+G` | go to the "Go to label" box |
| `Ctrl+Z` | undo |
| `Ctrl+S` | save |

**Next free** sets the current label to one more than any label in use, for
adding a new landmark.

## Rules that are checked

**Validate** (and every save) checks:

1. **Unique in the template.** Each label is one connected blob in 3-D
   (26-connectivity).
2. **Unique per hemisphere.** Each label is in only one section, as one blob
   (8-connectivity).
3. **Every hemisphere label exists in the template.** The reverse is not
   required: template points may be absent from a hemisphere.
4. **Two warnings for BrainBuilder.** Raw file names that are prefixes of each
   other are flagged, because BrainBuilder's `{root}*.nii.gz` glob would match
   two files. Two sections with the same file name are an error.

Problems appear in red in the **Labels** table and are listed in the
**Issues** tab; clicking an issue takes you to it. Saving always works (work in
progress is never lost), but a hemisphere's 3-D file is only written when that
hemisphere passes (see below).

### Per-specimen 3-D labels (File menu)

**File → Export specimen 3-D labels…** writes, for each specimen, a 3-D volume
holding only the template labels that specimen has on its sections, e.g.
`11530_L_landmarks_v1.4.nii.gz`. It works whether or not you have edited
anything.

- **Folder:** you choose it, so you can also refresh an existing version.
  Files are named after it (`v1.3` → `…_v1.3.nii.gz`), and existing files are
  replaced only after you confirm.
- **Unsaved edits:** the GUI offers to save first, so the 2-D files on disk
  match the volumes.
- **Validation errors:** if some specimens have them, you choose between
  exporting all specimens or only those without errors.
- **Labels without a 3-D counterpart** are left out of the volume and listed
  in the summary.

Saving also writes these files, but only for specimens without validation
errors.

### Repairing an older CSV

CSVs saved before this fix live *inside* the output folder, with paths relative
to it (`../preprocessed/…` and bare label file names). To rewrite one in the
project-folder convention:

```bash
python repair_landmark_csv.py v1.4/landmark_sections_v1.4.csv   # writes ./landmark_sections_v1.4.csv
```

`--fill-from v1.3` also fills sections without labels from
`v1.3/<image name>_landmark.nii.gz`, but **only if none of that file's labels
is already on another section of the same specimen**. A label moved with
Shift+click leaves its old section blank, and filling that section back would
put the label in two sections. Those sections are listed and left blank;
`--no-check` fills them anyway.

## Outputs (`--out-dir`)

The layout is flat and named like your existing landmark folders, so `v1.4/`
looks like `v1.3/`. The version tag is the folder name when it looks like
`v1.4`; set it with `--version-tag`, or leave it out for a plain name.

| file | contents |
|---|---|
| `<image name>_landmark.nii.gz` | 2-D labels for every section that has any. Same shape/layout as the autoradiograph; NIfTI copies its header/affine. The suffix is set by `--label-suffix`. `--label-format match` (default) writes NIfTI for NIfTI images and 16-bit PNG for PNG images. |
| `../landmark_sections_v1.4.csv` | **the new CSV**, in the project folder next to the output folder: your input CSV with the label column pointing at the new files (empty = no labels); paths are relative to the project folder (`preprocessed/…`, `v1.4/…`) |
| `MEBRAINS_L_landmarks_v1.4.nii.gz` | all 3-D labels, original and new. The name comes from `--template-labels` with its version replaced. |
| `11530_L_landmarks_v1.4.nii.gz`, … | per hemisphere: only the 3-D labels that hemisphere also has, so the 2-D↔3-D mapping is strictly one-to-one. Written only when that hemisphere has no validation errors (File → Export specimen 3-D labels writes them on demand). |
| `landmark_table.csv` | one row per label per space: centre of mass (voxel and mm), section, voxel count, blob count |
| `landmark_coverage.csv` | label × space matrix |
| `validation_report.txt` | the last validation |
| `qc/` | QC figures (with nothing saved, the QC button asks where to put them) |
| `session.json`, `gui_state.json` | for resuming: inputs, columns, files written, pane positions, contrast, brush sizes |

The per-hemisphere 3-D files are regenerated from the template label volume
each time; existing per-hemisphere files (e.g. `v1.3/11530_L_landmarks_v1.3.nii.gz`)
are not read.

## QC figures

One page per label. The template's coronal slice through the label sits next to
the section holding it in each hemisphere. The top row shows whole sections with
a crosshair, to compare anatomical position at a glance; the bottom row zooms
in. Missing labels get an empty panel, and problems are written in the panel
title. Page 1 is the coverage table.

```bash
python landmark_qc.py --out-dir v1.4                  # all labels of a saved session
python landmark_qc.py --out-dir v1.4 --labels 3 12    # some
```

## BrainBuilder

- **2-D landmarks.** Use the output folder (e.g. `v1.4/`) as the landmark directory.
  BrainBuilder finds each section's file with `{raw_root}*.nii.gz`, so the
  labels must be NIfTI. That happens automatically when the CSV's
  autoradiographs are BrainBuilder's NIfTI files; otherwise use
  `--label-format nifti`.
- **3-D landmarks.** Use `<sub>_<hemi>_landmarks_<tag>.nii.gz` as that
  hemisphere's `ref_landmark` in `chunk_info`.
- **Pixel layout.** Label files have exactly the pixel layout of the image
  referenced in the CSV. If BrainBuilder reads a different file (e.g. a NIfTI
  converted from these PNGs), convert the labels the same way, with
  nearest-neighbour interpolation.

## Try it

```bash
python make_synthetic_data.py demo
python landmark_gui.py --csv demo/sections.csv --template demo/template.nii.gz \
    --template-labels demo/template_landmarks.nii.gz --out-dir demo/out
```

The demo has two planted problems for the checks to find: a label placed in two
sections of 11539-R, and a label 99 in 11543-L with no template counterpart.

## Notes

- **Orientation.** Images are shown as stored (except specimens given to
  `--flip-lr`): PNG/TIFF row 0 at the top; 2-D NIfTI with x left→right and
  y bottom→top, like BrainBuilder's QC plots.
  The template's coronal slice is shown neurological (subject left on the
  left) and superior up, from its affine. Nothing is flipped or rotated on
  disk, and no orientation is recorded.
- **What BrainBuilder uses.** It only uses each landmark's centre of mass, so
  a small disk (default radius 3 px; 1 voxel in the template) is enough.
- **Large images.** Tested at 3120×4164 px per section: about 0.2–0.4 s to
  change section, under 0.1 s per click.

---

# Slab tool (human slabs → donor MRI)

`slab_gui.py` shows one slab of coronal autoradiograph sections next to the
donor MRI, each in coronal, sagittal and axial views. A floating window shows
the MRI resampled into slab space with `antsApplyTransforms`. The transform is
fitted to the landmark pairs and updates as you edit.

![slab layout](docs/slab_layout.png)

## Input

The same kind of CSV as above, one row per section, plus a **slab column**
(`chunk` or `slab`). Sections can be **2-D NIfTI** (pixel size from each
file's header) or **PNG/TIFF**. PNG/TIFF files have no header, so their pixel
size comes from `--pixel-size` (default **0.2 mm**, i.e. the 20 µm originals
downsampled 10×). PNG sections are shown as stored, row 0 at the top. Their
labels are painted and saved in the PNG's own layout, as 16-bit PNGs
(`--label-format match`, the default; `nifti` forces `.nii.gz`). Existing
labels (`labels`/`landmark` column) are optional.

A `master.csv` like the one `stack_view.py` reads works as it is (`base`,
`ligand`, `raw`, `mask`, `has_mask`, `section`, …); other columns are kept in
the new CSV.

| role | accepted names |
|---|---|
| subject, hemisphere | as above |
| slab | `chunk`, `slab` |
| section number | `section`, `sample`, … (optional; see *Stacking*) |
| image | `raw`, `autoradiograph`, … (2-D NIfTI or PNG/TIFF) |
| section labels | `labels`, `landmark`, … (optional) |
| ligand | `ligand`, `acquisition`, … (optional; used for the stacking order) |
| tissue mask | `mask` (optional; used only where `has_mask` is true, if that column exists) |
| section name | `base` (optional; default: the image file name) |

## Run

```bash
python slab_gui.py --csv sections.csv --mri MR1_T1.nii.gz \
    --mri-labels v1.0/mri_landmarks_v1.0.nii.gz \
    --sub MR1 --hemisphere L --chunk 6 --out-dir v1.1
# PNG sections are assumed to be 0.2 mm/pixel; for another resolution add e.g. --pixel-size 0.1

# straight from stack_view.py's master.csv (raw/*.png, mask_final/*.png):
python slab_gui.py --csv master.csv --mri MR1_T1.nii.gz --sub MR1 --hemisphere L --chunk 6 --out-dir v1.0
```

- **Choosing the slab.** If the CSV holds several slabs and you don't pick
  one, the GUI asks. Other slabs of the same subject and hemisphere are read
  only to check label uniqueness.
- **No `--mri-labels` file yet?** You start with an empty MRI label volume.

## Stacking (as in `stack_view.py`)

**Order along the slab (`--order`, default `auto`).**
- `stack`: like `stack_view.py`. Sections are sorted by (repeat, ligand, name);
  the repeat is the 7th `#` field of the name (`01` in
  `OL#HG#MR1S6#L#dpmg#6555#01#L`). Consecutive sections are
  `--slice-thickness` (0.02 mm) apart. Names without a repeat number (e.g.
  `…#UB#L`, which have no section number in `master.csv`) go at the end.
- `section`: position = `section number × --slice-thickness`.
- `auto`: `section` if every section has its own number, otherwise `stack`.
  In `master.csv` the `section` column is the repeat number, shared by all the
  ligands of a repeat (a slab of 591 sections has only 26 numbers), so `auto`
  stacks every slab like `stack_view.py`.

**In-plane placement (`--center`, default `auto`).**
- `mask`: like `stack_view.py`. Each section is shifted so that the centre of
  mass of its tissue mask (the `mask` column, e.g. `mask_final/<base>.png`)
  falls on the same point for all sections. A mask stored at another
  resolution is scaled. Sections without a usable mask (no file, `has_mask`
  false, empty mask) use the intensity-weighted centroid of the image, as in
  `stack_view.py`.
- `extent`: sections of different sizes are centred in a common extent, as
  BrainBuilder pads them.
- `auto`: `mask` if the CSV has a mask column, otherwise `extent`.

The **Issues** tab says which order and placement were used, how many
sections were centred on a mask, and which masks couldn't be used. Centres are
cached with the downsampled slab, so reopening a slab is quick. The 2-D label
files are always saved in each image's own pixel layout; the stacking only
decides where the sections sit in the slab frame used for the views and the
transform.

`stack_view.py` also applies the flips/rotations in `splits/*.qc.csv` and
prefers edited masks in `mask_qc/`. The slab tool does neither: it shows the
images as stored and uses the `mask` column.

## Geometry and display

- **In-plane pixel size:** from each NIfTI header, or `--pixel-size` for
  PNG/TIFF.
- **Physical aspect.** Every view is drawn at its true physical size, so
  nothing is stretched along the section axis. A slab 11 mm thick shows as a
  thin band in the sagittal and axial views; zoom with the wheel.
- **Coronal slab view:** the current section at full resolution. This is
  where slab landmarks are painted.
- **Sagittal/axial slab views:** built from a downsampled copy of every
  section (each section contrast-normalised). They fill in within a few
  seconds per hundred sections and are cached in `~/.cache/slab_landmark_gui`
  (`--cache-dir`, `--no-cache`); the cache never touches `--out-dir`.
  Landmarks near those planes are drawn as rings.
- **MRI views:** neurological coronal/axial (subject left on screen left),
  sagittal with anterior on the left. MRI landmarks can be painted in any of
  the three views.
- **`--flip-lr`:** shows the slab mirrored (display only), as in the
  macaque tool.

## Mouse and keys

| action | effect |
|---|---|
| **Paint mode (P)**: left click/drag | paint the current label (slab: coronal view; MRI: any view) |
| **Paint mode**: right click/drag | erase |
| **Navigate mode (N/Esc)**: left click | move the cursor (the other views follow) |
| Navigate: left-drag / right-drag | pan / zoom |
| any mode: wheel, middle-drag | zoom, pan |
| any mode: double-click | move the cursor there |
| Ctrl+click | pick the label under the cursor |
| Shift+click (Paint) | move the current label here (removes it elsewhere in the slab, or in the MRI) |
| ←/→, ↑/↓ | previous/next section (or MRI coronal slice, pane under the mouse) |
| box next to the section slider | go to a section: its number, or in a stacked slab its position (1…n, as shown above the views) or part of its name, e.g. `dpmg#6555#01` |
| `[` `]`, F, Ctrl+G, Ctrl+Z, Ctrl+S | brush size, fit views, go to label, undo, save |
| Ctrl+M (or View menu, or the "MRI in slab space" toolbar button) | show/hide the floating window; it opens over the lower-right of the main window and stays in front of it |

Clicking the slab's sagittal/axial views or the floating window always moves
the cursor.

**Go to label:**
- moves the MRI to the label's centre of mass (rounded to the nearest voxel)
  and the slab to the section that has it;
- if the slab doesn't have it yet but a transform exists, shows the
  **predicted** position as a dashed ring, so you can Shift+click there to
  place it;
- if the label isn't in the MRI, says so and does nothing else.

## The transform

**Points.** Each landmark is reduced to its centre of mass:
- slab: the section pixel → slab millimetres;
- MRI: the voxel → physical (LPS) coordinates.

A pair is used only if the label is one blob in one section of the slab, one
blob in the MRI, and not used in another slab. Labels left out are listed in
the **Transform** tab with the reason.

**Type by number of pairs** (adjustable in the **Transform** tab or on the
command line):

| pairs | transform |
|---|---|
| < 3 (`--rigid-min`) | none (the floating window says so) |
| 3–11 | rigid |
| 12–19 (`--affine-min`) | affine |
| ≥ 20 (`--nonlinear-min`) | affine + B-spline |

The defaults follow BrainBuilder (rigid below 12 and affine below 20; nonlinear
from 20).

**Nonlinear = affine first, then B-spline.** As in BrainBuilder, the B-spline
is fitted after the affine:
1. The MRI landmarks are mapped into slab space with the inverse affine.
2. The B-spline displacement field is fitted between those points and the
   slab landmarks.

BrainBuilder does step 1 by resampling the label volume through the affine.
Doing it on the centroids is the same operation without nearest-neighbour
resampling error.

- **ANTs functions:** `ants.fit_transform_to_paired_points` (rigid/affine),
  `ants.fit_bspline_displacement_field` (B-spline) and `ants.apply_transforms`
  (resampling).
- **Transform order:** `[nonlinear, affine]`, as BrainBuilder concatenates
  them. Checked in `test_slab_core.py`: the reversed order puts an MRI point
  several mm away from its slab landmark.

**Control-point spacing** (default 10 mm, `--cp-spacing`, live in the
Transform tab) is the B-spline control-point spacing at the finest level.
`--levels` (default 4) is the maximum number of coarse-to-fine levels. The
mesh doubles per level, so the slab's thin axis limits how many levels fit.
The status line shows the effective spacing on each axis.

**Handedness.** antspyx's rigid *and* affine landmark fits only produce proper
rotations, so a slab that is mirror-imaged relative to the MRI can't be fitted
directly.
- **Auto (default):** the GUI detects this from the landmarks (a least-squares
  affine with negative determinant) and mirrors the slab's coordinate frame
  in x before fitting.
- **Points needed:** at least 4 landmarks not in one plane. With only 3, it
  keeps the last decision.
- **Override:** "Slab handedness" in the Transform tab.

The window's status line shows the transform type, the number of pairs, the
RMS error, and whether the slab frame was mirrored. The **Fit error** column
gives each landmark's residual in MRI millimetres; residuals above 2 mm are
flagged in the Issues tab.

## Rules and outputs

**Saving** works as in the macaque tool:
- nothing is written unless an annotation changes;
- the first save with changes creates `--out-dir`;
- a non-empty folder needs confirmation or `--overwrite`;
- `--resume` / the resume prompt brings back the slab, labels and view
  positions.

**Checks:**
- each label is one blob in the MRI, and one blob in one section of the slab;
- every slab label exists in the MRI;
- **a label can't be in two slabs** of the same subject and hemisphere.

| file | contents |
|---|---|
| `<image name>_landmark.nii.gz` | 2-D labels of each labelled section in this slab (raw image header/affine) |
| `../landmark_sections_<tag>.csv` | in the project folder next to the output folder: the input CSV (all slabs) with this slab's label paths updated; paths relative to the project folder |
| `<mri labels stem>_<tag>.nii.gz` | all MRI landmarks (e.g. `mri_landmarks_v1.1.nii.gz`) |
| `sub-<sub>_hemi-<h>_chunk-<c>_landmarks_<tag>.nii.gz` | MRI landmarks of **this slab only**, one-to-one with its 2-D labels (BrainBuilder's per-chunk `ref_landmark`). Written only when the slab has no validation errors. |
| `landmark_fit.csv` | each pair: section number, position in the stack, name, slab mm, MRI RAS mm, fit error, transform type; excluded labels with the reason |
| `landmark_table.csv`, `landmark_coverage.csv`, `validation_report.txt`, `session.json`, `gui_state.json` | as above |

## Try it

```bash
python make_synthetic_slab.py demo_slab              # 14 landmarks placed -> affine
python slab_gui.py --csv demo_slab/sections.csv --mri demo_slab/mri.nii.gz \
    --mri-labels demo_slab/labels_in/mri_landmarks_v1.0.nii.gz --chunk 6 --out-dir demo_slab/v1.1
```

- Labels 15–22 exist in the MRI only: *Go to label* shows where the
  transform predicts them on the slab.
- `--mirror` makes the slab mirror-imaged relative to the MRI.
- `--conflict` also puts label 5 in slab 7.
- `--stacked` writes a `master.csv`-like slab instead: PNG sections named with
  repeat numbers, `section` = repeat, masks in `mask_final/` and random
  in-plane shifts that the mask centring undoes. Open it with
  `--pixel-size 0.3`.

## Performance

Measured on 3120×4164-pixel sections (0.02 mm):
- changing to a prefetched section ≈ 0.13 s, ≈ 0.75 s otherwise;
- a paint click ≈ 0.1 s;
- building the sagittal/axial views ≈ 0.4 s per section in the background
  (instant from the cache);
- each labelled section holds a 16-bit label image (26 MB at that size).

Fitting takes milliseconds. Resampling one plane through the transform takes
about 0.05 s.
