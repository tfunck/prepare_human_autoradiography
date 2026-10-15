# Autoradiograph QC GUI

A small desktop tool for quality-controlling raw 2-D autoradiograph images and
their crop-mask overlays. For each section it shows the raw image with its mask
drawn on top as a translucent red overlay, and lets a reviewer fix the section
orientation, correct or draw the mask by hand, and adjust contrast so even very
low-signal ligands are visible.

The raw images and the saved masks are **never** rotated or flipped on disk —
only the flip/rotation you choose is **recorded** to a CSV, so the pixel data
stays aligned and nothing is destructively modified. Painting is mapped back
through the current transform, so an edited mask always stays aligned with the
raw image.

---

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Recommended graphical tools](#recommended-graphical-tools)
  - [Step 1: Get the code](#step-1-get-the-code)
  - [Step 2: Install Python and the dependencies](#step-2-install-python-and-the-dependencies)
    - [Windows](#windows)
    - [macOS](#macos)
    - [Linux (Ubuntu / Debian)](#linux-ubuntu--debian)
- [Preparing the data](#preparing-the-data)
- [Running the GUI](#running-the-gui)
- [Using the GUI](#using-the-gui)
  - [Window layout](#window-layout)
  - [Keyboard shortcuts](#keyboard-shortcuts)
  - [Mouse](#mouse)
  - [Contrast and opacity](#contrast-and-opacity)
  - [Typical review workflow](#typical-review-workflow)
- [Outputs](#outputs)
- [Checking the edited masks](#checking-the-edited-masks)
- [Command-line options](#command-line-options)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [License](#license)

---

## Features

- Move through sections with the **arrow keys**.
- **Rotate** the image and overlay in 90° steps.
- **Flip** the image and overlay left/right or up/down.
- **Draw** and **erase** on the overlay with the mouse, plus a one-click
  **Clear** to wipe the whole overlay. Missing masks start as a blank overlay
  you can paint yourself.
- **Overlay opacity** slider to fade the mask in and out.
- **Contrast** (min/max) sliders with an **Auto** stretch, to reveal
  low-contrast ligands that are otherwise impossible to see.
- Every flip/rotation is **recorded** to a CSV, and edited masks are saved to a
  separate folder. Work autosaves and **resumes** where you left off.

---

## Repository layout

```
prepare_qc_data.py      # builds raw/, mask/, qc/ and the csv/ files from source TIFFs
qc_gui.py               # the QC GUI (this tool)
make_qc_overlays.py     # render raw + edited-mask overlays to check mask_qc/
common.py               # shared flip/rotation helpers used by the GUIs
rotate_gui.py           # related NIfTI flip/rotate GUI (not required for QC)
requirements.txt        # Python dependencies
csv/
  master.csv            # one row per image
  splits/               # one CSV per subject_hemisphere_ligand (the GUI input)
raw/                    # downsampled 8-bit PNGs (generated; not tracked in git)
mask/                   # crop-mask PNGs      (generated; not tracked in git)
qc/                     # raw+mask QC PNGs    (generated; not tracked in git)
mask_qc/                # masks you edit in the GUI (created on first save)
mask_qc_overlays/       # raw+mask overlays from make_qc_overlays.py
```

> **Note:** `raw/`, `mask/`, `qc/` and `csv/` are listed in `.gitignore` because
> the image data is large. A fresh clone of this repository will **not** contain
> them — you generate them with `prepare_qc_data.py` (see
> [Preparing the data](#preparing-the-data)) or copy them in from wherever the
> data lives.

---

## Requirements

- **Python 3.8 or newer**
- Python packages (installed via `requirements.txt`): `numpy`, `pandas`,
  `Pillow` (the GUI), plus `imageio` and `matplotlib` (data preparation).
- **Tkinter** — Python's built-in GUI toolkit. It ships with the official
  Python installers on Windows and macOS, but on Linux it must be installed as a
  separate system package (`python3-tk`). Installation steps below cover this
  for each OS.

A graphical desktop is required (the GUI opens a window). It will not run on a
headless server without a virtual display or remote-desktop/X-forwarding setup.

---

## Installation

> **New to git or the command line?** Start with
> [Recommended graphical tools](#recommended-graphical-tools) — you can get
> everything running with almost no typing.

### Recommended graphical tools

You do not need to be a programmer to run this tool. These free apps give you a
point-and-click way to get the code and manage Python:

| Purpose | Tool | Notes |
| ------- | ---- | ----- |
| Get the code (no git) | **Download ZIP** from the GitHub page | Click the green **Code** button → **Download ZIP**, then unzip. Nothing to install. |
| Git with a GUI | **[GitHub Desktop](https://desktop.github.com/)** | Windows & macOS. Click **Clone** to download the repo, then **Repository → Open in Command Prompt / Terminal** to get a terminal already in the right folder. |
| Python with a GUI | **[Anaconda](https://www.anaconda.com/download)** | Windows, macOS & Linux. Installs Python plus **Anaconda Navigator** (graphical) and **Anaconda Prompt** (a ready-to-use terminal). |
| Beginner Python editor | **[Thonny](https://thonny.org/)** | Bundles its own Python and has a simple interface for running scripts. |

> Even with these tools you will type **two short commands** once — to install
> the dependencies and start the program. Copy and paste them exactly as shown.

### Step 1: Get the code

- **Option A — no git (simplest):** on the project's GitHub page, click the
  green **Code** button → **Download ZIP**, then unzip it somewhere easy to find
  (such as your Desktop).
- **Option B — GitHub Desktop (a git GUI):** install GitHub Desktop, click
  **Clone**, paste the repository URL, and choose a local folder.
- **Option C — command line:**
  ```bash
  git clone https://github.com/<your-org>/<your-repo>.git
  cd <your-repo>
  ```

### Step 2: Install Python and the dependencies

Open a terminal **in the project folder** (the one containing `qc_gui.py`). The
GUI tools above can open one for you:

- **Anaconda:** open **Anaconda Prompt** from the Start menu / Launchpad.
- **GitHub Desktop:** **Repository → Open in Command Prompt / Terminal**.
- **Windows (no tools):** open the folder in File Explorer, click the address
  bar, type `cmd`, and press Enter.
- **macOS (no tools):** right-click the folder in Finder → **New Terminal at
  Folder**.
- **Linux (no tools):** right-click inside the folder → **Open in Terminal**.

Then follow the steps for your operating system below.

> The virtual environment (`.venv`) is optional but recommended. For the
> simplest path you may skip the `venv` and `activate` lines and just run
> `pip install -r requirements.txt` — that installs into your main Python and
> lets you **double-click the program on Windows** (see
> [Running the GUI](#running-the-gui)).

#### Windows

Install Python 3 from <https://www.python.org/downloads/windows/> and keep the
default option **"tcl/tk and IDLE"** checked (this provides Tkinter).

Open **PowerShell** or **Command Prompt** in the project folder:

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If `py` is not recognized (for example, if you installed **Anaconda**), use
> `python` in place of `py -3`, or run the commands in **Anaconda Prompt**.

Run the GUI:

```powershell
python qc_gui.py csv\splits\MR1_L_afdx.csv
```

#### macOS

Install Python 3 from <https://www.python.org/downloads/macos/> (the official
installer includes Tk). In **Terminal**, from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the GUI:

```bash
python qc_gui.py csv/splits/MR1_L_afdx.csv
```

> If you installed Python with **Homebrew** instead, also install the Tk add-on:
> `brew install python-tk`.

#### Linux (Ubuntu / Debian)

Install Python, venv and the Tkinter system package:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk
```

Then, from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the GUI:

```bash
python qc_gui.py csv/splits/MR1_L_afdx.csv
```

---

## Preparing the data

The GUI reads a **split CSV** (one subject/hemisphere/ligand per file) that
points at the raw and mask PNGs. These are produced by `prepare_qc_data.py`,
which converts the source `.TIF` autoradiographs to 8-bit PNGs **downsampled to
the crop-mask resolution** (about 10x smaller, so they are easy to share),
writes the crop masks at their native resolution, and builds `csv/master.csv`
plus one file per group in `csv/splits/`.

If you already have the `raw/`, `mask/` and `csv/` folders, skip this step.
Otherwise, point the script at your source images and masks:

```bash
python prepare_qc_data.py \
  --raw-src  /path/to/raw_tiffs \
  --mask-src /path/to/crop_masks \
  --out-dir  .
```

This creates `raw/`, `mask/`, `qc/`, `csv/master.csv` and `csv/splits/*.csv` in
the output directory. Raw images are downsampled onto each mask's pixel grid;
images without a mask are downsampled by `--downsample` (default 10). Re-running
is safe — existing outputs are skipped unless you pass `--overwrite`. See
`python prepare_qc_data.py --help` for all options (filter by `--subjects` /
`--hemispheres` / `--ligands`, `--limit` for a quick test, `--workers` for
parallelism, etc.).

The split CSVs are designed so QC work can be divided between people — give each
reviewer one or more `csv/splits/*.csv` files.

---

## Running the GUI

From a terminal in the project folder (with the virtual environment active, if
you created one):

```bash
# Linux/macOS
python qc_gui.py csv/splits/MR1_L_afdx.csv

# Windows
python qc_gui.py csv\splits\MR1_L_afdx.csv
```

**Don't want to type a file path?** Run it with no argument and a file picker
opens so you can choose the split CSV:

```bash
python qc_gui.py
```

On **Windows**, if you installed the dependencies into your main Python (rather
than only a virtual environment), you can also simply **double-click
`qc_gui.py`** in File Explorer — the same file picker appears. On macOS/Linux,
start it from the terminal or your Python editor's **Run** button.

Output locations are chosen automatically (see [Outputs](#outputs)) and can be
overridden — see [Command-line options](#command-line-options).

---

## Using the GUI

### Window layout

```
┌───────────────────────────────────────────────────────────────┐
│  < Prev  Next >   Rotate ±90   Flip H/V   Mode   Clear   Save  │  toolbar
├───────────────────────────────────────────────────────────────┤
│  Overlay opacity   Contrast min   Contrast max   Auto  Reset   │  sliders
├───────────────────────────────────────────────────────────────┤
│                                                               │
│            raw image with red mask overlay                    │  canvas
│                                                               │
├───────────────────────────────────────────────────────────────┤
│  3 / 137   OL#...#afdx#...   mask: edited (reviewed)   flip…   │  status
└───────────────────────────────────────────────────────────────┘
```

The status bar shows your position in the file, the current section name, where
the mask came from (`original`, `edited`, or `empty`) and whether it has been
reviewed, and the flip/rotation currently applied.

### Keyboard shortcuts

| Key          | Action                                   |
| ------------ | ---------------------------------------- |
| Left / Up    | Previous section                         |
| Right / Down | Next section                             |
| `r`          | Rotate 90°                               |
| `h`          | Flip horizontal (left/right)             |
| `v`          | Flip vertical (up/down)                  |
| `e`          | Toggle the left-mouse mode (draw/erase)  |
| `c`          | Clear the whole overlay                  |
| `[`          | Smaller brush                            |
| `]`          | Larger brush                             |
| `s`          | Save now                                 |
| `Esc` / `q`  | Save and quit                            |

### Mouse

| Action              | Effect                                             |
| ------------------- | -------------------------------------------------- |
| Left-click / drag   | Paint the mask (or erase if Mode is set to Erase)  |
| Right-click / drag  | Erase the mask (always)                            |
| Mouse wheel         | Change brush size                                  |
| Move over the image | Shows a circle previewing the current brush size   |

All toolbar actions are also available as buttons, so the GUI can be used with
the mouse alone.

### Contrast and opacity

- **Overlay opacity** — fades the red mask between invisible and fully opaque so
  you can inspect the raw pixels underneath.
- **Contrast min / max** — remaps the displayed brightness: anything at or below
  *min* becomes black, anything at or above *max* becomes white, with a linear
  ramp in between. Narrow the window to boost contrast.
- **Auto** — sets the window to the current image's actual darkest/brightest
  values. This is the quickest way to make a faint, low-signal ligand visible.
- **Reset** — restores the full range (no contrast change).

Contrast and opacity affect **display only** — they are never written to the raw
images or the saved masks. The settings persist as you move between sections,
which is convenient because each split CSV is a single ligand.

### Typical review workflow

1. Open a split CSV. The first section loads with its mask overlaid.
2. If the image looks too dark/flat, click **Auto** (or drag the contrast
   sliders) until the tissue is clearly visible.
3. Fix the orientation with **Rotate** / **Flip H** / **Flip V** if needed. The
   overlay rotates and flips together with the image.
4. Correct the mask: left-drag to add, right-drag to remove, or **Clear** and
   repaint. For sections with no mask, paint one from scratch.
5. Press **Right** to go to the next section. Your flip/rotation and edited mask
   are saved automatically.
6. Repeat to the end of the file. You can quit any time with `q`/`Esc` and
   re-open the same CSV later to resume.

---

## Outputs

Two things are written as you work (both autosave, with a short debounce, and on
navigation/quit):

1. **A results CSV** next to the input, named `<input>.qc.csv`
   (e.g. reviewing `csv/splits/MR1_L_afdx.csv` writes
   `csv/splits/MR1_L_afdx.qc.csv`). It contains every original column plus:

   | Column        | Meaning                                                         |
   | ------------- | --------------------------------------------------------------- |
   | `flip`        | Flip applied: empty, `LR`, `UP`, or `LR+UP`                      |
   | `rotation`    | Number of 90° steps applied (integer; `90° × rotation`)         |
   | `qc_mask`     | Path to the edited mask for this section (if any)               |
   | `qc_reviewed` | `True` once the section has been displayed in the GUI           |

2. **Edited masks** saved at the raw image's resolution to `mask_qc/`
   (one PNG per section, named after the section). Original masks in `mask/` are
   left untouched.

Re-opening the same CSV resumes from these files: previously recorded
flips/rotations are restored and your edited masks are loaded instead of the
originals.

> Both outputs are covered by `.gitignore` (`mask_qc/`, `mask_qc_overlays/` and
> the `csv/` folder), so your review artifacts are **not** committed to the
> public repository by default. Keep a separate backup/transfer of them if you
> need to share results.

---

## Checking the edited masks

After reviewing sections in the GUI you will have edited masks in `mask_qc/` and
a `<split>.qc.csv`. To eyeball them against the raw images, render overlay PNGs
(raw in grey, edited-mask outline in red, in the flip/rotation you chose) with:

```bash
python make_qc_overlays.py csv/splits/MR1_L_afdx.qc.csv
```

This writes one PNG per reviewed section to `mask_qc_overlays/`. It only
processes rows that have an edited mask, skips overlays that already exist (use
`--overwrite` to rebuild), and runs in parallel. These are the same style of
overlay `prepare_qc_data.py` writes to `qc/`, so you can flip through the folder
of thumbnails to spot any bad masks quickly.

| Option        | Default                   | Description                              |
| ------------- | ------------------------- | ---------------------------------------- |
| `CSV`         | *(required)*              | The `*.qc.csv` written by the GUI.       |
| `--root`      | auto-detected             | Directory holding `raw/` and `mask_qc/`. |
| `--out-dir`   | `<root>/mask_qc_overlays` | Where to write the overlay PNGs.         |
| `--workers`   | CPUs − 1                   | Parallel worker processes.               |
| `--overwrite` | off                       | Rebuild overlays that already exist.     |
| `--qc-dpi`    | 100                       | Resolution of the overlay figures.       |
| `--limit N`   | 0 (all)                   | Only build the first N overlays.         |

---

## Command-line options

```
python qc_gui.py [CSV] [--root DIR] [--out-csv FILE] [--mask-out-dir DIR]
```

| Option           | Default                              | Description                                                           |
| ---------------- | ------------------------------------ | --------------------------------------------------------------------- |
| `CSV`            | *(file picker if omitted)*           | The split CSV to review, e.g. `csv/splits/MR1_L_afdx.csv`.            |
| `--root`         | auto-detected                        | Directory that holds `raw/` and `mask/`. Auto-found from the CSV path. |
| `--out-csv`      | `<CSV>.qc.csv`                       | Where to write the recorded flips/rotations.                          |
| `--mask-out-dir` | `<root>/mask_qc`                     | Directory for the masks you edit.                                     |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tkinter'` (or `_tkinter`)**
Tkinter is not installed. On Ubuntu/Debian: `sudo apt install python3-tk`. On
macOS with Homebrew Python: `brew install python-tk`. On Windows, re-run the
Python installer and make sure **"tcl/tk and IDLE"** is selected.

**`No module named numpy` / `pandas` / `PIL`**
The dependencies aren't installed, or the virtual environment isn't active.
Activate it and run `pip install -r requirements.txt`. If you launched by
**double-clicking** `qc_gui.py`, that uses your main Python rather than the
virtual environment — install the dependencies into your main Python by running
`pip install -r requirements.txt` without creating/activating a `.venv`.

**`CSV not found`**
Pass a path to an existing split CSV (e.g. `csv/splits/MR1_L_afdx.csv`). If the
`csv/` folder doesn't exist, generate it first — see
[Preparing the data](#preparing-the-data).

**`Missing raw image: …`**
The CSV references a PNG that isn't present. Make sure the `raw/` and `mask/`
folders are next to the `csv/` folder, or pass `--root` to point at the
directory that contains them.

**The window doesn't open on a remote server**
The GUI needs a graphical display. Run it on a local machine, or over SSH with
X-forwarding (`ssh -X`), or through a remote-desktop/VNC session.

**A ligand is too faint to see**
Click **Auto** or narrow the **Contrast min/max** window.

**`python` runs the wrong version**
On some systems `python` is Python 2. Use `python3` (and `python3 -m venv`) for
everything, or activate the virtual environment created above.

---

## How it works

The raw image and the mask are held in memory in their **original (native)**
orientation. Flips and rotations are applied only to what is **displayed**; the
flip/rotation you choose is recorded as metadata in the results CSV. When you
paint, the brush position is mapped back through the current transform before it
is written into the mask, so the stored mask always lines up with the raw PNG on
disk regardless of how the section was rotated or flipped on screen. A separate
downstream step can then apply the recorded flip/rotation to both the image and
the mask when building final outputs. Contrast and opacity are purely
view-side and never alter saved data.

---

## License

No license file is included yet. If you are publishing this repository, add a
`LICENSE` file (for example, MIT or BSD-3-Clause) so others know how they may use
the code.
