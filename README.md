# measurement_ocr

Extract numeric readings from a display filmed with a handheld camera
(multimeter, scale, gauge, thermometer, scoreboard, etc). Single Python
file, cross-platform (Windows / macOS / Linux), CPU by default with an
optional GPU OCR backend.

## How the shake/deskew problem is actually solved

Rather than doing generic whole-frame video stabilization, this tracks the
one thing that matters: the display panel itself.

1. On the first frame, you click its 4 corners (or auto-detect).
2. Every later frame is matched **directly against that first frame** using
   ORB features + a RANSAC homography — not frame-to-frame. Matching
   against one fixed reference means there's no drift to accumulate, no
   matter how long the clip runs or how much the camera shakes/rotates/
   moves closer or further, as long as enough of the reference scene stays
   in view.
3. That homography simultaneously **undoes hand-shake and de-skews** the
   panel into a flat, front-on rectangle ("Rectified" window) — deskew and
   antishake fall out of the same computation. A light exponential
   smoothing pass (`--smoothing`) further damps any residual per-frame
   jitter in that homography.
4. If a single frame is too blurry/dark to match (e.g. fast motion), the
   last good pose is held instead of letting the overlay jump or vanish.

## Slow drift and blur

**Slow drift.** Matching every frame against one fixed reference (rather
than frame-to-frame) already means there's no *accumulating* drift. But
there's still a ceiling: ORB matching degrades as the view diverges further
from that original reference, and past a point it fails outright. To push
that ceiling out, the tracker watches its own match quality (judged
relative to *this scene's own* recent best inlier count, not a fixed
number, so it adapts whether the scene is feature-rich or sparse) and
quietly adopts a fresh reference frame — either after several consecutive
weak-but-tracked frames (`--rebase-after-weak`), or right after recovering
from a brief loss. A rebase only changes which image is used for ORB
matching; the canonical output coordinates stay perfectly continuous
across it. Verified this against simulated drift: it clearly extends the
range over a plain fixed reference, but the improvement is bounded — if
the camera keeps drifting toward a near-edge-on view of the display, no
amount of rebasing avoids the fact that a 2D feature matcher eventually
runs out of usable perspective to work with. For the drift ranges a person
actually produces holding a device steady-ish, this should be a
non-issue either way.

**Blur.** Two independent things fight this:
- *Sharpness gating* — frames are scored by variance-of-Laplacian, and OCR
  runs on the sharpest frame in a short rolling window (`--sharpness-window`)
  rather than whatever frame happened to land on the sampling interval.
- *Multi-frame stacking* (`--stack-frames`) — since every buffered frame is
  already warped into the same canonical coordinates, a short window of
  them can be median-stacked with no extra alignment work. This denoises
  and helps with blur that *varies* frame to frame (typical of hand shake).
  It can't do anything for a lens that's genuinely out of focus in every
  frame — that blur is identical across the stack and survives the median.

## Install

```bash
pip install opencv-python numpy pytesseract
```

Tesseract is a wrapper around the real Tesseract OCR engine, so also
install the binary:

- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr` (or your distro's package)

### Optional: GPU backend

```bash
pip install easyocr        # pulls in PyTorch
python measurement_ocr.py video.mp4 --ocr easyocr --gpu
```

EasyOCR needs no external binary and runs on CUDA if `--gpu` is passed and
a GPU/PyTorch-CUDA build is available; otherwise it falls back to CPU.

## Usage

```bash
python measurement_ocr.py video.mp4
python measurement_ocr.py 0                       # live camera 0
python measurement_ocr.py clip.mov --ocr easyocr --gpu --ocr-every 3
```

Two windows open:

- **Original** — the raw video with the tracked display outlined in green
  (red if tracking is temporarily lost), a status line, and a stats HUD in
  the corner (FPS, match/inlier counts, rebase count, current sharpness,
  OCR settings, field count).
- **Rectified** — the flattened, front-on display, with recognized
  text/boxes drawn live on top, and a row of clickable buttons underneath.

### Controls

Every action is available both as a keyboard shortcut and as an on-screen
button in the toolbar strip under the Rectified view — click Pause/Play,
Clear, Log, Re-pick, OCR-/OCR+, or Quit directly if you'd rather not use
the keyboard. Logging also flashes a green on-screen confirmation banner
(not just a terminal message).

| Key | Button | Action |
|---|---|---|
| `space` | Pause/Play | pause / resume |
| click or drag (on **Rectified**) | — | define a numeric field. A quick click drops a default-size box there; a drag lets you size it exactly — handy for multi-value displays (e.g. voltage *and* current) |
| `n` | — | rename the most recently added field (typed in the terminal; no button, since it needs typed text) |
| `c` | Clear | clear all fields |
| `e` | Log | **extract**: log the current reading of every field, with a timestamp, to the CSV |
| `r` | Re-pick | re-pick the 4 corners (reset tracking if the panel moved out of frame or a bad reference was chosen) |
| `+` / `-` | OCR+ / OCR- | OCR more/less often while playing (perf vs. responsiveness) |
| `q` / `Esc` | Quit | quit |

If you never define a field, OCR runs on the whole rectified display by
default — clicking is for isolating individual numbers on multi-value
displays, not required to get started.

### Output

Appends rows to `extracted_data.csv` (or `--output <path>`):

```
timestamp_s,frame_idx,field1,field2
12.480,374,12.34,0.55
```

## Useful flags

| Flag | Meaning |
|---|---|
| `--ocr {tesseract,easyocr}` | OCR backend (default: tesseract) |
| `--gpu` | run easyocr on GPU |
| `--ocr-every N` | run OCR every N frames while playing (default: 5) |
| `--upscale F` | upscale factor before OCR, useful for tiny digits (default: 3.0) |
| `--no-auto-invert` | disable automatic dark/light polarity correction |
| `--max-width N` | downscale incoming frames to this width for speed (default: 1280, 0 = off) |
| `--smoothing F` | 0–0.97, jitter damping on the tracked homography (default: 0.6) |
| `--sharpness-window N` | pick the sharpest of the last N frames before OCR when stacking is off (default: 5) |
| `--stack-frames N` | median-stack the last N aligned frames before OCR; 1 disables stacking (default: 3) |
| `--rebase-after-weak N` | adopt a fresh reference after N consecutive weak-match frames; 0 disables (default: 20) |
| `--no-rebase-on-recovery` | don't adopt a fresh reference right after recovering from a tracking loss |

## Tips for accuracy

- Good, even lighting on the display beats any amount of preprocessing.
- Keep the display reasonably large in frame; tiny digits upscale but
  don't gain detail that isn't there.
- Seven-segment LED/LCD digits sometimes confuse general-purpose OCR
  (e.g. 3 vs. 5, or broken segments read as extra strokes). If accuracy on
  a specific segmented display isn't good enough, a dedicated
  seven-segment recognizer (template-matching each segment) will usually
  outperform Tesseract/EasyOCR — that's a good next step if this is your
  bottleneck, and the `preprocess_for_ocr` step is a natural place to swap
  in that logic.
- If tracking is ever lost for long stretches (dim/low-texture scene, or
  the reference frame itself was blurry), press `r` on a sharp frame to
  re-pick a better reference.

## Files

- `measurement_ocr.py` — the whole program, single file.
- `requirements.txt` — pip dependencies.
