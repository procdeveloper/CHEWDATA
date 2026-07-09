# measurement_ocr

Standalone, cross-platform Python tool that extracts numeric readings from a
display (multimeter, scale, gauge, thermometer, etc.) filmed with a
handheld camera. Single file: `measurement_ocr.py`. OpenCV GUI (no Qt/Tk),
Tesseract OCR by default with an optional GPU-capable EasyOCR backend.

Read `README.md` first for user-facing usage/flags/keybindings — this file
is about *why* things are built the way they are, for whoever (human or
Claude) works on the internals next.

## Core architecture

**Don't stabilize the whole frame — track the display panel directly.**
`ReferenceTracker` matches every incoming frame against ONE fixed reference
frame (picked when the user clicks the display's 4 corners) using ORB
features + RANSAC homography. Matching every frame against a *fixed*
reference (not frame-to-frame) means error never accumulates — this one
homography simultaneously cancels hand-shake and de-skews the panel into a
flat rectangle ("Rectified" window). This was a deliberate choice over
generic video stabilization (e.g. `vidstab`, feature-tracking-and-smooth):
we only care about one planar region, so registering directly to it is
both simpler and more accurate than stabilizing the whole scene.

**Reference rebasing (for slow drift).** A fixed reference eventually runs
out of matchable viewpoint range if the camera keeps drifting. The tracker
watches its own match quality — judged *relative to this scene's own
recent peak inlier count* (`peak_inliers`, decaying at `PEAK_DECAY`), not a
fixed number — and quietly adopts a fresh reference frame when quality is
trending weak (`--rebase-after-weak`) or right after recovering from a
tracking loss. A rebase changes only what ORB matches against; canonical
output coordinates stay perfectly continuous across it (see `_rebase`: the
new reference's `ref_to_canonical` is just the current `H_cur_to_canonical`
at the moment of rebase). Verified via synthetic out-of-plane-rotation
tests that this measurably extends tracked range — but it's a bounded
improvement, not a fix for everything: if the camera drifts toward a
near-edge-on view, no rebasing scheme avoids the fact that 2D feature
matching runs out of usable perspective. Don't oversell this if asked.

**Known bug already found and fixed — watch for regressions here.** A
borderline RANSAC fit (inlier count just above the hard-fail threshold)
can occasionally produce a wildly wrong homography that still nominally
"passes." Caught this via a synthetic perspective-drift stress test: a
frame that should have failed instead returned `tracked_ok=True` with
~300px of positional error. Fixed with `_quad_plausible()` — rejects
non-convex quads and quads whose area jumped outside
`[MIN_AREA_RATIO, MAX_AREA_RATIO]` of the last accepted frame. Also added
`MIN_INLIER_RATIO` (inliers/good-matches) since raw inlier count alone
isn't a reliable trust signal at oblique angles. If you touch
`ReferenceTracker.register()`, re-run the plausibility regression test
(see Testing below) — it's easy to silently reintroduce this.

**Blur handling — two independent mechanisms, don't conflate them:**
- *Sharpness gating* (`sharpness_score` = variance of Laplacian,
  `pick_best_frame`): OCR runs on the sharpest frame in a short rolling
  window instead of whatever landed on the sampling interval.
- *Multi-frame stacking* (`stacked_frame`, `--stack-frames`): frames are
  already pixel-aligned (same canonical coords), so median-stacking a
  window needs no extra registration. This fights blur/noise that *varies*
  frame to frame (typical hand shake). It cannot fix a lens that's
  genuinely out of focus every frame — that blur is identical across the
  whole stack and survives the median untouched. Say this plainly if asked
  whether stacking "fixes blur" — it doesn't, for that case.

**UI layer (toolbar + stats HUD) is intentionally rudimentary**, per an
explicit user ask, not a design ambition — don't over-build this further
unprompted. Buttons and keyboard shortcuts funnel through one shared
`apply_action(action, ctx)` dispatcher so the two input paths can't drift
out of sync; if you add a new action, add it to *both*
`COMMAND_KEYS` and `BUTTON_TO_ACTION` pointing at the same action string,
never duplicate logic per-input-method. The rectified crop is often small,
so display-only upscaling (`compute_display_params`, capped at 3x) is
applied purely for on-screen legibility — mouse coordinates are converted
back through that scale factor in `make_composite_mouse_callback` before
being stored, so field-rect coordinates are always in true canonical
space regardless of display zoom. Don't let OCR ever read from the
upscaled display image — it must stay on the canonical-resolution
`rectified`/`ocr_source`.

## File layout

- `measurement_ocr.py` — everything (geometry helpers → `ReferenceTracker`
  → OCR backends → blur handling → drawing/UI → action dispatch → `main()`)
- `README.md` — user-facing docs (install, flags, keybindings)
- `requirements.txt` — `opencv-python`, `numpy`, `pytesseract` (+ commented
  optional `easyocr`); Tesseract's actual binary is a separate OS install,
  not pip-installable

## Testing approach (no GUI in a sandbox — do this, not manual clicking)

There's no display/X server available for interactive testing, so
everything was validated with synthetic scenes + headless assertions
rather than running the actual `cv2.imshow` loop:
- Build a textured synthetic "device" image (random circles + text +
  a light LCD-style panel with dark digits — realistic contrast matters,
  a low-contrast test scene will make Tesseract fail for reasons that have
  nothing to do with the code).
- Warp it with a *known* homography (shake: random perspective jitter;
  slow drift: `out_of_plane_homography()` — a proper pinhole-camera
  Y-axis-rotation homography, NOT `cv2.getRotationMatrix2D`, since in-plane
  rotation is something ORB is already very robust to and won't stress the
  tracker the way real viewpoint drift does).
- Feed the warped frame to `tracker.register()` and compare the returned
  quad against the ground-truth quad (computed by transforming the known
  original quad through the same known homography). Sub-pixel error is the
  bar for "shake" tests; a few px is fine for extreme synthetic drift tests.
- For OCR, actually call `pytesseract` (it's installed in this sandbox: the
  binary is at `/usr/bin/tesseract`) — don't just check that preprocessing
  ran without exceptions; verify it reads back something close to the
  known ground-truth string.
- Real caveat found during this process: repeatedly re-warping ONE flat
  reference image to simulate progressively more extreme drift is an
  artificially pessimistic test (each rewarp compounds foreshortening
  blur that a real camera wouldn't accumulate, since a real camera would
  reveal fresh native-resolution detail at each new angle instead). Good
  for finding bugs in the worst case; don't quote its exact numeric
  results as real-world performance guarantees.

Always run `python3 -m py_compile measurement_ocr.py` before anything else
after an edit — this file is large enough that a stray edit in one
function silently breaking an unrelated one is a real risk.

## Things NOT to change without discussion

- Don't switch the tracking approach to frame-to-frame chaining "for
  simplicity" — that's the drift problem the fixed-reference design
  specifically avoids.
- Don't remove the `_quad_plausible` check to "simplify" `register()` —
  see the bug note above.
- Don't let stacking/sharpness-gating logic run on the *display* image
  (post-upscale) — it must operate on canonical-resolution frames only.
