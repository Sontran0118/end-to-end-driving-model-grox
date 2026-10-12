# `tools/overlay_checks.py`

## Purpose and place

Makes focused height-colored overlay checks for selected low-speed and A/B examples. It follows `lidar_alignment.py` and is purely diagnostic.

## Inputs and outputs

It reads local feature ZIP members, extrinsics, camera intrinsics, keyframe/match data, and existing overlay images. It writes images under `outputs/overlays_check/`.

## Run

```powershell
python tools/overlay_checks.py
```

There are no CLI arguments.

## Walkthrough

`clip_context` loads one clip’s members and calibration. `project_height` transforms points to camera coordinates and uses the fixed rectified K. `draw_height` maps height percentiles to colors and draws one-pixel dots. `make_low_checks` creates plain/overlay and crop panels. `make_ab_check` compares the existing A/B images and prints median/max pixel movement.

## Math example

For a camera point `(x,y,z)=(2,0,20)`, `u≈369.504*2/20+640≈677`, while `v=192`; the point is then colored by its rig height.

## Constants and conventions

It uses 1280x384, the fixed K, 1-pixel dots, and the hard-coded checkpoint spins/crops in the file. It keeps height range from the 2nd to 98th percentile for display. It does not make an alignment judgment.

## Checks and failure modes

It prints the 18 existing overlay filenames and recorded speeds, then writes the requested checks. Missing source images or ZIP members stop processing. Crop labels describe intended regions; the code does not prove that a requested object is present.

## Explain it back

1. **What changes from the main overlay?** Dot color represents height, not depth.
2. **Why use a crop?** It makes vertical-edge review easier.
3. **What does A/B compare?** Static versus time-corrected projection on the same spin.
4. **What is printed for A/B?** Median and maximum pixel displacement.
5. **Does it modify labels?** No.
