# `tools/timing_check.py`

## Purpose and place

Creates one focused A/B timing review for spin 130 of clip `0c3e9b08`. It is a targeted diagnostic after the general alignment overlays.

## Inputs and outputs

It reads local LiDAR, camera, and egomotion ZIP members, extrinsics, camera intrinsics, and the matched rectified frame. It writes `outputs/timing_check/0c3e9b08_spin130_sign_4x.jpg` and prints JSON describing the crop and counts.

## Run

```powershell
python tools/timing_check.py
```

There are no CLI arguments.

## Walkthrough

The script loads spin 130, transforms points into rig coordinates, creates Version A in the camera frame, and creates Version B with per-point pose correction. It keeps points 10–40 m away, projects with the fixed K, draws one-pixel depth-colored dots, crops the configured region, enlarges it 4x, and labels plain/A/B panels.

## Math example

For a point at camera `(1,0,20)`, the projected horizontal pixel is about `369.504/20+640≈658.5`. Only points whose projected pixels lie in the crop contribute to that panel.

## Constants and conventions

`PREFIX`, `SPIN`, crop `(700,100,170,120)`, depth range 10–40 m, 4x scale, one-pixel dots, and the fixed K are defined at the top. The printed structure label is the recorded description “green roadside freeway sign reading Watsonville”; object presence is **UNVERIFIED** by code.

## Checks and failure modes

Missing ZIP members, calibration, or the matched image stops processing. The script reports panel/dot counts but does not judge whether timing correction improves alignment.

## Explain it back

1. **Which spin is checked?** `0c3e9b08` spin 130.
2. **What are A and B?** Static and motion-compensated projections.
3. **What depth range is used?** 10–40 m.
4. **Why enlarge the crop?** To make timing differences easier to inspect.
5. **What output is written?** One three-panel JPEG.
