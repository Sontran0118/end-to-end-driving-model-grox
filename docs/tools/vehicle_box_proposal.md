# `tools/vehicle_box_proposal.py`

## Purpose and place

Proposes and checks the own-vehicle exclusion box before occupancy labels are built. It is a calibration/data-quality diagnostic, not the label writer.

## Inputs and outputs

It reads `clips.csv`, vehicle-dimension Parquets, LiDAR ZIP members, and sensor extrinsics. It writes three zoomed proposal plots under `outputs/vehicle_box_proposal/` and prints per-plot and all-keyframe inside/shell counts.

## Run

```powershell
python tools/vehicle_box_proposal.py
```

There are no CLI arguments.

## Walkthrough

`load_dimensions` joins dimensions by clip. `box_extents` creates an axis-aligned box from `rear_axle_to_bbox_center`, length, width, and height. `load_clip_spin` decodes a spin and applies static LiDAR-to-rig extrinsics. `make_plot` marks inside points orange, other points blue, and draws the box, sensors, origin, and x=0. `all_keyframe_counts` repeats the masks over all keyframes.

## Math example

For length `L` and center `c`, x extents are `c-L/2-0.2` through `c+L/2+0.2`; y extents are `-W/2-0.2` through `W/2+0.2`; z runs from `-0.2` through `H+0.2`.

## Constants and conventions

`MARGIN_M=0.20`, top-down x range `-4..10`, y range `-5..5`, side-view z range `-2..4`, and the three checkpoint clips are constants. Rig axes are X forward, Y left, Z up. Physical origin/box interpretation is **UNVERIFIED** beyond the fields used by the code.

## Checks and failure modes

The script prints total, inside, shell, and exact extents. Recorded proposal counts were 622/0, 670/34, and 942/0 for the three checkpoint plots; all-keyframe medians/maxima were 798/1660 inside and 0/5584 shell. Missing dimensions, member paths, or extrinsics stop processing. It does not apply motion compensation for the proposal counts.

## Explain it back

1. **Why make this proposal?** To avoid labeling the recording vehicle as scene OCC.
2. **What is the margin?** 0.20 m on every side.
3. **What is the shell?** Non-inside points within 0.5 m of the box.
4. **How are points colored?** Orange inside, blue outside.
5. **Are these final labels?** No; they are proposal diagnostics.
