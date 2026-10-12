# `tools/lidar_alignment.py`

## Purpose and place

Creates review overlays from LiDAR and rectified front-wide images for three checkpoint clips. It is an alignment diagnostic between calibration, time matching, and rectification.

## Inputs and outputs

It reads local LiDAR/camera/egomotion ZIP members, offline extrinsics and camera intrinsics, `keyframes.parquet`, and rectified images. It writes 18 JPEGs under `outputs/overlays/` and prints point counts, speed choices, sensor positions, and diagnostic statistics.

## Run

```powershell
python tools/lidar_alignment.py
```

There are no CLI arguments.

## Walkthrough

`decode_spin` decodes Draco points and preserves their clip-relative timestamps. `select_keyframes` chooses low, middle, and fast ego-speed examples. `vectorized_anchor_poses` interpolates poses. `project` applies LiDAR-to-rig, inverse camera extrinsics, and the rectified K, keeping camera Z 2–80 m and image bounds. `draw_overlay` colors dots by depth. Version A is static; Version B compensates each point to camera time.

## Math example

Given a rig point transformed to camera coordinates with `z=20`, projection is `u=fx*x/z+cx`, `v=fy*y/z+cy`; points outside the image are discarded. Version B additionally applies the relative pose from each point time to camera time.

## Constants and conventions

The helper uses `K≈[[369.504,0,640],[0,369.504,192],[0,0,1]]`, 1280x384, Z range 2–80 m, and 2-pixel dots. It uses the three clip prefixes and selected spin choices defined in the file. The code does not judge alignment.

## Checks and failure modes

It prints total, in-front, and in-image counts, centerline statistics, and sensor positions. Missing local members, calibration, images, or pose coverage stops the run. A visually poor overlay can arise from timestamp offset, transform direction, quaternion/order, K crop, motion distortion, or bad calibration; the script does not distinguish them automatically.

## Explain it back

1. **What is Version A?** No time correction.
2. **What is Version B?** Per-point motion compensation to camera time.
3. **Why keep Z positive and bounded?** Behind-camera/far points are not useful for this overlay.
4. **How are dots colored?** By depth.
5. **Does the script certify alignment?** No; it produces review images.
