# `tools/show_sample.py`

## Purpose and place

Creates one review figure combining a keyframe, its three history images, a voxel-dot projection, and two occupancy-grid views. It is a read-only debugging tool after labels, matches, rectified images, and calibration already exist.

## Inputs and outputs

It reads `clips.csv`, `keyframes.parquet`, `matches.parquet`, the selected NPZ under `data/labels/`, the four rectified JPEGs, offline egomotion/extrinsics, and vehicle-dimension Parquets. It writes one PNG to `outputs/samples/<prefix>_spin<spin>.png` and prints the same metadata as JSON.

## Run

```powershell
python tools/show_sample.py 49eb6e65 15
```

The prefix must identify exactly one clip and the spin must be a saved keyframe. There is no `--all` mode.

## Walkthrough

`_clip_row` resolves the prefix. `_dimensions` loads per-clip vehicle fields. `_ego_speed` reproduces the position-gradient speed calculation. `_load_grid` constructs `Grid`. `_voxel_overlay` applies `surface_mask`, computes voxel centers, projects them with `project_rectified_points`, and blends 2-pixel height-colored dots over the keyframe. `make_sample` orders the keyframe and `HISTORY_OFFSETS`, loads matched `after_*.jpg` files, draws the four panels, grid views, and text box.

## Math example

For voxel index `(3,4,5)`, its center is `origin+(idx+0.5)*voxel`. With the current configuration this is `(1.4,-18.2,1.2)` before the camera transform and pinhole projection.

## Constants and conventions

The script imports `VOXEL`, `SHAPE`, `ORIGIN`, `OWN_VEHICLE_MARGIN`, `HFOV_DEGREES`, K, and `HISTORY_OFFSETS` from existing modules, so the displayed configuration follows those modules. It uses offline camera extrinsics and the fixed rectified K. The physical meaning of the rig origin and camera timestamp phase remain **UNVERIFIED** here.

## Checks and failure modes

The recorded run for `49eb6e65` spin 15 printed split `test`, speed `0.018341... m/s`, and `n_occ_voxels=2498`, and wrote the sample PNG. Missing/ambiguous clips, keyframes, matches, dimensions, NPZs, calibration, or images raise errors. The script does not recompute labels or judge alignment.

## Explain it back

1. **How are history images chosen?** From the matched rows for offsets `[-5,-10,-15]`.
2. **What does the top-down view show?** The maximum OCC projection over Z.
3. **What does the side view show?** The maximum OCC projection over Y.
4. **How are voxel dots colored?** By voxel-center height using `viridis`.
5. **Does it change saved labels?** No; it only reads them and writes a review PNG.
