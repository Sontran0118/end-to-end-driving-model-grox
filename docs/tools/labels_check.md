# `tools/labels_check.py`

## Purpose and place

Reviews saved labels without changing them. It compares OCC voxel projections with the earlier LiDAR Version-B projection and makes top-down, side, dot, and square overlays.

## Inputs and outputs

It reads saved NPZ labels, `labels_manifest.parquet`, local calibration/LiDAR/egomotion/camera members, and rectified images. It writes review images under `outputs/labels_check/`.

## Run

```powershell
python tools/labels_check.py
```

There are no CLI arguments.

## Walkthrough

`project_rectified_points` applies inverse camera extrinsics and the fixed K. `motion_compensated_lidar` reproduces Version B. `overlay` loads each saved grid, applies viewer `surface_mask`, computes centers, and renders dot and square versions. `alpha_save` blends the layer once over `photo.copy()` and reports mean absolute difference. `plot_views` creates grid views.

## Math example

For voxel index `(3,4,5)`, center is `origin+(idx+0.5)*voxel=(1.4,-18.2,1.2)` with the current constants. Projection then uses `u=fx*x/z+cx`, `v=fy*y/z+cy` after camera transformation.

## Constants and conventions

Output geometry is 1280x384 with `fx=fy≈369.504`, `cx=640`, `cy=192`. Three checkpoint clip/spin pairs are hard-coded. Points with invalid depth or image bounds are rejected.

## Checks and failure modes

The script prints surface count, camera depth and pixel ranges, inside counts, principal-point clustering, and dot/square blend differences. The earlier black-image bug was repeated blending of a mostly-black layer; the current path blends once. Negative or extreme projected depths indicate calibration/time/grid-frame problems, not necessarily an alpha bug.

## Explain it back

1. **What is a voxel center?** `origin+(idx+0.5)*voxel`.
2. **What is compared first?** Original motion-compensated LiDAR against the verified helper.
3. **What is a surface voxel?** An OCC voxel selected by viewer face-neighbour logic.
4. **Why blend once?** Repeated whole-frame blending darkens the photo.
5. **Does this modify NPZ labels?** No.
