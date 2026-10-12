# `tools/build_labels.py`

## Purpose and place

Builds sparse OCC labels from motion-compensated LiDAR. It is the final label-building stage after keyframe matching, calibration, and the vehicle-box proposal.

## Inputs and outputs

Inputs are `clips.csv`, `data/processed/matches.parquet`, local ZIP/Parquet members under `data/raw/`, and `calibration/vehicle_dimensions/*.parquet`. Outputs are `data/labels/<split>/<clip_id>/<spin>.npz`, `data/labels/labels_manifest.parquet`, and diagnostic plots under `outputs/labels_diagnostics/`.

## Run

```powershell
python tools/build_labels.py
python tools/build_labels.py --clip-prefix 49eb6e65
python tools/build_labels.py --exclude-prefix 49eb6e65
python tools/build_labels.py --plots-only
```

## Walkthrough

`vehicle_dimensions` joins per-clip dimensions to the clip list. `box_extents` expands the box by `OWN_VEHICLE_MARGIN`; `box_masks` returns the inside and 0.5 m shell masks. `transform_points_version_b` reuses the Version-B timing method. `voxelize` floors coordinates, drops out-of-grid points, deduplicates indices, and marks OCC. `save_label` constructs `Grid` before writing the compressed NPZ.

## Math example

With origin `(0,-20,-1)`, voxel `0.4`, and point `(10,-19.6,-0.6)`, the index is `floor((10,0.4,0.4)/0.4)=(25,1,1)`. A second point in the same cell still produces one OCC voxel, although the manifest point count includes both valid returns.

## Constants and conventions

`VOXEL=0.4`, `SHAPE=(200,100,16)`, `ORIGIN=(0,-20,-1)`, `OWN_VEHICLE_MARGIN=0.20`, and `SHELL_METRES=0.5`. The grid is at camera-frame time. Rig axes are X forward, Y left, Z up. `pose` is `T_anchor_from_rig` at camera time. The exact physical rig origin is **UNVERIFIED**.

## Checks and failure modes

`synthetic_tests` checks wall indices, boundaries, voxel centers, vehicle filtering, Grid construction, and NPZ round-trip. A missing dimension, ZIP member, ego timestamp range, or required match column raises an error. A stale manifest column can break `--plots-only`: the current code expects `n_occ`, while the saved manifest uses `n_points_in_grid` and `n_occ_voxels`.

## Explain it back

1. **What is removed?** Points inside the per-clip expanded vehicle box.
2. **What is marked OCC?** Every unique in-grid voxel containing a retained point.
3. **Is FREE ray casting performed?** No; untouched cells remain UNKNOWN.
4. **Why construct Grid first?** Its validation runs before serialization.
5. **What does the manifest’s old `n_occ` count?** Valid in-grid point returns before voxel deduplication; the clarified manifest calls it `n_points_in_grid`.
