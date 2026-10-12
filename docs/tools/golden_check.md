# `tools/golden_check.py`

## Purpose and place

Recomputes one saved occupancy label from local raw members and checks it against the saved result. It is a regression check for label construction, not a label-generation command.

## Inputs and outputs

It reads `clips.csv`, `matches.parquet`, offline extrinsics, LiDAR and egomotion ZIP members, vehicle dimensions, the saved manifest, and the target NPZ. It writes no files; it prints `PASS` or `FAIL` and exits nonzero on failure.

## Run

```powershell
python tools/golden_check.py
```

The target is fixed in code: clip prefix `49eb6e65`, spin 15.

## Walkthrough

The script resolves the clip and keyframe match, reads one LiDAR and one egomotion member, decodes the spin, and calls the existing `transform_points_version_b`, `box_extents`, `box_masks`, and `voxelize` helpers from `build_labels.py`. It computes the camera-time pose, loads the saved NPZ, and checks OCC, origin, voxel size, pose, format version, clip id, spin index, and manifest point count.

## Math example

The recomputed point path is `LiDAR -> rig -> camera-time rig` using Version B. Voxel indices then use `floor((point-origin)/voxel)`; repeated points in one cell produce one OCC cell but remain part of the in-grid point count.

## Constants and conventions

The test inherits `VOXEL`, `ORIGIN`, and `OCC` from the existing modules. It uses the saved match’s `camera_timestamp` and the per-clip vehicle box. It compares array values exactly; it does **not** compare the compressed NPZ file’s raw bytes, whose compression/container bytes may differ.

## Checks and failure modes

The recorded result is `PASS` for `49eb6e65` spin 15 with `points_in_grid=54388` and `occ_voxels=2498`. Missing members, calibration, manifest rows, or NPZ keys raise errors. Any mismatch prints the field names and exits with status 1.

## Explain it back

1. **Which label is checked?** `49eb6e65` spin 15.
2. **What timing is used?** The saved keyframe’s matched camera timestamp.
3. **Is vehicle filtering included?** Yes, using the existing box helpers.
4. **What does PASS mean?** Required saved arrays and metadata equal the recomputation.
5. **Does it rewrite the NPZ?** No.
