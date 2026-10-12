# `tools/transforms.py`

## Purpose and place

Provides the shared rigid-transform and pose-interpolation helpers used by alignment and label code. It is a library module, not a command-line pipeline stage.

## Inputs and outputs

It accepts calibration/egomotion rows or Parquet members and returns 4x4 NumPy transforms, interpolated poses, or relative ego transforms. It does not write files by itself.

## Run

There is no CLI. Run its unit tests with:

```powershell
python tools/test_transforms.py
```

## Walkthrough

`_matrix` builds a homogeneous matrix. `sensor_to_ego_transform`, `camera_to_ego_transform`, and `lidar_to_ego_transform` adapt sensor rows. `_pose_from_row` reads an egomotion pose. `interpolate_anchor_from_ego` performs exact lookup or translation interpolation plus rotation slerp. `ego_t_to_ego_t_minus_k` forms a relative transform. ZIP Parquet readers provide local data access.

## Math example

For a sensor point `p`, `T_ego_from_sensor @ [p,1]` gives rig coordinates. For poses `P_old` and `P_now` into one anchor, the transform from current rig coordinates to old-rig coordinates is `inv(P_old) @ P_now`.

## Constants and conventions

Axes are FLU: X forward, Y left, Z up. Camera axes are OpenCV X right, Y down, Z forward. Quaternion order is `(qx,qy,qz,qw)` and SciPy uses active rotations. Timestamps are integer microseconds; interpolation does not extrapolate. Exact physical rig-origin placement is **UNVERIFIED**.

## Checks and failure modes

`test_transforms.py` checks inverse identity, rotation orthonormality/determinant, known points, exact samples, and midpoint interpolation. Missing columns, unsorted/duplicate timestamps, or requests outside the pose interval raise errors.

## Explain it back

1. **What does the name mean?** `T_target_from_source` maps source coordinates into target coordinates.
2. **What do extrinsics map?** Sensor frame to rig frame.
3. **What do egomotion rows map?** Rig/ego frame to local anchor.
4. **How is rotation interpolated?** Spherical linear interpolation.
5. **What happens outside the clip?** The helper refuses extrapolation.
