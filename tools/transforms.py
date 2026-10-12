"""Small, explicit SE(3) helpers for PhysicalAI-AV calibration and egomotion.

Conventions taken from the NVIDIA PhysicalAI-AV wiki and installed devkit:

* The rig frame is right-handed: +X forward, +Y left, +Z up.  This matches
  the teammate's Grid convention.  The public wiki does not specify whether
  the rig origin is exactly at the rear axle, bumper, or a stated height above
  the road; this module therefore does not invent that physical reference.
* Camera rays use OpenCV axes: +X right, +Y down, +Z forward.  This is the
  same convention used by rectify_camera.py's pinhole ray grid.
* LiDAR points are Cartesian coordinates in the LiDAR sensor frame.
* Calibration and egomotion quaternions are in qx, qy, qz, qw order and are
  interpreted with scipy's standard active rotation convention.
* Sensor extrinsics provide the sensor pose in the rig frame, so the matrix
  constructed here is T_ego_from_sensor and maps a sensor point into the rig.
* Egomotion poses map the ego-rig frame at time t into the clip-local anchor
  frame: T_anchor_from_ego(t).  Consequently, the relative motion requested
  as T_ego(t-k)_from_ego(t) is inv(T_anchor_from_ego(t-k)) @
  T_anchor_from_ego(t).

Translations are metres and timestamps are integer microseconds.  Egomotion
interpolation is linear in translation and spherical in rotation; timestamps
outside the supplied clip range raise ValueError (no extrapolation).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


def _matrix(rotation: Rotation, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.as_matrix()
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def sensor_to_ego_transform(row: pd.Series | dict[str, float]) -> np.ndarray:
    """Return ``T_ego_from_sensor`` from one offline extrinsics row."""
    q = np.asarray([row[k] for k in ("qx", "qy", "qz", "qw")], dtype=float)
    t = np.asarray([row[k] for k in ("x", "y", "z")], dtype=float)
    return _matrix(Rotation.from_quat(q), t)


def camera_to_ego_transform(row: pd.Series | dict[str, float]) -> np.ndarray:
    """Return ``T_ego_from_camera`` for an OpenCV camera frame."""
    return sensor_to_ego_transform(row)


def lidar_to_ego_transform(row: pd.Series | dict[str, float]) -> np.ndarray:
    """Return ``T_ego_from_lidar`` for a Cartesian LiDAR sensor frame."""
    return sensor_to_ego_transform(row)


def _pose_from_row(row: pd.Series) -> np.ndarray:
    q = np.asarray([row[k] for k in ("qx", "qy", "qz", "qw")], dtype=float)
    t = np.asarray([row[k] for k in ("x", "y", "z")], dtype=float)
    return _matrix(Rotation.from_quat(q), t)


def interpolate_anchor_from_ego(egomotion: pd.DataFrame, timestamp_us: int) -> np.ndarray:
    """Return interpolated ``T_anchor_from_ego(timestamp_us)``.

    The first and last timestamps are inclusive; no extrapolation is allowed.
    """
    required = {"timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"}
    missing = required - set(egomotion.columns)
    if missing:
        raise ValueError(f"egomotion is missing columns: {sorted(missing)}")
    df = egomotion.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = df["timestamp"].to_numpy(dtype=np.float64)
    t = float(timestamp_us)
    if t < times[0] or t > times[-1]:
        raise ValueError(f"timestamp {timestamp_us} is outside [{int(times[0])}, {int(times[-1])}]")
    exact = np.flatnonzero(times == t)
    if exact.size:
        return _pose_from_row(df.iloc[int(exact[0])])
    rotations = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy(float))
    rotation = Slerp(times, rotations)([t])[0]
    translation = np.array([
        np.interp(t, times, df[column].to_numpy(float)) for column in ("x", "y", "z")
    ])
    return _matrix(rotation, translation)


def ego_t_to_ego_t_minus_k(
    egomotion: pd.DataFrame, timestamp_us: int, k_us: int
) -> np.ndarray:
    """Return ``T_ego(t-k)_from_ego(t)`` without extrapolating."""
    if k_us < 0:
        raise ValueError("k_us must be non-negative")
    current = interpolate_anchor_from_ego(egomotion, timestamp_us)
    previous = interpolate_anchor_from_ego(egomotion, timestamp_us - k_us)
    return np.linalg.inv(previous) @ current


def read_zip_parquet_member(zip_path: str | Path, member_name: str) -> pd.DataFrame:
    """Read one parquet member from a local devkit-style ZIP, without extraction."""
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as member:
            return pd.read_parquet(io.BytesIO(member.read()))


def read_clip_egomotion(zip_path: str | Path, clip_id: str) -> pd.DataFrame:
    """Read one ``{clip_id}.egomotion.offline.parquet`` member from a ZIP."""
    member = f"{clip_id}.egomotion.offline.parquet"
    return read_zip_parquet_member(zip_path, member)


def read_sensor_extrinsics(parquet_path: str | Path, clip_id: str) -> pd.DataFrame:
    """Read only one clip's rows from a local offline calibration parquet."""
    return pd.read_parquet(parquet_path).loc[clip_id]

