"""Create A/B LiDAR overlays for a small alignment checkpoint.

This script reads individual members from the existing local ZIPs.  It never
extracts a whole archive and it never modifies transforms.py or rectified JPEGs.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import DracoPy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from physical_ai_av.dataset import Features
from transforms import camera_to_ego_transform, interpolate_anchor_from_ego, lidar_to_ego_transform


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "overlays"
CLIPS = ["13ae70ee", "49eb6e65", "0c3e9b08"]
K = np.array([[369.5041722813606, 0.0, 640.0], [0.0, 369.5041722813606, 192.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1280, 384)


def load_features() -> Features:
    table = pd.read_csv(ROOT / "data" / "metadata" / "features.csv", index_col="feature")
    table["clip_files_in_zip"] = table["clip_files_in_zip"].map(json.loads, na_action="ignore")
    return Features(table)


def local_path(features: Features, chunk: int, feature: str) -> Path:
    return RAW / Path(features.get_chunk_feature_filename(chunk, feature))


def read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(zf.read(member)))


def decode_spin(encoded: bytes) -> tuple[np.ndarray, np.ndarray]:
    cloud = DracoPy.decode(encoded)
    points = np.asarray(cloud.points, dtype=np.float64)
    timestamp = np.asarray(cloud.get_attribute_by_name("timestamp")["data"]).reshape(-1).astype(np.int64)
    # DracoPy has already decoded the stored timestamp attribute to the
    # clip-relative microsecond clock; do not subtract the wiki encoding
    # midpoint a second time.
    return points, timestamp


def select_keyframes(clip_id: str, lidar: pd.DataFrame, egomotion: pd.DataFrame) -> list[tuple[int, float]]:
    eligible = pd.read_parquet(PROCESSED / "keyframes.parquet")
    eligible = eligible[eligible.clip_id.str.startswith(clip_id)].copy()
    ts = lidar["spin_start_timestamp"].to_numpy(np.float64)
    positions = egomotion.sort_values("timestamp")[["timestamp", "x", "y", "z"]]
    et = positions.timestamp.to_numpy(np.float64)
    ep = positions[["x", "y", "z"]].to_numpy(np.float64)
    speed = np.linalg.norm(np.gradient(ep, et, axis=0), axis=1) * 1_000_000.0
    speeds = np.interp(eligible.reference_timestamp.to_numpy(np.float64), et, speed)
    choices = []
    low = int(np.argmin(speeds))
    fast = int(np.argmax(speeds))
    median = float(np.median(speeds))
    medium = int(np.argmin(np.abs(speeds - median)))
    for label, index in (("low", low), ("medium", medium), ("fast", fast)):
        choices.append((int(eligible.iloc[index].spin_index), float(speeds[index])))
        print(json.dumps({"clip": clip_id, "speed_class": label, "spin_index": choices[-1][0], "speed_mps": choices[-1][1]}))
    return choices


def vectorized_anchor_poses(egomotion: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    df = egomotion.sort_values("timestamp").reset_index(drop=True)
    times = df.timestamp.to_numpy(np.float64)
    query = np.asarray(timestamps, dtype=np.float64)
    if np.any(query < times[0]) or np.any(query > times[-1]):
        raise ValueError("point timestamp is outside egomotion range; refusing extrapolation")
    rotations = Slerp(times, Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy(float)))(query)
    out = np.repeat(np.eye(4)[None, :, :], len(query), axis=0)
    out[:, :3, :3] = rotations.as_matrix()
    for j, column in enumerate(("x", "y", "z")):
        out[:, j, 3] = np.interp(query, times, df[column].to_numpy(float))
    return out


def project(points_camera: np.ndarray, source: dict) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    front = (z >= 2.0) & (z <= 80.0)
    points = points_camera[front]
    z = points[:, 2]
    rays = points / z[:, None]
    angle = np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2])
    radius = np.polynomial.polynomial.polyval(angle, np.asarray(source["angle_to_pixeldist_poly"]))
    norm = np.hypot(rays[:, 0], rays[:, 1])
    scale = np.divide(radius, norm, out=np.zeros_like(radius), where=norm > 1e-12)
    source_xy = np.column_stack((source["principal_point"][0] + rays[:, 0] * scale, source["principal_point"][1] + rays[:, 1] * scale))
    # Rectified pinhole image uses the common HFOV=120 K from Phase 5.
    uv = np.column_stack((K[0, 0] * rays[:, 0] + K[0, 2], K[1, 1] * rays[:, 1] + K[1, 2]))
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_SIZE[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_SIZE[1])
    return uv[inside], points[inside, 2]


def draw_overlay(image_path: Path, uv: np.ndarray, depths: np.ndarray, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    t = np.clip((depths - 2.0) / 78.0, 0.0, 1.0)
    # Compact blue-to-green-to-red depth coloring; BGR order for OpenCV.
    colors = np.column_stack((255.0 * (1.0 - t), 255.0 * (1.0 - np.abs(2.0 * t - 1.0)), 255.0 * t)).astype(np.uint8)
    for (u, v), color in zip(uv.astype(int), colors):
        cv2.circle(image, (int(u), int(v)), 2, tuple(int(x) for x in color), -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write {output_path}")


def process_clip(clip_row: pd.Series, features: Features) -> None:
    clip_id, chunk = str(clip_row.clip_id), int(clip_row.chunk)
    lidar_zip = local_path(features, chunk, "lidar_top_360fov")
    camera_zip = local_path(features, chunk, "camera_front_wide_120fov")
    lidar_cal = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
    camera_cal = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
    lidar_to_rig = lidar_to_ego_transform(lidar_cal.loc["lidar_top_360fov"])
    camera_to_rig = camera_to_ego_transform(camera_cal.loc["camera_front_wide_120fov"])
    camera_from_rig = np.linalg.inv(camera_to_rig)
    source = json.loads((PROCESSED / "K" / f"{clip_id}.json").read_text(encoding="utf-8"))["source_intrinsics"]
    egomotion_zip = local_path(features, chunk, "egomotion.offline")
    egomotion_member = features.get_clip_files_in_zip(clip_id, "egomotion.offline")["egomotion.offline"]
    with zipfile.ZipFile(lidar_zip) as lz, zipfile.ZipFile(camera_zip) as cz, zipfile.ZipFile(egomotion_zip) as ez:
        lidar_member = features.get_clip_files_in_zip(clip_id, "lidar_top_360fov")["pointclouds"]
        camera_member = features.get_clip_files_in_zip(clip_id, "camera_front_wide_120fov")["frame_timestamps"]
        ego = read_member(ez, egomotion_member)
        lidar = read_member(lz, lidar_member).sort_values("spin_start_timestamp").reset_index(drop=True)
        camera_ts = read_member(cz, camera_member).timestamp.to_numpy(np.int64)
        choices = select_keyframes(clip_id, lidar, ego)
        for spin_index, speed in choices:
            row = lidar[lidar.spin_index == spin_index].iloc[0]
            reference = int(row.spin_start_timestamp)
            frame_index = int(np.argmin(np.abs(camera_ts - reference)))
            image_path = PROCESSED / "images" / clip_id / f"after_{frame_index:04d}.jpg"
            points, relative_us = decode_spin(row.draco_encoded_pointcloud)
            point_times = relative_us
            rig_points = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
            center_mask = (rig_points[:, 0] >= 5) & (rig_points[:, 0] <= 25) & (np.abs(rig_points[:, 1]) <= 1.5)
            ground_z = float(np.median(rig_points[center_mask, 2])) if np.any(center_mask) else None
            for version in ("A", "B"):
                transformed = rig_points
                valid_time = np.ones(len(points), dtype=bool)
                if version == "B":
                    camera_time = int(camera_ts[frame_index])
                    valid_time = (point_times >= int(ego.timestamp.min())) & (point_times <= int(ego.timestamp.max()))
                    point_poses = vectorized_anchor_poses(ego, point_times[valid_time])
                    camera_pose = interpolate_anchor_from_ego(ego, camera_time)
                    homogeneous = np.column_stack((rig_points[valid_time], np.ones(valid_time.sum())))
                    relative = np.linalg.inv(camera_pose) @ point_poses
                    transformed = np.einsum("nij,nj->ni", relative, homogeneous)[:, :3]
                else:
                    transformed = rig_points
                camera_points = (camera_from_rig @ np.column_stack((transformed, np.ones(len(transformed)))).T).T[:, :3]
                uv, depths = project(camera_points, source)
                output = OUTPUT / f"{clip_id}_{spin_index:03d}_{version}.jpg"
                draw_overlay(image_path, uv, depths, output)
                in_front = (camera_points[:, 2] >= 2.0) & (camera_points[:, 2] <= 80.0)
                print(json.dumps({"clip": clip_id, "spin_index": spin_index, "speed_mps": speed, "version": version, "points_total": len(points), "points_in_front": int(in_front.sum()), "points_in_image": len(uv), "median_rig_z_centerline": ground_z, "camera_xyz_rig": camera_to_rig[:3, 3].tolist(), "lidar_xyz_rig": lidar_to_rig[:3, 3].tolist(), "output": str(output.relative_to(ROOT))}))


def main() -> None:
    features = load_features()
    clips = pd.read_csv(ROOT / "clips.csv")
    selected = clips[clips.clip_id.str[:8].isin(CLIPS)]
    if len(selected) != len(CLIPS):
        raise ValueError("requested clip prefixes were not all found")
    for _, row in selected.iterrows():
        process_clip(row, features)


if __name__ == "__main__":
    main()
