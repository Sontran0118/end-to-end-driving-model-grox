"""Make one focused timing-check crop without modifying existing images."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import lidar_alignment as la
from transforms import camera_to_ego_transform, interpolate_anchor_from_ego, lidar_to_ego_transform

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "0c3e9b08"
SPIN = 130
CROP = (700, 100, 170, 120)
K = np.array([[369.5041722813606, 0.0, 640.0], [0.0, 369.5041722813606, 192.0], [0.0, 0.0, 1.0]])


def main() -> None:
    features = la.load_features()
    clips = pd.read_csv(ROOT / "clips.csv")
    row = clips[clips.clip_id.str.startswith(PREFIX)].iloc[0]
    chunk = int(row.chunk)
    extrinsics = pd.read_parquet(la.local_path(features, chunk, "sensor_extrinsics.offline")).loc[row.clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    camera_to_rig = camera_to_ego_transform(extrinsics.loc["camera_front_wide_120fov"])
    camera_from_rig = np.linalg.inv(camera_to_rig)
    source = json.loads((ROOT / "data" / "processed" / "K" / f"{row.clip_id}.json").read_text())["source_intrinsics"]
    with zipfile.ZipFile(la.local_path(features, chunk, "lidar_top_360fov")) as lz, zipfile.ZipFile(la.local_path(features, chunk, "camera_front_wide_120fov")) as cz, zipfile.ZipFile(la.local_path(features, chunk, "egomotion.offline")) as ez:
        lidar = la.read_member(lz, features.get_clip_files_in_zip(row.clip_id, "lidar_top_360fov")["pointclouds"])
        camera_ts = la.read_member(cz, features.get_clip_files_in_zip(row.clip_id, "camera_front_wide_120fov")["frame_timestamps"]).timestamp.to_numpy(np.int64)
        ego = la.read_member(ez, features.get_clip_files_in_zip(row.clip_id, "egomotion.offline")["egomotion.offline"])
        spin = lidar[lidar.spin_index == SPIN].iloc[0]
        points, point_times = la.decode_spin(spin.draco_encoded_pointcloud)
    frame = int(np.argmin(np.abs(camera_ts - int(spin.spin_start_timestamp))))
    plain = cv2.imread(str(ROOT / "data" / "processed" / "images" / row.clip_id / f"after_{frame:04d}.jpg"), cv2.IMREAD_COLOR)
    rig = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    camera_a = (camera_from_rig @ np.column_stack((rig, np.ones(len(rig)))).T).T[:, :3]
    valid_time = (point_times >= int(ego.timestamp.min())) & (point_times <= int(ego.timestamp.max()))
    poses = la.vectorized_anchor_poses(ego, point_times[valid_time])
    camera_pose = interpolate_anchor_from_ego(ego, int(camera_ts[frame]))
    relative = np.linalg.inv(camera_pose) @ poses
    rig_b = np.einsum("nij,nj->ni", relative, np.column_stack((rig[valid_time], np.ones(valid_time.sum()))))[:, :3]
    camera_b = (camera_from_rig @ np.column_stack((rig_b, np.ones(len(rig_b)))).T).T[:, :3]

    def render(camera: np.ndarray, rig_points: np.ndarray) -> np.ndarray:
        mask = (camera[:, 2] >= 10.0) & (camera[:, 2] <= 40.0)
        p, r = camera[mask], rig_points[mask]
        rays = p / p[:, 2, None]
        uv = np.column_stack((K[0, 0] * rays[:, 0] + K[0, 2], K[1, 1] * rays[:, 1] + K[1, 2]))
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 384)
        uv, z = uv[inside], p[inside, 2]
        t = np.clip((z - 10.0) / 30.0, 0.0, 1.0)
        colors = np.column_stack((255 * (1 - t), 255 * (1 - np.abs(2 * t - 1)), 255 * t)).astype(np.uint8)
        out = plain.copy()
        for (u, v), color in zip(uv.astype(int), colors):
            cv2.circle(out, (int(u), int(v)), 1, tuple(int(x) for x in color), -1)
        return out

    a, b = render(camera_a, rig), render(camera_b, rig[valid_time])
    x, y, w, h = CROP
    panels = [plain[y:y+h, x:x+w], a[y:y+h, x:x+w], b[y:y+h, x:x+w]]
    panels = [cv2.resize(p, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST) for p in panels]
    for panel, label in zip(panels, ("plain photo", "Version A", "Version B")):
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 34), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    output = ROOT / "outputs" / "timing_check" / "0c3e9b08_spin130_sign_4x.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(json.dumps({"output": str(output.relative_to(ROOT)), "clip": row.clip_id, "spin_index": SPIN, "frame_index": frame, "crop": CROP, "structure": "green roadside freeway sign reading Watsonville", "range_m": [10, 40], "dots": "1 px", "panels": ["plain photo", "Version A", "Version B"]}, indent=2))


if __name__ == "__main__":
    main()
