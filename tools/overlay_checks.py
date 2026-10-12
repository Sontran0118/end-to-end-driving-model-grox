"""Create review-only LiDAR overlay checks without touching prior outputs."""

from __future__ import annotations

import cv2
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import lidar_alignment as la  # noqa: E402
from transforms import camera_to_ego_transform, lidar_to_ego_transform  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "outputs" / "overlays_check"
LOW_SPINS = {"13ae70ee": 110, "49eb6e65": 15, "0c3e9b08": 195}
# (x, y, width, height), chosen from the three lowest-speed rectified frames.
CROPS = {
    "13ae70ee": {"pole": (690, 70, 100, 190), "car_rear": (850, 145, 180, 130), "building_corner": (880, 70, 180, 190)},
    "49eb6e65": {"pole": (1080, 0, 150, 180), "car_rear": (560, 135, 190, 150), "building_corner": (1040, 105, 200, 190)},
    "0c3e9b08": {"pole": (590, 90, 100, 170), "car_rear": (430, 150, 210, 150), "structure_corner": (1030, 80, 200, 220)},
}


def clip_context(prefix: str):
    features = la.load_features()
    clips = pd.read_csv(ROOT / "clips.csv")
    row = clips[clips.clip_id.str.startswith(prefix)].iloc[0]
    chunk = int(row.chunk)
    extrinsics = pd.read_parquet(la.local_path(features, chunk, "sensor_extrinsics.offline")).loc[row.clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    camera_to_rig = camera_to_ego_transform(extrinsics.loc["camera_front_wide_120fov"])
    source = json.loads((ROOT / "data" / "processed" / "K" / f"{row.clip_id}.json").read_text())["source_intrinsics"]
    with __import__("zipfile").ZipFile(la.local_path(features, chunk, "lidar_top_360fov")) as lz, __import__("zipfile").ZipFile(la.local_path(features, chunk, "camera_front_wide_120fov")) as cz, __import__("zipfile").ZipFile(la.local_path(features, chunk, "egomotion.offline")) as ez:
        lm = features.get_clip_files_in_zip(row.clip_id, "lidar_top_360fov")["pointclouds"]
        cm = features.get_clip_files_in_zip(row.clip_id, "camera_front_wide_120fov")["frame_timestamps"]
        lidar = la.read_member(lz, lm).sort_values("spin_start_timestamp").reset_index(drop=True)
        camera_ts = la.read_member(cz, cm).timestamp.to_numpy(np.int64)
        em = features.get_clip_files_in_zip(row.clip_id, "egomotion.offline")["egomotion.offline"]
        ego = la.read_member(ez, em)
        spin = LOW_SPINS[prefix]
        lidar_row = lidar[lidar.spin_index == spin].iloc[0]
        frame_index = int(np.argmin(np.abs(camera_ts - int(lidar_row.spin_start_timestamp))))
        points, point_times = la.decode_spin(lidar_row.draco_encoded_pointcloud)
    rig = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    camera = (np.linalg.inv(camera_to_rig) @ np.column_stack((rig, np.ones(len(rig)))).T).T[:, :3]
    return row, spin, frame_index, int(camera_ts[frame_index]), source, camera, rig, point_times, ego, camera_to_rig, lidar_to_rig


def project_height(camera: np.ndarray, rig: np.ndarray, source: dict) -> tuple[np.ndarray, np.ndarray]:
    valid = (camera[:, 2] >= 2.0) & (camera[:, 2] <= 40.0)
    p = camera[valid]
    r = rig[valid]
    rays = p / p[:, 2, None]
    uv = np.column_stack((369.5041722813606 * rays[:, 0] + 640.0, 369.5041722813606 * rays[:, 1] + 192.0))
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 384)
    return uv[inside], r[inside, 2], np.flatnonzero(valid)[inside]


def draw_height(image: np.ndarray, uv: np.ndarray, heights: np.ndarray) -> np.ndarray:
    out = image.copy()
    lo, hi = np.percentile(heights, [2, 98]) if len(heights) else (-1.0, 1.0)
    t = np.clip((heights - lo) / max(hi - lo, 1e-6), 0, 1)
    colors = np.column_stack((255 * (1 - t), 255 * (1 - np.abs(2 * t - 1)), 255 * t)).astype(np.uint8)
    for (u, v), color in zip(uv.astype(int), colors):
        cv2.circle(out, (int(u), int(v)), 1, tuple(int(x) for x in color), -1)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.hstack((left, right))


def crop_pair(plain: np.ndarray, overlay: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    a = cv2.resize(plain[y:y+h, x:x+w], (2*w, 2*h), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(overlay[y:y+h, x:x+w], (2*w, 2*h), interpolation=cv2.INTER_NEAREST)
    return side_by_side(a, b)


def make_low_checks(prefix: str) -> None:
    row, spin, frame, _, source, camera, rig, _, _, camera_to_rig, lidar_to_rig = clip_context(prefix)
    image_path = ROOT / "data" / "processed" / "images" / row.clip_id / f"after_{frame:04d}.jpg"
    plain = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    uv, heights, _ = project_height(camera, rig, source)
    overlay = draw_height(plain, uv, heights)
    CHECK.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CHECK / f"{prefix}_low_{spin:03d}_plain_overlay.jpg"), side_by_side(plain, overlay), [cv2.IMWRITE_JPEG_QUALITY, 95])
    for name, box in CROPS[prefix].items():
        cv2.imwrite(str(CHECK / f"{prefix}_low_{spin:03d}_{name}_2x.jpg"), crop_pair(plain, overlay, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(json.dumps({"clip": prefix, "spin_index": spin, "frame_index": frame, "points_5_40_in_image": len(uv), "plain_overlay": str((CHECK / f'{prefix}_low_{spin:03d}_plain_overlay.jpg').relative_to(ROOT)), "sensor_xyz_rig": {"camera": camera_to_rig[:3, 3].tolist(), "lidar": lidar_to_rig[:3, 3].tolist()}}))


def make_ab_check(prefix: str = "0c3e9b08") -> None:
    row, spin, frame, camera_time, source, camera, rig, point_times, ego, camera_to_rig, lidar_to_rig = clip_context(prefix)
    a_path = ROOT / "outputs" / "overlays" / f"{row.clip_id}_{spin:03d}_A.jpg"
    b_path = ROOT / "outputs" / "overlays" / f"{row.clip_id}_{spin:03d}_B.jpg"
    a, b = cv2.imread(str(a_path)), cv2.imread(str(b_path))
    cv2.imwrite(str(CHECK / f"{prefix}_{spin:03d}_A_B_side_by_side.jpg"), side_by_side(a, b), [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Recompute corresponding rectified projections for common raw points.
    uv_a, _, ids_a = project_height(camera, rig, source)
    valid_time = (point_times >= int(ego.timestamp.min())) & (point_times <= int(ego.timestamp.max()))
    valid_indices = np.flatnonzero(valid_time)
    poses = la.vectorized_anchor_poses(ego, point_times[valid_time])
    camera_pose = la.interpolate_anchor_from_ego(ego, camera_time)
    relative = np.linalg.inv(camera_pose) @ poses
    moved = np.einsum("nij,nj->ni", relative, np.column_stack((rig[valid_time], np.ones(valid_time.sum()))))[:, :3]
    camera_b = (np.linalg.inv(camera_to_rig) @ np.column_stack((moved, np.ones(len(moved)))).T).T[:, :3]
    uv_b, _, ids_b = project_height(camera_b, rig[valid_time], source)
    ids_b = valid_indices[ids_b]
    common, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    displacement = np.linalg.norm(uv_b[ib] - uv_a[ia], axis=1)
    print(json.dumps({"clip": prefix, "spin_index": spin, "A_B_side_by_side": str((CHECK / f'{prefix}_{spin:03d}_A_B_side_by_side.jpg').relative_to(ROOT)), "common_points_5_40": len(common), "median_overlay_move_pixels": float(np.median(displacement)), "max_overlay_move_pixels": float(np.max(displacement))}))


def main() -> None:
    overlay_dir = ROOT / "outputs" / "overlays"
    rows = []
    speed_by = {"13ae70ee": {30: 10.218988626789717, 110: 0.324073636522264, 160: 7.288027235428657}, "49eb6e65": {15: 0.018341131688149503, 105: 6.682076020182105, 195: 12.507166587555348}, "0c3e9b08": {55: 29.9959408187287, 130: 29.68549562227266, 195: 28.03764806310256}}
    for path in sorted(overlay_dir.glob("*.jpg")):
        prefix = path.name[:8]
        spin = int(path.name.split("_")[-2])
        rows.append({"filename": path.name, "spin_index": spin, "ego_speed_mps": speed_by[prefix][spin]})
    print(pd.DataFrame(rows).to_string(index=False))
    for prefix in LOW_SPINS:
        make_low_checks(prefix)
    make_ab_check()


if __name__ == "__main__":
    main()
