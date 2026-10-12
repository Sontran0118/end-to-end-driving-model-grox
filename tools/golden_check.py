"""Recompute one saved label and compare it byte-for-byte at the Grid fields."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_labels import ORIGIN, VOXEL, box_extents, box_masks, transform_points_version_b, vehicle_dimensions, voxelize
from occgrid import OCC
from lidar_alignment import decode_spin, load_features, local_path, read_member
from transforms import interpolate_anchor_from_ego, lidar_to_ego_transform


def main() -> None:
    clips = pd.read_csv(ROOT / "clips.csv", dtype={"clip_id": str})
    clip = clips[clips.clip_id.str.startswith("49eb6e65")].iloc[0]
    clip_id, chunk, spin_index = str(clip.clip_id), int(clip.chunk), 15
    matches = pd.read_parquet(ROOT / "data" / "processed" / "matches.parquet")
    match = matches[(matches.clip_id == clip_id) & (matches.target_kind == "keyframe") &
                    (matches.target_spin_index == spin_index)].iloc[0]
    features = load_features()
    dimensions = vehicle_dimensions(clips)
    dims = dimensions[dimensions.clip_id == clip_id].iloc[0]
    extrinsics = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    lidar_member = features.get_clip_files_in_zip(clip_id, "lidar_top_360fov")["pointclouds"]
    ego_member = features.get_clip_files_in_zip(clip_id, "egomotion.offline")["egomotion.offline"]
    with zipfile.ZipFile(local_path(features, chunk, "lidar_top_360fov")) as lz, \
            zipfile.ZipFile(local_path(features, chunk, "egomotion.offline")) as ez:
        lidar = read_member(lz, lidar_member)
        ego = read_member(ez, ego_member)
    spin = lidar[lidar.spin_index == spin_index].iloc[0]
    points, point_times = decode_spin(spin.draco_encoded_pointcloud)
    transformed = transform_points_version_b(points, point_times, lidar_to_rig, ego, int(match.camera_timestamp))
    inside, _ = box_masks(transformed, box_extents(dims))
    expected_occ, expected_points = voxelize(transformed[~inside])
    expected_pose = interpolate_anchor_from_ego(ego, int(match.camera_timestamp)).astype(np.float32)
    manifest = pd.read_parquet(ROOT / "data" / "labels" / "labels_manifest.parquet")
    record = manifest[(manifest.clip_id == clip_id) & (manifest.spin_index == spin_index)].iloc[0]
    failures = []
    with np.load(ROOT / str(record.path)) as saved:
        checks = {
            "occ": np.array_equal(saved["occ"], expected_occ),
            "origin": np.array_equal(saved["origin"], ORIGIN),
            "voxel": np.array_equal(saved["voxel"], np.float32(VOXEL)),
            "pose": np.array_equal(saved["pose"], expected_pose),
            "format_version": int(saved["format_version"]) == 1,
            "clip_id": str(saved["clip_id"]) == clip_id,
            "spin_index": int(saved["spin_index"]) == spin_index,
        }
        failures.extend(name for name, passed in checks.items() if not passed)
    manifest_points = int(record.n_points_in_grid)
    if manifest_points != expected_points:
        failures.append("manifest.n_points_in_grid")
    if failures:
        print(f"FAIL: {clip_id[:8]} spin {spin_index}; mismatches={failures}")
        raise SystemExit(1)
    print(f"PASS: {clip_id[:8]} spin {spin_index}; points_in_grid={expected_points}; occ_voxels={int(np.count_nonzero(expected_occ == OCC))}")


if __name__ == "__main__":
    main()
