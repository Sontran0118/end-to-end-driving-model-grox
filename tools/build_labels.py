"""Build sparse-return occupancy labels from local PhysicalAI-AV ZIP members.

The grid is evaluated at the camera-frame time.  LiDAR points are transformed
with the same per-point Version-B motion compensation used by
``tools/lidar_alignment.py``.  The configured vehicle box is expressed in the
camera-time rig frame and is removed before voxelization.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lidar_alignment import (
    decode_spin,
    load_features,
    local_path,
    read_member,
    vectorized_anchor_poses,
)
from occgrid import Grid, FREE, OCC, UNKNOWN
from transforms import interpolate_anchor_from_ego, lidar_to_ego_transform

VOXEL = 0.4
SHAPE = (200, 100, 16)
ORIGIN = np.array((0.0, -20.0, -1.0), dtype=np.float32)
OWN_VEHICLE_MARGIN = 0.20
SHELL_METRES = 0.5


def vehicle_dimensions(clips: pd.DataFrame) -> pd.DataFrame:
    paths = sorted((ROOT / "data" / "raw" / "calibration" / "vehicle_dimensions").glob("*.parquet"))
    dimensions = pd.concat([pd.read_parquet(p).reset_index() for p in paths], ignore_index=True)
    dimensions = dimensions.drop_duplicates("clip_id")
    return clips[["clip_id", "split", "chunk"]].merge(dimensions, on="clip_id", how="left", validate="one_to_one")


def box_extents(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    margin = OWN_VEHICLE_MARGIN
    center_x = float(row.rear_axle_to_bbox_center)
    length = float(row.length)
    width = float(row.width)
    height = float(row.height)
    return (
        center_x - length / 2 - margin,
        center_x + length / 2 + margin,
        -width / 2 - margin,
        width / 2 + margin,
        -margin,
        height + margin,
    )


def box_masks(points: np.ndarray, extents: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1, z0, z1 = extents
    inside = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
              (points[:, 1] >= y0) & (points[:, 1] <= y1) &
              (points[:, 2] >= z0) & (points[:, 2] <= z1))
    distance_components = np.column_stack([
        np.maximum(x0 - points[:, 0], 0.0), np.maximum(points[:, 0] - x1, 0.0),
        np.maximum(y0 - points[:, 1], 0.0), np.maximum(points[:, 1] - y1, 0.0),
        np.maximum(z0 - points[:, 2], 0.0), np.maximum(points[:, 2] - z1, 0.0),
    ])
    shell = (~inside) & (np.linalg.norm(distance_components, axis=1) <= SHELL_METRES)
    return inside, shell


def transform_points_version_b(points: np.ndarray, point_times: np.ndarray,
                               lidar_to_rig: np.ndarray, ego: pd.DataFrame,
                               camera_time: int) -> np.ndarray:
    """Return points in the rig frame at camera time, matching lidar_alignment.py Version B."""
    rig_points = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    valid = (point_times >= int(ego.timestamp.min())) & (point_times <= int(ego.timestamp.max()))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    point_poses = vectorized_anchor_poses(ego, point_times[valid])
    camera_pose = interpolate_anchor_from_ego(ego, int(camera_time))
    relative = np.linalg.inv(camera_pose) @ point_poses
    homogeneous = np.column_stack((rig_points[valid], np.ones(valid.sum())))
    return np.einsum("nij,nj->ni", relative, homogeneous)[:, :3]


def voxelize(points: np.ndarray) -> tuple[np.ndarray, int]:
    """Mark every in-grid return occupied; leave all other voxels UNKNOWN."""
    occ = np.full(SHAPE, UNKNOWN, dtype=np.uint8)
    indices = np.floor((points - ORIGIN[None, :]) / VOXEL).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.array(SHAPE)), axis=1)
    if np.any(valid):
        unique = np.unique(indices[valid], axis=0)
        occ[unique[:, 0], unique[:, 1], unique[:, 2]] = OCC
    return occ, int(np.sum(valid))


def save_label(path: Path, occ: np.ndarray, pose: np.ndarray, clip_id: str, spin_index: int) -> None:
    # Grid construction is intentionally before serialization so validation runs.
    grid = Grid(occ=occ, origin=ORIGIN.copy(), voxel=VOXEL, sem=None, flow=None, pose=pose.astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, occ=grid.occ, origin=grid.origin, voxel=np.float32(grid.voxel),
                        pose=grid.pose, format_version=np.int32(1), clip_id=clip_id,
                        spin_index=np.int64(spin_index))


def synthetic_tests() -> None:
    wall = np.array([[10.0, -20.0, -1.0], [10.0, -19.59, -0.59]], dtype=np.float64)
    occ, n = voxelize(wall)
    assert n == 2 and occ[25, 0, 0] == OCC and occ[25, 1, 1] == OCC
    corner = np.array([[0.0, -20.0, -1.0], [-0.001, -20.0, -1.0]])
    occ, _ = voxelize(corner)
    assert occ[0, 0, 0] == OCC and np.count_nonzero(occ == OCC) == 1
    idx = np.array([3, 4, 5])
    center = ORIGIN + (idx + 0.5) * VOXEL
    assert np.array_equal(np.floor((center - ORIGIN) / VOXEL).astype(int), idx)
    inside, _ = box_masks(np.array([[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]]), (-1, 1, -1, 1, -1, 1))
    assert inside.tolist() == [True, False]
    grid = Grid(occ=np.zeros(SHAPE, np.uint8), origin=ORIGIN.copy(), voxel=VOXEL,
                pose=np.eye(4, dtype=np.float32))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.npz"
        np.savez_compressed(path, occ=grid.occ, origin=grid.origin, voxel=grid.voxel, pose=grid.pose)
        with np.load(path) as loaded:
            reloaded = Grid(loaded["occ"], loaded["origin"], float(loaded["voxel"]), pose=loaded["pose"])
        assert np.array_equal(reloaded.occ, grid.occ) and np.array_equal(reloaded.pose, grid.pose)
    print("synthetic_tests: PASS; wall indices=(25,0,0),(25,1,1); corner boundary accepted; outside rejected")


def process_clip(row: pd.Series, matches: pd.DataFrame, features, labels_root: Path) -> list[dict]:
    clip_id, chunk = str(row.clip_id), int(row.chunk)
    dimensions = row
    extents = box_extents(dimensions)
    lidar_zip = local_path(features, chunk, "lidar_top_360fov")
    ego_zip = local_path(features, chunk, "egomotion.offline")
    extrinsics = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    lidar_member = features.get_clip_files_in_zip(clip_id, "lidar_top_360fov")["pointclouds"]
    ego_member = features.get_clip_files_in_zip(clip_id, "egomotion.offline")["egomotion.offline"]
    target_matches = matches[(matches.clip_id == clip_id) & (matches.target_kind == "keyframe")].sort_values("keyframe_spin_index")
    rows = []
    with zipfile.ZipFile(lidar_zip) as lz, zipfile.ZipFile(ego_zip) as ez:
        lidar = read_member(lz, lidar_member)
        ego = read_member(ez, ego_member)
        for match in target_matches.itertuples(index=False):
            spin_index = int(match.target_spin_index)
            spin = lidar[lidar.spin_index == spin_index].iloc[0]
            points, point_times = decode_spin(spin.draco_encoded_pointcloud)
            transformed = transform_points_version_b(points, point_times, lidar_to_rig, ego, int(match.camera_timestamp))
            inside, shell = box_masks(transformed, extents)
            filtered = transformed[~inside]
            occ, n_in_grid = voxelize(filtered)
            pose = interpolate_anchor_from_ego(ego, int(match.camera_timestamp)).astype(np.float32)
            path = labels_root / str(row.split) / clip_id / f"{spin_index:03d}.npz"
            save_label(path, occ, pose, clip_id, spin_index)
            rows.append({"clip_id": clip_id, "split": str(row.split), "spin_index": spin_index,
                         "camera_frame_index": int(match.frame_index), "n_occ": n_in_grid,
                         "n_removed_own_vehicle": int(inside.sum()), "n_shell_0p5m": int(shell.sum()),
                         "path": str(path.relative_to(ROOT)).replace("\\", "/")})
    return rows


def plot_diagnostic(record: dict, dims: pd.DataFrame, features, matches: pd.DataFrame, labels_root: Path, name: str) -> None:
    row = dims[dims.clip_id == record["clip_id"]].iloc[0]
    extents = box_extents(row)
    lidar_zip = local_path(features, int(row.chunk), "lidar_top_360fov")
    extrinsics = pd.read_parquet(local_path(features, int(row.chunk), "sensor_extrinsics.offline")).loc[row.clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    ego_zip = local_path(features, int(row.chunk), "egomotion.offline")
    lm = features.get_clip_files_in_zip(row.clip_id, "lidar_top_360fov")["pointclouds"]
    em = features.get_clip_files_in_zip(row.clip_id, "egomotion.offline")["egomotion.offline"]
    match = matches[(matches.clip_id == row.clip_id) & (matches.target_spin_index == record["spin_index"]) & (matches.target_kind == "keyframe")].iloc[0]
    with zipfile.ZipFile(lidar_zip) as lz, zipfile.ZipFile(ego_zip) as ez:
        lidar = read_member(lz, lm); ego = read_member(ez, em)
    spin = lidar[lidar.spin_index == int(record["spin_index"])].iloc[0]
    points, point_times = decode_spin(spin.draco_encoded_pointcloud)
    rig = transform_points_version_b(points, point_times, lidar_to_rig, ego, int(match.camera_timestamp))
    inside, _ = box_masks(rig, extents)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(rig[~inside, 0], rig[~inside, 1], s=16, c="#1769aa", alpha=0.45, linewidths=0)
    ax.scatter(rig[inside, 0], rig[inside, 1], s=16, c="#f28e2b", alpha=0.9, linewidths=0)
    x0, x1, y0, y1, _, _ = extents
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "r-", linewidth=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(-4, 10), ylim=(-5, 5), xlabel="rig X forward (m)", ylabel="rig Y left (m)",
           title=f"{name}: {row.clip_id[:8]} spin {record['spin_index']} — removed={record['n_removed_own_vehicle']}, shell={record['n_shell_0p5m']}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-4, 11, 1), minor=True); ax.set_yticks(np.arange(-5, 6, 1), minor=True)
    ax.grid(which="minor", alpha=0.5)
    output = ROOT / "outputs" / "labels_diagnostics" / f"{name}_{row.clip_id[:8]}_spin{int(record['spin_index']):03d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
    print(f"diagnostic_plot={output.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-prefix", help="process only one clip prefix")
    parser.add_argument("--exclude-prefix", help="skip one clip prefix when completing a prior run")
    parser.add_argument("--plots-only", action="store_true", help="read the completed manifest and make diagnostics only")
    args = parser.parse_args()
    synthetic_tests()
    clips = pd.read_csv(ROOT / "clips.csv")
    dims = vehicle_dimensions(clips)
    matches = pd.read_parquet(ROOT / "data" / "processed" / "matches.parquet")
    features = load_features()
    selected = dims.copy()
    if args.clip_prefix:
        selected = selected[selected.clip_id.str.startswith(args.clip_prefix)]
    if args.exclude_prefix:
        selected = selected[~selected.clip_id.str.startswith(args.exclude_prefix)]
    if len(selected) == 0:
        raise ValueError("no clips selected")
    labels_root = ROOT / "data" / "labels"
    if args.plots_only:
        manifest = pd.read_parquet(labels_root / "labels_manifest.parquet")
        for rank, (_, record) in enumerate(manifest.nlargest(10, "n_shell_0p5m").iterrows(), 1):
            print(json.dumps({"shell_rank": rank, "clip": record.clip_id[:8], "spin": int(record.spin_index), "count": int(record.n_shell_0p5m)}))
        for rank, (_, record) in enumerate(manifest.nlargest(10, "n_removed_own_vehicle").iterrows(), 1):
            print(json.dumps({"removed_rank": rank, "clip": record.clip_id[:8], "spin": int(record.spin_index), "count": int(record.n_removed_own_vehicle)}))
        for prefix, name in [("n_shell_0p5m", "largest_shell"), ("n_removed_own_vehicle", "largest_removed")]:
            for rank, (_, record) in enumerate(manifest.nlargest(2, prefix).iterrows(), 1):
                plot_diagnostic(record.to_dict(), dims, features, matches, labels_root, f"{name}_rank{rank}")
        print(json.dumps({"n_occ_min": int(manifest.n_occ.min()), "n_occ_median": float(manifest.n_occ.median()),
                          "n_occ_max": int(manifest.n_occ.max()), "bytes": sum(p.stat().st_size for p in labels_root.rglob("*.npz"))}))
        return
    new_rows = []
    for _, row in selected.iterrows():
        clip_rows = process_clip(row, matches, features, labels_root)
        new_rows.extend(clip_rows)
        print(json.dumps({"clip": str(row.clip_id)[:8], "keyframes": len(clip_rows),
                          "n_occ_min": min(x["n_occ"] for x in clip_rows),
                          "n_occ_max": max(x["n_occ"] for x in clip_rows)}))
    manifest_path = labels_root / "labels_manifest.parquet"
    if args.exclude_prefix and manifest_path.exists():
        old = pd.read_parquet(manifest_path)
        manifest = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
        manifest = manifest.drop_duplicates(["clip_id", "spin_index"], keep="last")
    else:
        manifest = pd.DataFrame(new_rows)
    manifest = manifest.sort_values(["clip_id", "spin_index"]).reset_index(drop=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "rows": len(manifest),
                      "own_vehicle_margin_m": OWN_VEHICLE_MARGIN}))
    if not args.clip_prefix:
        for rank, (_, record) in enumerate(manifest.nlargest(10, "n_shell_0p5m").iterrows(), 1):
            print(json.dumps({"shell_rank": rank, "clip": record.clip_id[:8], "spin": int(record.spin_index), "count": int(record.n_shell_0p5m)}))
        for rank, (_, record) in enumerate(manifest.nlargest(10, "n_removed_own_vehicle").iterrows(), 1):
            print(json.dumps({"removed_rank": rank, "clip": record.clip_id[:8], "spin": int(record.spin_index), "count": int(record.n_removed_own_vehicle)}))
        for prefix, name in [("n_shell_0p5m", "largest_shell"), ("n_removed_own_vehicle", "largest_removed")]:
            for rank, (_, record) in enumerate(manifest.nlargest(2, prefix).iterrows(), 1):
                plot_diagnostic(record.to_dict(), dims, features, matches, labels_root, f"{name}_rank{rank}")
        print(json.dumps({"n_occ_min": int(manifest.n_occ.min()), "n_occ_median": float(manifest.n_occ.median()),
                          "n_occ_max": int(manifest.n_occ.max()), "bytes": sum(p.stat().st_size for p in labels_root.rglob("*.npz"))}))


if __name__ == "__main__":
    main()
