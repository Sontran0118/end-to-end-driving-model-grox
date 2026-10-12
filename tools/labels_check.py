"""Review saved occupancy labels without recomputing or rewriting them."""

from __future__ import annotations

import shutil
import sys
import io
import json
import zipfile
from pathlib import Path

import cv2
import DracoPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from occgrid import Grid, OCC
from lidar_alignment import draw_overlay as draw_lidar_overlay
from lidar_alignment import load_features, local_path, project as lidar_project, read_member, vectorized_anchor_poses
from transforms import camera_to_ego_transform, interpolate_anchor_from_ego, lidar_to_ego_transform
from viewer.grid import surface_mask

K = np.array([[369.504, 0.0, 640.0], [0.0, 369.504, 192.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1280, 384)
TARGETS = {"49eb6e65": 15, "13ae70ee": 110, "0c3e9b08": 195}


def dimensions_table() -> pd.DataFrame:
    clips = pd.read_csv(ROOT / "clips.csv")
    paths = sorted((ROOT / "data" / "raw" / "calibration" / "vehicle_dimensions").glob("*.parquet"))
    dims = pd.concat([pd.read_parquet(path).reset_index() for path in paths], ignore_index=True).drop_duplicates("clip_id")
    return clips[["clip_id", "split", "chunk"]].merge(dims, on="clip_id", validate="one_to_one")


def box_extents(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    margin = 0.20
    center = float(row.rear_axle_to_bbox_center)
    return (center - float(row.length) / 2 - margin, center + float(row.length) / 2 + margin,
            -float(row.width) / 2 - margin, float(row.width) / 2 + margin,
            -margin, float(row.height) + margin)


def project_rectified_points(points_rig: np.ndarray, camera_to_rig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project rig-frame points with the verified rectified pinhole camera path."""
    rig_to_camera = np.linalg.inv(camera_to_rig)
    camera = (rig_to_camera @ np.column_stack((points_rig, np.ones(len(points_rig)))).T).T[:, :3]
    valid_depth = camera[:, 2] > 0.0
    uv = np.full((len(camera), 2), np.nan, dtype=np.float64)
    uv[valid_depth, 0] = K[0, 0] * camera[valid_depth, 0] / camera[valid_depth, 2] + K[0, 2]
    uv[valid_depth, 1] = K[1, 1] * camera[valid_depth, 1] / camera[valid_depth, 2] + K[1, 2]
    valid_image = valid_depth & (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_SIZE[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_SIZE[1])
    return camera, uv, valid_image


def label_path(manifest: pd.DataFrame, clip_prefix: str, spin: int) -> tuple[pd.Series, Path]:
    record = manifest[(manifest.clip_id.str.startswith(clip_prefix)) & (manifest.spin_index == spin)].iloc[0]
    return record, ROOT / record.path


def plot_views(grid: Grid, row: pd.Series, prefix: str, spin: int, output: Path) -> None:
    occ = grid.occ == OCC
    top = occ.any(axis=2)
    side = occ.any(axis=1)
    x0, x1 = float(grid.origin[0]), float(grid.origin[0] + grid.occ.shape[0] * grid.voxel)
    y0, y1 = float(grid.origin[1]), float(grid.origin[1] + grid.occ.shape[1] * grid.voxel)
    z0, z1 = float(grid.origin[2]), float(grid.origin[2] + grid.occ.shape[2] * grid.voxel)
    bx0, bx1, by0, by1, bz0, bz1 = box_extents(row)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(top.T, origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest", cmap="Greys")
    axes[0].plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0], "r-", linewidth=2)
    axes[0].set(xlim=(0, 80), ylim=(-20, 20), xlabel="rig X forward (m)", ylabel="rig Y left (m)", title="Top-down OCC")
    axes[1].imshow(side.T, origin="lower", extent=(x0, x1, z0, z1), interpolation="nearest", cmap="Greys")
    axes[1].plot([bx0, bx1, bx1, bx0, bx0], [bz0, bz0, bz1, bz1, bz0], "r-", linewidth=2)
    axes[1].set(xlim=(0, 80), ylim=(-1, 5), xlabel="rig X forward (m)", ylabel="rig Z up (m)", title="Side OCC")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, 81, 1), minor=True)
        ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0]), ax.get_ylim()[1] + 0.1, 1), minor=True)
        ax.grid(which="minor", color="#999999", alpha=0.45, linewidth=0.5)
    fig.suptitle(f"{prefix} spin {spin} — saved Grid projection")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_spin{spin:03d}_grid_views.png", dpi=160)
    plt.close(fig)
    for name, image, extent, other0, other1, ylabel, title, step in (
        ("topdown", top, (x0, x1, y0, y1), -20, 20, "rig Y left (m)", "Top-down OCC", 1.0),
        ("side", side, (x0, x1, z0, z1), -1, 5, "rig Z up (m)", "Side OCC", 0.5),
    ):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(image.T, origin="lower", extent=extent, interpolation="nearest", cmap="Greys")
        a0, a1 = (by0, by1) if name == "topdown" else (bz0, bz1)
        ax.plot([bx0, bx1, bx1, bx0, bx0], [a0, a0, a1, a1, a0], "r-", linewidth=2)
        ax.set(xlim=(0, 80), ylim=(other0, other1), xlabel="rig X forward (m)", ylabel=ylabel, title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, 81, 1), minor=True)
        ax.set_yticks(np.arange(other0, other1 + 0.1, step), minor=True)
        ax.grid(which="minor", color="#999999", alpha=0.45, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_spin{spin:03d}_{name}.png", dpi=160)
        plt.close(fig)


def decode_spin(encoded: bytes) -> tuple[np.ndarray, np.ndarray]:
    cloud = DracoPy.decode(encoded)
    points = np.asarray(cloud.points, dtype=np.float64)
    timestamps = np.asarray(cloud.get_attribute_by_name("timestamp")["data"]).reshape(-1).astype(np.int64)
    return points, timestamps


def motion_compensated_lidar(row: pd.Series, spin: int, camera_time: int, features):
    lidar_zip = local_path(features, int(row.chunk), "lidar_top_360fov")
    ego_zip = local_path(features, int(row.chunk), "egomotion.offline")
    extrinsics_path = next((ROOT / "data" / "raw" / "calibration" / "sensor_extrinsics.offline").glob(
        f"*chunk_{int(row.chunk):04d}.parquet"))
    extrinsics = pd.read_parquet(extrinsics_path).loc[row.clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    lidar_member = features.get_clip_files_in_zip(row.clip_id, "lidar_top_360fov")["pointclouds"]
    ego_member = features.get_clip_files_in_zip(row.clip_id, "egomotion.offline")["egomotion.offline"]
    with zipfile.ZipFile(lidar_zip) as lz, zipfile.ZipFile(ego_zip) as ez:
        lidar = read_member(lz, lidar_member)
        ego = read_member(ez, ego_member)
    spin_row = lidar[lidar.spin_index == spin].iloc[0]
    points, point_times = decode_spin(spin_row.draco_encoded_pointcloud)
    rig_points = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    valid = (point_times >= int(ego.timestamp.min())) & (point_times <= int(ego.timestamp.max()))
    point_poses = vectorized_anchor_poses(ego, point_times[valid])
    camera_pose = interpolate_anchor_from_ego(ego, camera_time)
    relative = np.linalg.inv(camera_pose) @ point_poses
    transformed = np.einsum("nij,nj->ni", relative,
                            np.column_stack((rig_points[valid], np.ones(valid.sum()))))[:, :3]
    camera_to_rig = camera_to_ego_transform(extrinsics.loc["camera_front_wide_120fov"])
    camera_points = (np.linalg.inv(camera_to_rig) @ np.column_stack((transformed, np.ones(len(transformed)))).T).T[:, :3]
    return camera_points


def projection_stats(name: str, camera_points: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> None:
    z = camera_points[:, 2]
    near = np.hypot(uv[:, 0] - 640.0, uv[:, 1] - 192.0) <= 100.0
    print(f"{name}: depth_z_min_median_max={z.min():.3f},{np.median(z):.3f},{z.max():.3f}; "
          f"u_min_median_max={uv[:, 0].min():.1f},{np.median(uv[:, 0]):.1f},{uv[:, 0].max():.1f}; "
          f"v_min_median_max={uv[:, 1].min():.1f},{np.median(uv[:, 1]):.1f},{uv[:, 1].max():.1f}; "
          f"inside_1280x384={int(valid.sum())}/{len(valid)}; "
          f"within_100px_of_(640,192)={int(near.sum())}/{len(near)}")


def alpha_save(photo: np.ndarray, layer: np.ndarray, path: Path) -> float:
    result = cv2.addWeighted(photo, 0.55, layer, 0.45, 0.0)
    difference = float(np.mean(np.abs(result.astype(np.int16) - photo.astype(np.int16))))
    if difference > 0.0:
        cv2.imwrite(str(path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"saved={path.relative_to(ROOT)} mean_absolute_difference={difference:.6f}")
    return difference


def overlay(grid: Grid, row: pd.Series, prefix: str, spin: int, output: Path, dims: pd.DataFrame, features) -> None:
    matches = pd.read_parquet(ROOT / "data" / "processed" / "matches.parquet")
    match = matches[(matches.clip_id == row.clip_id) & (matches.target_kind == "keyframe") &
                    (matches.target_spin_index == spin)].iloc[0]
    extrinsics_path = next((ROOT / "data" / "raw" / "calibration" / "sensor_extrinsics.offline").glob(
        f"*chunk_{int(row.chunk):04d}.parquet"))
    extrinsics = pd.read_parquet(extrinsics_path)
    camera_row = extrinsics.loc[row.clip_id, "camera_front_wide_120fov"]
    camera_to_rig = camera_to_ego_transform(camera_row)
    rig_to_camera = np.linalg.inv(camera_to_rig)
    image = ROOT / "data" / "processed" / "images" / row.clip_id / f"after_{int(match.frame_index):04d}.jpg"
    plain = output / f"{prefix}_spin{spin:03d}_plain.jpg"
    overlay_path = output / f"{prefix}_spin{spin:03d}_overlay.jpg"
    photo = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(image)
    shutil.copyfile(image, plain)
    surface = surface_mask(grid.occ)
    idx = np.argwhere(surface)
    centers = grid.origin + (idx + 0.5) * grid.voxel
    cam, uv, valid = project_rectified_points(centers, camera_to_rig)
    projection_stats(f"{prefix} spin {spin} voxels", cam, uv, valid)
    print(f"{prefix} spin {spin}: center_formula=origin+(idx+0.5)*voxel")
    values = centers[:, 2]
    norm = plt.Normalize(float(grid.origin[2]), float(grid.origin[2] + grid.occ.shape[2] * grid.voxel))
    colors = (plt.get_cmap("viridis")(norm(values))[:, :3] * 255).astype(np.uint8)
    dots = photo.copy()
    for (u, v), color in zip(uv[valid], colors[valid]):
        cv2.circle(dots, (int(round(u)), int(round(v))), 2,
                   (int(color[2]), int(color[1]), int(color[0])), -1, cv2.LINE_AA)
    alpha_save(photo, dots, output / f"{prefix}_spin{spin:03d}_voxel_dots.jpg")
    squares = photo.copy()
    for (u, v), z, color in zip(uv[valid], cam[valid, 2], colors[valid]):
        half_w = max(1.0, K[0, 0] * grid.voxel / (2.0 * z))
        half_h = max(1.0, K[1, 1] * grid.voxel / (2.0 * z))
        x0, x1 = int(round(u - half_w)), int(round(u + half_w))
        y0, y1 = int(round(v - half_h)), int(round(v + half_h))
        cv2.rectangle(squares, (x0, y0), (x1, y1),
                      (int(color[2]), int(color[1]), int(color[0])), -1)
    alpha_save(photo, squares, output / f"{prefix}_spin{spin:03d}_voxel_squares.jpg")
    alpha_save(photo, squares, overlay_path)
    lidar_points = motion_compensated_lidar(row, spin, int(match.camera_timestamp), features)
    source = json.loads((ROOT / "data" / "processed" / "K" / f"{row.clip_id}.json").read_text())['source_intrinsics']
    lidar_uv, lidar_depth = lidar_project(lidar_points, source)
    front = (lidar_points[:, 2] >= 2.0) & (lidar_points[:, 2] <= 80.0)
    lidar_front = lidar_points[front]
    lidar_rect_uv = np.column_stack((K[0, 0] * lidar_front[:, 0] / lidar_front[:, 2] + K[0, 2],
                                     K[1, 1] * lidar_front[:, 1] / lidar_front[:, 2] + K[1, 2]))
    lidar_inside = ((lidar_rect_uv[:, 0] >= 0) & (lidar_rect_uv[:, 0] < IMAGE_SIZE[0]) &
                    (lidar_rect_uv[:, 1] >= 0) & (lidar_rect_uv[:, 1] < IMAGE_SIZE[1]))
    projection_stats(f"{prefix} spin {spin} LiDAR Version B", lidar_front, lidar_rect_uv, lidar_inside)
    print(f"{prefix} spin {spin}: verified_lidar_helper_projected={len(lidar_uv)}")
    draw_lidar_overlay(image, lidar_uv, lidar_depth, output / f"{prefix}_spin{spin:03d}_lidar_B.jpg")
    print(f"{prefix} spin {spin}: surface_voxels={len(idx)}, projected_voxels={int(valid.sum())}, plain={plain.relative_to(ROOT)}")


def main() -> None:
    manifest = pd.read_parquet(ROOT / "data" / "labels" / "labels_manifest.parquet")
    dims = dimensions_table()
    output = ROOT / "outputs" / "labels_check"
    output.mkdir(parents=True, exist_ok=True)
    features = load_features()
    for prefix, spin in TARGETS.items():
        record, path = label_path(manifest, prefix, spin)
        with np.load(path) as data:
            grid = Grid(data["occ"], data["origin"], float(data["voxel"]), pose=data["pose"])
        plot_views(grid, dims[dims.clip_id == record.clip_id].iloc[0], prefix, spin, output)
        overlay(grid, dims[dims.clip_id == record.clip_id].iloc[0], prefix, spin, output, dims, features)
        counts = np.sum(grid.occ == OCC, axis=(0, 1))
        nonzero = np.flatnonzero(counts)
        print(f"{prefix} spin {spin}: OCC_per_z=" + ",".join(f"{i}:{int(n)}" for i, n in enumerate(counts) if n))
        print(f"{prefix} spin {spin}: OCC z-index range={int(nonzero.min()) if len(nonzero) else 'none'}..{int(nonzero.max()) if len(nonzero) else 'none'}")


if __name__ == "__main__":
    main()
