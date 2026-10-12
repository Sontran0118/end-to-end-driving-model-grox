"""Create zoomed vehicle-box review plots and count box-near LiDAR returns."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lidar_alignment import decode_spin, load_features, local_path, read_member
from transforms import lidar_to_ego_transform, sensor_to_ego_transform

ROOT = Path(__file__).resolve().parents[1]
MARGIN_M = 0.20
PLOT_CLIPS = {"13ae70ee": 110, "49eb6e65": 15, "0c3e9b08": 195}
PLOT_X = (-4.0, 10.0)
PLOT_Y = (-5.0, 5.0)
PLOT_Z = (-2.0, 4.0)


def dimensions_table(clips: pd.DataFrame) -> pd.DataFrame:
    pieces = [pd.read_parquet(path).reset_index() for path in sorted(
        (ROOT / "data" / "raw" / "calibration" / "vehicle_dimensions").glob("*.parquet")
    )]
    dims = pd.concat(pieces, ignore_index=True).drop_duplicates("clip_id")
    return clips[["clip_id", "split", "chunk"]].merge(dims, on="clip_id", how="left")


def box_extents(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    x0 = float(row.rear_axle_to_bbox_center) - float(row.length) / 2 - MARGIN_M
    x1 = float(row.rear_axle_to_bbox_center) + float(row.length) / 2 + MARGIN_M
    y0 = -float(row.width) / 2 - MARGIN_M
    y1 = float(row.width) / 2 + MARGIN_M
    return x0, x1, y0, y1, -MARGIN_M, float(row.height) + MARGIN_M


def masks(points: np.ndarray, extents: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1, z0, z1 = extents
    inside = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
              (points[:, 1] >= y0) & (points[:, 1] <= y1) &
              (points[:, 2] >= z0) & (points[:, 2] <= z1))
    delta = np.column_stack([
        np.maximum(x0 - points[:, 0], 0.0), np.maximum(points[:, 0] - x1, 0.0),
        np.maximum(y0 - points[:, 1], 0.0), np.maximum(points[:, 1] - y1, 0.0),
        np.maximum(z0 - points[:, 2], 0.0), np.maximum(points[:, 2] - z1, 0.0),
    ])
    shell = (~inside) & (np.linalg.norm(delta, axis=1) <= 0.5)
    return inside, shell


def load_clip_spin(row: pd.Series, features, spin_index: int):
    clip_id, chunk = str(row.clip_id), int(row.chunk)
    lidar_zip = local_path(features, chunk, "lidar_top_360fov")
    extrinsics = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
    lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
    camera_to_rig = sensor_to_ego_transform(extrinsics.loc["camera_front_wide_120fov"])
    with zipfile.ZipFile(lidar_zip) as zf:
        member = features.get_clip_files_in_zip(clip_id, "lidar_top_360fov")["pointclouds"]
        lidar = read_member(zf, member)
    spin = lidar[lidar.spin_index == spin_index].iloc[0]
    points, timestamps = decode_spin(spin.draco_encoded_pointcloud)
    rig = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    return rig, lidar_to_rig[:3, 3], camera_to_rig[:3, 3], timestamps


def style_axis(axis, xlim, ylim, minor_step: float) -> None:
    axis.set_xlim(*xlim); axis.set_ylim(*ylim)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(np.arange(np.ceil(xlim[0]), xlim[1] + 0.01, minor_step), minor=True)
    axis.set_yticks(np.arange(np.ceil(ylim[0]), ylim[1] + 0.01, minor_step), minor=True)
    axis.grid(which="minor", color="#b8b8b8", linewidth=0.5, alpha=0.6)
    axis.grid(which="major", color="#888888", linewidth=0.8, alpha=0.5)


def draw_box(axis, extents, vertical: bool) -> None:
    x0, x1, y0, y1, z0, z1 = extents
    a0, a1 = (z0, z1) if vertical else (y0, y1)
    axis.plot([x0, x1, x1, x0, x0], [a0, a0, a1, a1, a0], "r-", linewidth=2)


def plot_clip(prefix: str, spin_index: int, row: pd.Series, features) -> dict:
    rig, lidar_xyz, camera_xyz, _ = load_clip_spin(row, features, spin_index)
    extents = box_extents(row)
    inside, shell = masks(rig, extents)
    out = ROOT / "outputs" / "vehicle_box_proposal"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].scatter(rig[~inside, 0], rig[~inside, 1], s=16, c="#1769aa", alpha=0.45, linewidths=0)
    axes[0].scatter(rig[inside, 0], rig[inside, 1], s=16, c="#f28e2b", alpha=0.9, linewidths=0)
    draw_box(axes[0], extents, False)
    axes[0].scatter([0], [0], marker="+", s=100, c="black")
    axes[0].scatter([lidar_xyz[0]], [lidar_xyz[1]], marker="^", s=65, c="purple")
    axes[0].scatter([camera_xyz[0]], [camera_xyz[1]], marker="s", s=55, c="green")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("rig X forward (m)"); axes[0].set_ylabel("rig Y left (m)"); axes[0].set_title("Top-down")
    style_axis(axes[0], PLOT_X, PLOT_Y, 1.0)
    axes[1].scatter(rig[~inside, 0], rig[~inside, 2], s=16, c="#1769aa", alpha=0.45, linewidths=0)
    axes[1].scatter(rig[inside, 0], rig[inside, 2], s=16, c="#f28e2b", alpha=0.9, linewidths=0)
    draw_box(axes[1], extents, True)
    axes[1].scatter([0], [0], marker="+", s=100, c="black")
    axes[1].scatter([lidar_xyz[0]], [lidar_xyz[2]], marker="^", s=65, c="purple")
    axes[1].scatter([camera_xyz[0]], [camera_xyz[2]], marker="s", s=55, c="green")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("rig X forward (m)"); axes[1].set_ylabel("rig Z up (m)"); axes[1].set_title("Side view")
    style_axis(axes[1], PLOT_X, PLOT_Z, 0.5)
    fig.suptitle(f"{prefix} spin {spin_index} — proposed margin {MARGIN_M:.2f} m")
    fig.tight_layout()
    path = out / f"zoom_{prefix}_spin{spin_index:03d}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"clip_id": prefix, "spin_index": spin_index, "total": len(rig),
            "inside": int(inside.sum()), "shell": int(shell.sum()), "extents": extents,
            "output": str(path.relative_to(ROOT))}


def all_keyframe_counts(clips: pd.DataFrame, features) -> pd.DataFrame:
    keyframes = pd.read_parquet(ROOT / "data" / "processed" / "keyframes.parquet")
    rows = []
    for _, row in clips.iterrows():
        clip_id, chunk = str(row.clip_id), int(row.chunk)
        lidar_zip = local_path(features, chunk, "lidar_top_360fov")
        extrinsics = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
        lidar_to_rig = lidar_to_ego_transform(extrinsics.loc["lidar_top_360fov"])
        extents = box_extents(row)
        target = keyframes[keyframes.clip_id == clip_id]
        with zipfile.ZipFile(lidar_zip) as zf:
            member = features.get_clip_files_in_zip(clip_id, "lidar_top_360fov")["pointclouds"]
            lidar = read_member(zf, member)
            for spin_index in target.spin_index.astype(int):
                spin = lidar[lidar.spin_index == spin_index].iloc[0]
                points, _ = decode_spin(spin.draco_encoded_pointcloud)
                rig = (lidar_to_rig @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
                inside, shell = masks(rig, extents)
                rows.append((clip_id, int(spin_index), len(rig), int(inside.sum()), int(shell.sum())))
    return pd.DataFrame(rows, columns=["clip_id", "spin_index", "total", "inside", "shell"])


def main() -> None:
    clips = pd.read_csv(ROOT / "clips.csv")
    dims = dimensions_table(clips)
    features = load_features()
    # Tiny geometry smoke test before touching dataset members.
    print("tiny_geometry_test", masks(np.array([[0., 0., 0.], [100., 100., 100.]]), (-1, 1, -1, 1, -1, 1))[0].tolist())
    for prefix, spin in PLOT_CLIPS.items():
        result = plot_clip(prefix, spin, dims[dims.clip_id.str.startswith(prefix)].iloc[0], features)
        print(json.dumps(result))
    counts = all_keyframe_counts(dims, features)
    print(json.dumps({"keyframes": len(counts), "inside_median": float(counts.inside.median()),
                      "inside_max": int(counts.inside.max()), "shell_median": float(counts.shell.median()),
                      "shell_max": int(counts.shell.max())}, indent=2))


if __name__ == "__main__":
    main()
