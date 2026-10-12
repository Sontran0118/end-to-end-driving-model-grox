"""Save one readable four-frame and occupancy-grid sample figure."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_labels import ORIGIN, OWN_VEHICLE_MARGIN, SHAPE, VOXEL
from keyframes_matches import HISTORY_OFFSETS
from labels_check import K, project_rectified_points
from lidar_alignment import load_features, local_path, read_member
from occgrid import Grid, OCC
from rectify_camera import HFOV_DEGREES
from transforms import camera_to_ego_transform
from viewer.grid import surface_mask


def _clip_row(prefix: str) -> pd.Series:
    clips = pd.read_csv(ROOT / "clips.csv", dtype={"clip_id": str})
    rows = clips[clips.clip_id.str.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError(f"expected one clip for prefix {prefix}, found {len(rows)}")
    return rows.iloc[0]


def _dimensions(clip_id: str) -> pd.Series:
    paths = sorted((ROOT / "data" / "raw" / "calibration" / "vehicle_dimensions").glob("*.parquet"))
    table = pd.concat([pd.read_parquet(path).reset_index() for path in paths], ignore_index=True)
    table = table.drop_duplicates("clip_id")
    return table[table.clip_id == clip_id].iloc[0]


def _ego_speed(ego: pd.DataFrame, timestamp: int) -> float:
    ego = ego.sort_values("timestamp")
    times = ego.timestamp.to_numpy(np.float64)
    positions = ego[["x", "y", "z"]].to_numpy(np.float64)
    speed = np.linalg.norm(np.gradient(positions, times, axis=0), axis=1) * 1_000_000.0
    return float(np.interp(timestamp, times, speed))


def _load_grid(record: pd.Series) -> Grid:
    with np.load(ROOT / str(record.path)) as data:
        return Grid(data["occ"], data["origin"], float(data["voxel"]), pose=data["pose"])


def _voxel_overlay(photo: np.ndarray, grid: Grid, camera_to_rig: np.ndarray) -> np.ndarray:
    surface = surface_mask(grid.occ)
    indices = np.argwhere(surface)
    centers = grid.origin + (indices + 0.5) * grid.voxel
    camera, uv, valid = project_rectified_points(centers, camera_to_rig)
    del camera
    colors = (plt.get_cmap("viridis")(
        plt.Normalize(float(grid.origin[2]), float(grid.origin[2] + grid.occ.shape[2] * grid.voxel))(centers[:, 2])
    )[:, :3] * 255).astype(np.uint8)
    dots = photo.copy()
    for (u, v), color in zip(uv[valid], colors[valid]):
        cv2.circle(dots, (int(round(u)), int(round(v))), 2,
                   (int(color[2]), int(color[1]), int(color[0])), -1, cv2.LINE_AA)
    return cv2.addWeighted(photo, 0.55, dots, 0.45, 0.0)


def _grid_view(ax, image: np.ndarray, extent: tuple[float, ...], title: str, xlabel: str, ylabel: str) -> None:
    ax.imshow(image.T, origin="lower", extent=extent, interpolation="nearest", cmap="Greys")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")


def make_sample(prefix: str, spin: int) -> tuple[Path, dict]:
    row = _clip_row(prefix)
    clip_id, chunk = str(row.clip_id), int(row.chunk)
    keyframes = pd.read_parquet(ROOT / "data" / "processed" / "keyframes.parquet")
    keyframe = keyframes[(keyframes.clip_id == clip_id) & (keyframes.spin_index == spin)].iloc[0]
    matches = pd.read_parquet(ROOT / "data" / "processed" / "matches.parquet")
    match_rows = matches[(matches.clip_id == clip_id) & (matches.keyframe_spin_index == spin)]
    by_spin = {int(item.target_spin_index): item for item in match_rows.itertuples(index=False)}
    ordered_spins = [spin] + [int(value) for value in keyframe.history_spin_indices]
    labels = ["keyframe"] + [f"history {offset}" for offset in HISTORY_OFFSETS]

    features = load_features()
    ego_zip_path = local_path(features, chunk, "egomotion.offline")
    ego_member = features.get_clip_files_in_zip(clip_id, "egomotion.offline")["egomotion.offline"]
    with zipfile.ZipFile(ego_zip_path) as archive:
        ego = read_member(archive, ego_member)
    speed = _ego_speed(ego, int(keyframe.reference_timestamp))

    extrinsics = pd.read_parquet(local_path(features, chunk, "sensor_extrinsics.offline")).loc[clip_id]
    camera_to_rig = camera_to_ego_transform(extrinsics.loc["camera_front_wide_120fov"])
    manifest = pd.read_parquet(ROOT / "data" / "labels" / "labels_manifest.parquet")
    record = manifest[(manifest.clip_id == clip_id) & (manifest.spin_index == spin)].iloc[0]
    grid = _load_grid(record)
    dimensions = _dimensions(clip_id)

    image_paths = []
    for target_spin, label in zip(ordered_spins, labels):
        if target_spin not in by_spin:
            raise ValueError(f"missing match for {clip_id} spin {target_spin}")
        frame = int(by_spin[target_spin].frame_index)
        image_paths.append((label, ROOT / "data" / "processed" / "images" / clip_id / f"after_{frame:04d}.jpg"))

    photo = cv2.imread(str(image_paths[0][1]), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(image_paths[0][1])
    voxel_image = _voxel_overlay(photo, grid, camera_to_rig)
    occ = grid.occ == OCC
    top = occ.any(axis=2)
    side = occ.any(axis=1)
    x0, x1 = float(grid.origin[0]), float(grid.origin[0] + grid.occ.shape[0] * grid.voxel)
    y0, y1 = float(grid.origin[1]), float(grid.origin[1] + grid.occ.shape[1] * grid.voxel)
    z0, z1 = float(grid.origin[2]), float(grid.origin[2] + grid.occ.shape[2] * grid.voxel)

    info = {
        "clip": prefix,
        "split": str(row.split),
        "spin": int(spin),
        "ego_speed_mps": speed,
        "n_occ_voxels": int(np.count_nonzero(occ)),
        "vehicle_dimensions": {name: float(dimensions[name]) for name in ("length", "width", "height", "rear_axle_to_bbox_center")},
        "constants": {"VOXEL": VOXEL, "SHAPE": list(SHAPE), "ORIGIN": [float(v) for v in ORIGIN],
                      "OWN_VEHICLE_MARGIN": OWN_VEHICLE_MARGIN, "HFOV": HFOV_DEGREES, "history_offsets": HISTORY_OFFSETS,
                      "K": K.tolist()},
    }
    output = ROOT / "outputs" / "samples" / f"{prefix}_spin{spin:03d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.15))
    for column, (label, path) in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        ax = fig.add_subplot(grid_spec[0, column])
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")
    ax = fig.add_subplot(grid_spec[1, 0])
    ax.imshow(cv2.cvtColor(voxel_image, cv2.COLOR_BGR2RGB))
    ax.set_title("keyframe voxel dots")
    ax.axis("off")
    _grid_view(fig.add_subplot(grid_spec[1, 1]), top, (x0, x1, y0, y1), "top-down OCC", "rig X forward (m)", "rig Y left (m)")
    _grid_view(fig.add_subplot(grid_spec[1, 2]), side, (x0, x1, z0, z1), "side OCC", "rig X forward (m)", "rig Z up (m)")
    text_ax = fig.add_subplot(grid_spec[1, 3])
    text_ax.axis("off")
    text_ax.text(0.0, 1.0, json.dumps(info, indent=2), va="top", family="monospace", fontsize=8)
    fig.suptitle(f"PhysicalAI-AV sample: {prefix} spin {spin}")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(json.dumps({**info, "output": str(output.relative_to(ROOT))}, indent=2))
    return output, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_id_prefix")
    parser.add_argument("spin", type=int)
    args = parser.parse_args()
    make_sample(args.clip_id_prefix, args.spin)


if __name__ == "__main__":
    main()
