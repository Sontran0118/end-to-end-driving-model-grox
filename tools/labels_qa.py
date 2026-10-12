"""Validate saved occupancy labels and make a 50-clip voxel-dot contact sheet."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labels_check import K, IMAGE_SIZE, dimensions_table, project_rectified_points
from occgrid import Grid, OCC
from viewer.grid import surface_mask

THUMBNAIL_SIZE = (320, 96)
TARGET_CLIP_COUNT = {"train": 35, "val": 7, "test": 8}


def validate_labels(manifest: pd.DataFrame) -> None:
    label_files = sorted((ROOT / "data" / "labels").rglob("*.npz"))
    failures: list[str] = []
    first_report = None
    for path in label_files:
        try:
            with np.load(path) as data:
                if first_report is None:
                    first_report = {
                        key: {"dtype": str(data[key].dtype), "shape": list(data[key].shape)}
                        for key in data.files
                    }
                Grid(data["occ"], data["origin"], float(data["voxel"]), pose=data["pose"])
        except Exception as exc:  # report every failure, continue validating the rest
            failures.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    missing = [str(path) for path in manifest.path if not (ROOT / path).exists()]
    print(f"npz_count={len(label_files)} grid_validation_failures={len(failures)}")
    if failures:
        print("validation_failures:")
        print("\n".join(failures))
    print(f"manifest_rows={len(manifest)} manifest_missing_paths={len(missing)}")
    if missing:
        print("missing_manifest_paths:")
        print("\n".join(missing[:20]))
    print(f"one_file={label_files[0].relative_to(ROOT)} keys_dtypes_shapes={first_report}")


def load_extrinsics(row: pd.Series) -> np.ndarray:
    directory = ROOT / "data" / "raw" / "calibration" / "sensor_extrinsics.offline"
    path = next(directory.glob(f"*chunk_{int(row.chunk):04d}.parquet"))
    table = pd.read_parquet(path)
    return table.loc[row.clip_id, "camera_front_wide_120fov"]


def make_voxel_dot_thumbnail(record: pd.Series, row: pd.Series, output: Path) -> np.ndarray:
    label_path = ROOT / record.path
    with np.load(label_path) as data:
        grid = Grid(data["occ"], data["origin"], float(data["voxel"]), pose=data["pose"])
    idx = np.argwhere(surface_mask(grid.occ))
    centers = grid.origin + (idx + 0.5) * grid.voxel
    camera_to_rig = load_extrinsics(row)
    # load_extrinsics returns a Series; the transform helper accepts its fields.
    from transforms import camera_to_ego_transform
    camera_to_rig_matrix = camera_to_ego_transform(camera_to_rig)
    camera, uv, valid = project_rectified_points(centers, camera_to_rig_matrix)
    valid &= camera[:, 2] > 0.5
    image_path = ROOT / "data" / "processed" / "images" / row.clip_id / f"after_{int(record.camera_frame_index):04d}.jpg"
    photo = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(image_path)
    overlay = photo.copy()
    colors = (plt.get_cmap("viridis")(
        np.clip((centers[:, 2] - float(grid.origin[2])) / (grid.occ.shape[2] * grid.voxel), 0.0, 1.0)
    )[:, :3] * 255).astype(np.uint8)
    for (u, v), color in zip(uv[valid], colors[valid]):
        cv2.circle(overlay, (int(round(u)), int(round(v))), 2,
                   (int(color[2]), int(color[1]), int(color[0])), -1, cv2.LINE_AA)
    blended = cv2.addWeighted(photo, 0.55, overlay, 0.45, 0.0)
    cv2.putText(blended, f"{row.clip_id[:8]}  n_occ={int(record.n_occ)}", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blended, f"{row.clip_id[:8]}  n_occ={int(record.n_occ)}", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    thumbnail = cv2.resize(blended, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    print(f"contact_clip={row.clip_id[:8]} spin={int(record.spin_index)} n_occ={int(record.n_occ)} "
          f"surface={len(idx)} projected_depth_gt_0p5={int(valid.sum())}")
    return thumbnail


def make_contact_sheet(manifest: pd.DataFrame, dims: pd.DataFrame, output: Path) -> None:
    thumbs = []
    for clip_id, group in manifest.groupby("clip_id", sort=True):
        group = group.sort_values("spin_index").reset_index(drop=True)
        record = group.iloc[len(group) // 2]
        row = dims[dims.clip_id == clip_id].iloc[0]
        thumbs.append(make_voxel_dot_thumbnail(record, row, output))
    rows = []
    blank = np.zeros((THUMBNAIL_SIZE[1], THUMBNAIL_SIZE[0], 3), dtype=np.uint8)
    for start in range(0, len(thumbs), 4):
        row = thumbs[start:start + 4]
        row += [blank] * (4 - len(row))
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "contact_sheet_voxel_dots.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"contact_sheet={path.relative_to(ROOT)} size={sheet.shape[1]}x{sheet.shape[0]}")


def main() -> None:
    manifest = pd.read_parquet(ROOT / "data" / "labels" / "labels_manifest.parquet")
    validate_labels(manifest)
    print("n_occ_overall=" + ",".join(f"{key}={value}" for key, value in manifest.n_occ.agg(["min", "median", "max"]).items()))
    print("smallest_n_occ:")
    print(manifest.nsmallest(10, "n_occ")[['clip_id', 'spin_index', 'n_occ']].assign(clip=lambda x: x.clip_id.str[:8]).drop(columns="clip_id").to_string(index=False))
    clip_counts = manifest.groupby("split").clip_id.nunique().to_dict()
    print(f"clip_counts_per_split={clip_counts} expected={TARGET_CLIP_COUNT}")
    make_contact_sheet(manifest, dimensions_table(), ROOT / "outputs" / "labels_qa")


if __name__ == "__main__":
    main()
