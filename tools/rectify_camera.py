"""Decode matched camera frames and rectify f-theta images to pinhole.

This is an independent implementation.  It is informed by the public
Cosmos-Drive-Dreams rectification concept (ray grid -> f-theta projection ->
image sampling), but does not copy its source code.  See the attribution
record written with each run.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from physical_ai_av.dataset import Features
from physical_ai_av.video import SeekVideoReader


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CLIPS = ROOT / "clips.csv"
MATCHES = PROCESSED / "matches.parquet"
FEATURES_CSV = ROOT / "data" / "metadata" / "features.csv"

CAMERA_NAME = "camera_front_wide_120fov"
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 384
HFOV_DEGREES = 120.0
JPEG_QUALITY = 95


def load_features() -> Features:
    df = pd.read_csv(FEATURES_CSV, index_col="feature")
    df["clip_files_in_zip"] = df["clip_files_in_zip"].map(json.loads, na_action="ignore")
    return Features(df)


def camera_zip_path(features: Features, chunk: int) -> Path:
    return RAW / Path(features.get_chunk_feature_filename(chunk, CAMERA_NAME))


def load_intrinsics(chunk: int, clip_id: str) -> tuple[dict, dict]:
    path = RAW / "calibration" / "camera_intrinsics.offline" / (
        f"camera_intrinsics.offline.chunk_{chunk:04d}.parquet"
    )
    df = pd.read_parquet(path)
    row = df.loc[(clip_id, CAMERA_NAME)]
    model_type = str(row["model_type"])
    params = json.loads(row["model_parameters"])
    if model_type != "ftheta":
        raise ValueError(f"{clip_id}: expected ftheta, got {model_type}")
    return {"model_type": model_type, **params}, params


def polyval(coeffs: list[float], x: np.ndarray | float) -> np.ndarray:
    return np.polynomial.polynomial.polyval(x, np.asarray(coeffs, dtype=np.float64))


def model_matrices(source: dict) -> tuple[np.ndarray, np.ndarray]:
    c, d, e = [float(v) for v in source.get("linear_cde", [1.0, 0.0, 0.0])]
    a = np.array([[c, d], [e, 1.0]], dtype=np.float64)
    return a, np.linalg.inv(a)


def source_pixel_to_angle(source: dict, pixels: np.ndarray) -> np.ndarray:
    center = np.asarray(source["principal_point"], dtype=np.float64)
    _, inv_a = model_matrices(source)
    transformed = (inv_a @ (pixels - center).T).T
    radius = np.linalg.norm(transformed, axis=1)
    return polyval(source["pixeldist_to_angle_poly"], radius)


def pinhole_to_source_map(source: dict, hfov_degrees: float) -> tuple[np.ndarray, np.ndarray, dict]:
    width, height = OUTPUT_WIDTH, OUTPUT_HEIGHT
    hfov = np.deg2rad(hfov_degrees)
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    xy_norm = np.sqrt(x * x + y * y)
    angle = np.arctan2(xy_norm, z)
    radius = polyval(source["angle_to_pixeldist_poly"], angle)
    safe_norm = np.where(xy_norm > 0, xy_norm, 1.0)
    radial = np.stack([radius * x / safe_norm, radius * y / safe_norm], axis=-1)
    a, _ = model_matrices(source)
    source_offsets = radial @ a.T
    center = np.asarray(source["principal_point"], dtype=np.float64)
    source_xy = source_offsets + center
    map_x = source_xy[..., 0].astype(np.float32)
    map_y = source_xy[..., 1].astype(np.float32)
    outside = (
        (source_xy[..., 0] < 0)
        | (source_xy[..., 0] > float(source["resolution"][0] - 1))
        | (source_xy[..., 1] < 0)
        | (source_xy[..., 1] > float(source["resolution"][1] - 1))
    )
    return map_x, map_y, {
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "horizontal_fov_degrees": float(hfov_degrees),
        "vertical_fov_degrees": float(np.rad2deg(2 * np.arctan((height / 2.0) / fy))),
        "outside_fraction": float(outside.mean()),
        "outside_pixels": int(outside.sum()),
    }


def largest_hfov_without_borders(source: dict) -> float:
    lo, hi = 0.0, 180.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        _, _, stats = pinhole_to_source_map(source, mid)
        if stats["outside_pixels"] == 0:
            lo = mid
        else:
            hi = mid
    return lo


def geometry_report(source: dict) -> dict:
    w, h = source["resolution"]
    cx, cy = source["principal_point"]
    edge_points = np.array([[0.0, cy], [float(w - 1), cy]], dtype=np.float64)
    corner_points = np.array([[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]])
    edge_angles = np.rad2deg(source_pixel_to_angle(source, edge_points))
    corner_angles = np.rad2deg(source_pixel_to_angle(source, corner_points))
    _, _, output_stats = pinhole_to_source_map(source, HFOV_DEGREES)
    center_map = pinhole_to_source_map(source, HFOV_DEGREES)[2]
    center_ray_source = np.asarray(source["principal_point"], dtype=np.float64)
    center_error = float(np.linalg.norm(center_ray_source - np.asarray([cx, cy])))
    return {
        "source_left_edge_angle_degrees": float(edge_angles[0]),
        "source_right_edge_angle_degrees": float(edge_angles[1]),
        "source_corner_angles_degrees": [float(v) for v in corner_angles],
        "source_corner_angle_max_degrees": float(corner_angles.max()),
        "output_at_hfov_120": output_stats,
        "largest_hfov_without_borders_degrees": largest_hfov_without_borders(source),
        "principal_ray_source_pixel": [float(v) for v in center_ray_source],
        "principal_ray_center_error_pixels": center_error,
        "source_resolution": [int(w), int(h)],
        "source_principal_point": [float(cx), float(cy)],
    }


def rectify_image(image_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def process_clip(clip_row: pd.Series, features: Features, matches: pd.DataFrame) -> dict:
    clip_id = str(clip_row.clip_id)
    chunk = int(clip_row.chunk)
    source, source_params = load_intrinsics(chunk, clip_id)
    map_x, map_y, output_stats = pinhole_to_source_map(source, HFOV_DEGREES)
    report = geometry_report(source)
    if report["principal_ray_center_error_pixels"] > 1e-9:
        raise AssertionError("Principal-point ray did not map to the source principal point")

    clip_matches = matches[matches.clip_id == clip_id]
    frame_indices = np.sort(clip_matches.frame_index.astype(np.int64).unique())
    member = features.get_clip_files_in_zip(clip_id, CAMERA_NAME)["video"]
    with zipfile.ZipFile(camera_zip_path(features, chunk), "r") as zf:
        video_bytes = zf.read(member)
    reader = SeekVideoReader(io.BytesIO(video_bytes), thread_count=1)
    try:
        images = reader.decode_images_from_frame_indices(frame_indices)
    finally:
        reader.close()

    image_dir = PROCESSED / "images" / clip_id
    image_dir.mkdir(parents=True, exist_ok=True)
    brightness_values = []
    for frame_index, image_rgb in zip(frame_indices, images):
        rectified = rectify_image(image_rgb, map_x, map_y)
        cv2.imwrite(str(image_dir / f"before_{int(frame_index):04d}.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        cv2.imwrite(str(image_dir / f"after_{int(frame_index):04d}.jpg"), cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        brightness_values.append(float(rectified.mean()))

    k_path = PROCESSED / "K" / f"{clip_id}.json"
    k_path.parent.mkdir(parents=True, exist_ok=True)
    k_path.write_text(json.dumps({
        "clip_id": clip_id,
        "camera_name": CAMERA_NAME,
        "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "K": output_stats["K"],
        "source_intrinsics": {"model_type": source["model_type"], **source_params},
        "geometry": report,
        "jpeg_quality": JPEG_QUALITY,
        "cosmos_drive_dreams_reference": {
            "repository": "https://github.com/nv-tlabs/Cosmos-Drive-Dreams",
            "license": "Apache-2.0",
            "borrowed": "Algorithmic idea only: construct a pinhole ray grid, project rays through the f-theta model, and sample the source image. No source code copied.",
        },
    }, indent=2), encoding="utf-8")
    return {
        "clip_id": clip_id,
        "frames": int(len(frame_indices)),
        "bytes": int(sum(p.stat().st_size for p in image_dir.glob("*.jpg"))),
        "mean_brightness": float(np.mean(brightness_values)),
        "geometry": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clip")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    features = load_features()
    clips = pd.read_csv(CLIPS, dtype={"clip_id": str})
    matches = pd.read_parquet(MATCHES)
    if args.clip:
        row = clips.loc[clips.clip_id == args.clip].iloc[0]
        result = process_clip(row, features, matches)
        print(json.dumps(result, indent=2))
        return

    results = [process_clip(row, features, matches) for _, row in clips.iterrows()]
    results = sorted(results, key=lambda item: item["mean_brightness"])
    matches_by_clip = matches.groupby("clip_id")["frame_index"].nunique().to_dict()
    k_files = {p.stem for p in (PROCESSED / "K").glob("*.json")}
    expected_ids = set(clips.clip_id)
    image_files_by_clip = {
        clip_id: sorted((PROCESSED / "images" / clip_id).glob("*.jpg"))
        for clip_id in expected_ids
    }
    verification = {
        "clips": len(results),
        "all_have_40_unique_match_frames": all(matches_by_clip.get(cid) == 40 for cid in expected_ids),
        "total_unique_match_frames": int(sum(matches_by_clip.values())),
        "all_match_frames_have_images": all(
            all((PROCESSED / "images" / cid / f"before_{int(frame):04d}.jpg").exists()
                and (PROCESSED / "images" / cid / f"after_{int(frame):04d}.jpg").exists()
                for frame in matches.loc[matches.clip_id == cid, "frame_index"].unique())
            for cid in expected_ids
        ),
        "all_clips_have_K_json": expected_ids == k_files,
        "total_jpeg_bytes": int(sum(p.stat().st_size for files in image_files_by_clip.values() for p in files)),
    }
    darkest = results[:10]
    pair_clips = [darkest[0]["clip_id"], results[len(results) // 2]["clip_id"], results[-1]["clip_id"]]
    report = {
        "hfov_degrees": HFOV_DEGREES,
        "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "per_clip_mean_brightness": results,
        "darkest_10": darkest,
        "three_before_after_pairs": [
            {
                "clip_id": cid,
                "before": str(next(iter((PROCESSED / "images" / cid).glob("before_*.jpg"))).relative_to(ROOT)),
                "after": str(next(iter((PROCESSED / "images" / cid).glob("after_*.jpg"))).relative_to(ROOT)),
            }
            for cid in pair_clips
        ],
        "verification": verification,
    }
    (PROCESSED / "rectification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
