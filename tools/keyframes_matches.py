"""Create LiDAR keyframes and camera timestamp matches from local ZIP members.

The ZIP archives are read in place; no archive is extracted.  The devkit's
Features class supplies the repository file/member naming rules.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from physical_ai_av.dataset import Features


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CLIPS = ROOT / "clips.csv"
FEATURES_CSV = ROOT / "data" / "metadata" / "features.csv"

KEYFRAME_STRIDE = 5
HISTORY_OFFSETS = [-5, -10, -15]


def load_features() -> Features:
    features_df = pd.read_csv(FEATURES_CSV, index_col="feature")
    features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
        json.loads, na_action="ignore"
    )
    return Features(features_df)


def local_feature_path(features: Features, chunk: int, feature: str) -> Path:
    return RAW / Path(features.get_chunk_feature_filename(chunk, feature))


def read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(zf.read(member)))


def make_zip_map(features: Features, chunks: list[int]) -> dict[tuple[int, str], zipfile.ZipFile]:
    # Kept for type clarity; actual ZIP handles are opened by process_clip.
    return {
        (chunk, feature): zipfile.ZipFile(local_feature_path(features, chunk, feature), "r")
        for chunk in chunks
        for feature in ("lidar_top_360fov", "camera_front_wide_120fov")
    }


def process_clip(
    row: pd.Series,
    features: Features,
    lidar_zf: zipfile.ZipFile,
    camera_zf: zipfile.ZipFile,
) -> tuple[list[dict], list[dict], dict]:
    clip_id = str(row.clip_id)
    split = str(row.split)
    chunk = int(row.chunk)

    lidar_member = features.get_clip_files_in_zip(clip_id, "lidar_top_360fov")["pointclouds"]
    lidar = read_member(lidar_zf, lidar_member)
    required = {"spin_start_timestamp", "spin_end_timestamp"}
    missing = required - set(lidar.columns)
    if missing:
        raise ValueError(f"{clip_id}: LiDAR schema missing {sorted(missing)}")
    lidar = lidar.sort_values("spin_start_timestamp", kind="stable").reset_index(drop=True)
    # The downloaded schema does not contain a literal reference_timestamp;
    # spin_start_timestamp is the start-of-sweep reference timestamp.
    ref_ts = lidar["spin_start_timestamp"].astype("int64").to_numpy()

    camera_member = features.get_clip_files_in_zip(
        clip_id, "camera_front_wide_120fov"
    )["frame_timestamps"]
    camera = read_member(camera_zf, camera_member)
    if "timestamp" not in camera.columns:
        raise ValueError(f"{clip_id}: camera timestamp schema missing timestamp")
    camera_ts = camera["timestamp"].astype("int64").to_numpy()
    if not np.all(camera_ts[1:] >= camera_ts[:-1]):
        order = np.argsort(camera_ts, kind="stable")
        camera_ts = camera_ts[order]

    keyframes: list[dict] = []
    matches: list[dict] = []
    first_keyframe = None
    for spin_index in range(0, len(ref_ts), KEYFRAME_STRIDE):
        if any(spin_index + offset < 0 for offset in HISTORY_OFFSETS):
            continue
        if first_keyframe is None:
            first_keyframe = spin_index
        history_indices = [spin_index + offset for offset in HISTORY_OFFSETS]
        history_timestamps = [int(ref_ts[i]) for i in history_indices]
        keyframes.append(
            {
                "clip_id": clip_id,
                "split": split,
                "spin_index": spin_index,
                "reference_timestamp": int(ref_ts[spin_index]),
                "history_spin_indices": history_indices,
                "history_timestamps": history_timestamps,
            }
        )
        targets = [("keyframe", spin_index, int(ref_ts[spin_index]))] + [
            ("history", i, int(ref_ts[i])) for i in history_indices
        ]
        for target_kind, target_spin, target_ts in targets:
            pos = int(np.searchsorted(camera_ts, target_ts, side="left"))
            candidates = [i for i in (pos - 1, pos) if 0 <= i < len(camera_ts)]
            frame_index = min(candidates, key=lambda i: (abs(int(camera_ts[i]) - target_ts), i))
            matches.append(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "keyframe_spin_index": spin_index,
                    "target_kind": target_kind,
                    "target_spin_index": target_spin,
                    "target_timestamp": target_ts,
                    "frame_index": frame_index,
                    "camera_timestamp": int(camera_ts[frame_index]),
                    "time_difference_us": abs(int(camera_ts[frame_index]) - target_ts),
                }
            )
    return keyframes, matches, {
        "clip_id": clip_id,
        "chunk": chunk,
        "lidar_rows": len(lidar),
        "camera_frames": len(camera_ts),
        "first_keyframe_spin_index": first_keyframe,
        "keyframe_count": len(keyframes),
    }


def run(clip_ids: list[str], write_outputs: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    features = load_features()
    clips = pd.read_csv(CLIPS, dtype={"clip_id": str})
    clips = clips[clips["clip_id"].isin(clip_ids)].copy()
    if len(clips) != len(clip_ids):
        raise ValueError("One or more requested clip IDs are not in clips.csv")
    zip_handles: dict[tuple[int, str], zipfile.ZipFile] = {}
    all_keyframes: list[dict] = []
    all_matches: list[dict] = []
    diagnostics: list[dict] = []
    try:
        for row in clips.itertuples(index=False):
            key = (int(row.chunk), "lidar_top_360fov")
            if key not in zip_handles:
                zip_handles[key] = zipfile.ZipFile(local_feature_path(features, key[0], key[1]), "r")
            key = (int(row.chunk), "camera_front_wide_120fov")
            if key not in zip_handles:
                zip_handles[key] = zipfile.ZipFile(local_feature_path(features, key[0], key[1]), "r")
            k, m, d = process_clip(
                pd.Series(row._asdict()),
                features,
                zip_handles[(int(row.chunk), "lidar_top_360fov")],
                zip_handles[(int(row.chunk), "camera_front_wide_120fov")],
            )
            all_keyframes.extend(k)
            all_matches.extend(m)
            diagnostics.append(d)
    finally:
        for zf in zip_handles.values():
            zf.close()

    keyframes_df = pd.DataFrame(all_keyframes)
    matches_df = pd.DataFrame(all_matches)
    if write_outputs:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        keyframes_df.to_parquet(PROCESSED / "keyframes.parquet", index=False)
        matches_df.to_parquet(PROCESSED / "matches.parquet", index=False)
        notes = {
            "keyframe_stride_spins": KEYFRAME_STRIDE,
            "history_offsets_spins": HISTORY_OFFSETS,
            "first_keyframe_rule": "First eligible zero-based sorted spin index is 15: indices 0, 5, and 10 cannot provide all three negative history offsets.",
            "lidar_timestamp_source": "spin_start_timestamp, used as reference_timestamp because the downloaded parquet schema has no literal reference_timestamp column.",
            "camera_frame_index_source": "zero-based positional row index in the camera timestamps parquet; the file has timestamp but no frame_index column.",
            "camera_matching": "numpy.searchsorted with both neighboring candidates checked; ties choose the lower frame index.",
            "wiki_timestamp_facts": "Wiki says all streams share a common clip-relative clock and LiDAR reference timestamps are scan starts. It does not specify whether camera timestamps denote frame start, middle, or end.",
            "clips": diagnostics,
        }
        (PROCESSED / "keyframe_matches_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return keyframes_df, matches_df, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="append", dest="clips")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    clips = pd.read_csv(CLIPS, dtype={"clip_id": str})
    if args.all:
        clip_ids = clips.clip_id.tolist()
    elif args.clips:
        clip_ids = args.clips
    else:
        raise SystemExit("Choose --clip CLIP_ID or --all")
    keyframes, matches, diagnostics = run(clip_ids, args.write)
    diffs = matches.time_difference_us.to_numpy()
    print(json.dumps({
        "clips": len(clip_ids),
        "keyframes": len(keyframes),
        "matches": len(matches),
        "first_keyframe_spin_index": int(diagnostics[0]["first_keyframe_spin_index"]),
        "keyframes_per_clip": sorted({d["keyframe_count"] for d in diagnostics}),
        "mean_difference_us": float(diffs.mean()),
        "max_difference_us": int(diffs.max()),
        "over_20ms": int((diffs > 20_000).sum()),
        "written": args.write,
    }, indent=2))


if __name__ == "__main__":
    main()
