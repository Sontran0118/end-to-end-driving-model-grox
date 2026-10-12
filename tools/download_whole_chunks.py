import json
import pathlib
import zipfile

import pandas as pd
from huggingface_hub import get_token
from physical_ai_av.utils.hf_interface import HfRepoInterface


REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
METADATA = ROOT / "data" / "metadata"
CLIPS = ROOT / "clips.csv"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(CLIPS)
    chunks = sorted({int(value) for value in clips["chunk"]})
    features = pd.read_csv(METADATA / "features.csv").set_index("feature")

    requested = [
        "lidar_top_360fov",
        "camera_front_wide_120fov",
        "egomotion.offline",
        "camera_intrinsics",
        "camera_intrinsics.offline",
        "sensor_extrinsics",
        "sensor_extrinsics.offline",
        "lidar_intrinsics.offline",
    ]
    files = []
    path_features = {}
    for feature in requested:
        template = features.loc[feature, "chunk_path"]
        for chunk in chunks:
            remote_path = template.format(chunk_id=chunk)
            files.append(remote_path)
            path_features[remote_path] = feature

    interface = HfRepoInterface(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=get_token(),
        local_dir=RAW,
        cache_dir=None,
        confirm_download_threshold_gb=120.0,
    )
    path_info = interface.api.get_paths_info(
        paths=files, repo_id=REPO_ID, repo_type=REPO_TYPE, revision=interface.revision
    )
    expected = {item.path: item.size for item in path_info}
    (RAW / "expected_sizes.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"revision={interface.revision}", flush=True)
    print(f"files={len(files)}", flush=True)
    print(f"expected_bytes={sum(expected.values())}", flush=True)
    print(f"local_dir={RAW}", flush=True)
    print("cache_dir=None; local_dir mode avoids the global Hugging Face blob cache", flush=True)

    # Use the Hub API's normal progress implementation. The devkit wrapper's
    # aggregated progress class is incompatible with the installed Xet client.
    for remote_path in files:
        print(f"downloading={remote_path}", flush=True)
        interface.api.hf_hub_download(
            filename=remote_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=interface.revision,
            token=get_token(),
            local_dir=RAW,
            cache_dir=None,
            force_download=False,
        )

    missing = []
    wrong_sizes = []
    manifest = []
    for remote_path in files:
        local_path = RAW / remote_path
        if not local_path.exists():
            missing.append(remote_path)
            continue
        actual = local_path.stat().st_size
        if actual != expected[remote_path]:
            wrong_sizes.append((remote_path, expected[remote_path], actual))
        manifest.append(
            {
                "file": remote_path,
                "bytes": actual,
                "expected_bytes": expected[remote_path],
                "features": [path_features[remote_path]],
            }
        )

    if missing or wrong_sizes:
        raise RuntimeError(f"missing={missing}; wrong_sizes={wrong_sizes}")

    selected = set(clips["clip_id"].astype(str))
    zip_checks = []
    for feature in ["lidar_top_360fov", "camera_front_wide_120fov", "egomotion.offline"]:
        templates = json.loads(features.loc[feature, "clip_files_in_zip"])
        for chunk in chunks:
            remote_path = features.loc[feature, "chunk_path"].format(chunk_id=chunk)
            with zipfile.ZipFile(RAW / remote_path) as archive:
                names = set(archive.namelist())
                for clip_id in clips.loc[clips["chunk"] == chunk, "clip_id"].astype(str):
                    for template in templates.values():
                        member = template.format(clip_id=clip_id)
                        if member not in names:
                            raise RuntimeError(f"missing ZIP member: {remote_path}:{member}")
                zip_checks.append({"file": remote_path, "opens": True})

    (RAW / "manifest.json").write_text(
        json.dumps(
            {
                "repo": REPO_ID,
                "revision": interface.revision,
                "files": manifest,
                "zip_checks": zip_checks,
                "selected_clip_count": len(selected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RAW / "expected_sizes.json").unlink()
    print("verification=passed", flush=True)
    print("manifest=data/raw/manifest.json", flush=True)


if __name__ == "__main__":
    main()
