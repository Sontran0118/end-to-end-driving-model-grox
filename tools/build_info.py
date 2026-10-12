"""Append reproducibility metadata for successful pipeline runs."""

from __future__ import annotations

import datetime as dt
import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_INFO = ROOT / "docs" / "build_info.json"


def current_config() -> dict:
    """Read the current pipeline constants without running either pipeline."""
    labels = importlib.import_module("build_labels")
    rectify = importlib.import_module("rectify_camera")
    matches = importlib.import_module("keyframes_matches")
    fx = (rectify.OUTPUT_WIDTH / 2.0) / __import__("math").tan(
        __import__("math").radians(rectify.HFOV_DEGREES) / 2.0
    )
    return {
        "VOXEL": float(labels.VOXEL),
        "SHAPE": [int(v) for v in labels.SHAPE],
        "ORIGIN": [float(v) for v in labels.ORIGIN],
        "OWN_VEHICLE_MARGIN": float(labels.OWN_VEHICLE_MARGIN),
        "HFOV": float(rectify.HFOV_DEGREES),
        "history_offsets": [int(v) for v in matches.HISTORY_OFFSETS],
        "K": [[float(fx), 0.0, float(rectify.OUTPUT_WIDTH / 2.0)],
              [0.0, float(fx), float(rectify.OUTPUT_HEIGHT / 2.0)],
              [0.0, 0.0, 1.0]],
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _dataset_revision() -> str | None:
    manifest = ROOT / "data" / "raw" / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("revision")
    except (OSError, json.JSONDecodeError):
        return None


def _library_versions() -> dict[str, str | None]:
    names = [
        "physical_ai_av", "huggingface_hub", "pandas", "pyarrow", "numpy", "scipy",
        "opencv-python", "matplotlib", "ipykernel", "DracoPy",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def append_build_info(script: str, status: str = "executed") -> None:
    """Append one entry; existing entries are retained in order."""
    if BUILD_INFO.exists():
        data = json.loads(BUILD_INFO.read_text(encoding="utf-8"))
    else:
        data = {"schema_version": 1, "entries": []}
    if isinstance(data, list):
        data = {"schema_version": 1, "entries": data}
    data.setdefault("schema_version", 1)
    data.setdefault("entries", [])
    entry = {
        "script": script,
        "status": status,
        "git_commit": _git_commit(),
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        **current_config(),
        "library_versions": _library_versions(),
        "dataset_revision": _dataset_revision(),
    }
    data["entries"].append(entry)
    BUILD_INFO.parent.mkdir(parents=True, exist_ok=True)
    BUILD_INFO.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
