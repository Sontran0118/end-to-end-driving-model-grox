# `tools/build_info.py`

## Purpose and place

Provides append-only reproducibility metadata for successful label-building and rectification runs. It is a support module imported by `build_labels.py` and `rectify_camera.py`, not a data-processing stage by itself.

## Inputs and outputs

`current_config` imports the current constants from `build_labels.py`, `rectify_camera.py`, and `keyframes_matches.py`. The module reads Git metadata, installed package metadata, and `data/raw/manifest.json` when present. `append_build_info` appends an entry to `docs/build_info.json`.

## Run

There is no standalone CLI. A successful run of either instrumented pipeline calls:

```text
append_build_info("tools/build_labels.py")
append_build_info("tools/rectify_camera.py")
```

The initial file entry was manually marked `status: reconstructed`; the append hooks use `status: executed` by default.

## Walkthrough

`current_config` derives K from output dimensions and HFOV and reads voxel/grid/history constants. `_git_commit` runs `git rev-parse HEAD`. `_dataset_revision` reads the revision recorded in the local raw manifest. `_library_versions` queries package metadata and records `null` if a package is unavailable. `append_build_info` retains existing entries, appends the new object, and writes the JSON document.

## Math example

For width 1280 and HFOV 120 degrees, the focal length is `(1280/2)/tan(120°/2)≈369.504`; the recorded K uses that value with principal point `(640,192)`.

## Constants and conventions

Recorded fields are `VOXEL`, `SHAPE`, `ORIGIN`, `OWN_VEHICLE_MARGIN`, `HFOV`, history offsets, K, library versions, Git commit, UTC date, and dataset revision. The first date is the reconstruction record’s stored timestamp. A missing raw manifest means dataset revision is `null`; whether a future remote revision can be queried is **UNVERIFIED** by this support module.

## Checks and failure modes

The current JSON was parsed successfully with one reconstructed entry and revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`. Imports, malformed JSON, or filesystem/Git failures can stop an append. Existing entries are retained in order; the code does not deduplicate entries.

## Explain it back

1. **Where is metadata stored?** `docs/build_info.json`.
2. **Does a new run replace old entries?** No; it appends another entry.
3. **Where does the revision come from?** The local download manifest when available.
4. **What happens if a package is missing?** Its version is recorded as `null`.
5. **Does this module download data?** No.
