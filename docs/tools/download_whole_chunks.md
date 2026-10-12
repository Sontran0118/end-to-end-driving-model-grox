# `tools/download_whole_chunks.py`

## Purpose and place

Downloads the explicit whole-chunk fallback file list for the selected clips. It is the data-acquisition stage and does not extract archives.

## Inputs and outputs

Reads `clips.csv` and `data/metadata/features.csv`. It downloads the listed ZIP and Parquet paths into `data/raw/`, verifies byte sizes and selected ZIP members, and writes `data/raw/manifest.json`. The requested feature names are LiDAR, front-wide camera, offline egomotion, camera intrinsics, offline camera intrinsics, sensor extrinsics, offline sensor extrinsics, and offline LiDAR intrinsics.

## Run

```powershell
python tools/download_whole_chunks.py
```

The dataset repository is `nvidia/PhysicalAI-Autonomous-Vehicles`; the script uses the devkit interface with `cache_dir=None`. Authentication and license acceptance are prerequisites; no token belongs in this document.

## Walkthrough

`main` finds unique chunks, reads `features.csv`, builds the explicit feature/chunk path list, queries expected sizes, downloads one file at a time, checks exact local byte counts, opens key ZIPs, validates selected clip members, and writes the manifest. A temporary expected-size file is removed at the end by the script.

## Math example

If three chunks are selected and eight feature families are requested, the explicit list has `3*8=24` paths before any feature-specific path expansion. The exact list comes from the feature table, not a guessed folder glob.

## Constants and conventions

The dataset repository, raw directory, feature list, and 120 GiB confirmation threshold are constants. `cache_dir=None` is intended to avoid a second Hugging Face cache copy. The script itself does not include `vehicle_dimensions`, so that feature is a separate scope decision.

## Checks and failure modes

Missing `features.csv`, missing files, byte-size mismatches, missing selected members, or ZIP-open errors stop the run. Network/authentication errors also stop it. The script assumes the explicit clip/chunk selection is already correct; it does not silently discover or download the whole dataset.

## Explain it back

1. **What is downloaded?** Only the feature/chunk paths built by the script.
2. **Are ZIPs extracted?** No.
3. **How is duplication reduced?** Downloads use `cache_dir=None` and write to `data/raw/`.
4. **What proves a file is complete?** Its local byte size equals the API-reported size.
5. **Where is the download record?** `data/raw/manifest.json`.
