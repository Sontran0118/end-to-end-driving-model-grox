# `tools/keyframes_matches.py`

## Purpose and place

Creates LiDAR keyframes and matches each keyframe/history timestamp to a front-wide camera frame. It is the temporal-indexing stage before rectification and labels.

## Inputs and outputs

It reads `clips.csv`, metadata features, and LiDAR/camera ZIP members from `data/raw/`. With `--write`, it writes `data/processed/keyframes.parquet`, `matches.parquet`, and a notes JSON.

## Run

```powershell
python tools/keyframes_matches.py --all --write
python tools/keyframes_matches.py --clip 13ae70ee --write
```

The CLI requires either `--all` or one or more `--clip` values, and `--write` enables output writing.

## Walkthrough

`load_features` parses the feature table and ZIP-member mapping. `read_member` reads Parquet bytes inside a ZIP. `process_clip` validates timestamps, sorts spins, selects every fifth eligible spin, and records history offsets. `nearest_frame` checks the two `searchsorted` neighbors. `run` caches open ZIP handles and `write_outputs` serializes results.

## Math example

For camera times `[100,140,190]` and target `165`, `searchsorted` checks 140 and 190; both differ by 25, so the lower frame index wins. The stored difference is in microseconds.

## Constants and conventions

`KEYFRAME_STRIDE=5`; history offsets are `[-5,-10,-15]`. The first eligible keyframe is spin 15 because all three earlier indices are needed. `spin_start_timestamp` is used as the LiDAR reference timestamp. Streams are treated as sharing a clip-relative clock; camera frame phase is **UNVERIFIED**.

## Checks and failure modes

Missing required timestamp columns or members raises an error. Non-monotonic camera timestamps are sorted. Recorded output was 1,850 keyframes and 7,400 matches for 50 clips. `make_zip_map` is defined but unused by the run path.

## Explain it back

1. **Why skip early spins?** They lack complete history.
2. **How often is a keyframe selected?** Every fifth eligible spin.
3. **What are history rows?** The three earlier configured spin timestamps.
4. **How is a camera frame selected?** Nearest timestamp, checking both neighbors.
5. **What does `time_difference_us` measure?** Absolute target-camera timestamp difference in microseconds.
