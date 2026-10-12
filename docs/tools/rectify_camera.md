# `tools/rectify_camera.py`

## Purpose and place

Rectifies selected front-wide f-theta video frames to a 1280x384 pinhole image. It follows camera matching and supplies the images used by alignment and label review.

## Inputs and outputs

Inputs are `clips.csv`, `data/processed/matches.parquet`, camera ZIP members, and each clip’s `camera_intrinsics.offline` Parquet row. Outputs are `data/processed/images/<clip_id>/`, `data/processed/K/<clip_id>.json`, `rectification_report.json`, and summary files.

## Run

```powershell
python tools/rectify_camera.py --clip 49eb6e65
python tools/rectify_camera.py --all
```

The CLI requires `--clip` or `--all`.

## Walkthrough

`load_intrinsics` reads the f-theta parameters. `pinhole_to_source_map` builds one inverse map: output pinhole ray to source angle, polynomial pixel radius, and source pixel. `cv2.remap` samples the source image. `process_clip` decodes selected indices sequentially once, writes JPEGs, and saves K/source-intrinsic JSON. `geometry_report` measures edge angles, corners, outside pixels, and the largest no-border HFOV.

## Math example

For output pixel `(u,v)`, the pinhole ray is proportional to `((u-cx)/fx,(v-cy)/fy,1)`. Its horizontal angle is `atan2(x,z)`; the f-theta polynomial then maps that angle to a source radius before the source principal point is added.

## Constants and conventions

`HFOV_DEGREES=120`, `OUTPUT_WIDTH=1280`, `OUTPUT_HEIGHT=384`, and `JPEG_QUALITY=95`. The shared output K is approximately `fx=fy=369.504`, `cx=640`, `cy=192`. Black remap borders are possible and are reported; this code does not silently crop them.

## Checks and failure modes

The all-clips path verifies 40 unique matched frames per clip, image existence, K JSON existence, total bytes, and brightness ranking. Missing f-theta/model fields, video members, or invalid ZIP access stops processing. `SeekVideoReader` seeking can be inexact, so frames are decoded sequentially for selected indices.

## Explain it back

1. **What is inverse mapping?** For each output pixel, find the source pixel that should supply it.
2. **Why use one K?** The chosen rectified pinhole geometry is shared across clips.
3. **What does HFOV change?** It changes the horizontal ray range and therefore focal length.
4. **What does the JSON preserve?** Output K, size, geometry report, and source intrinsics.
5. **Why decode once?** Repeated seeks can be inaccurate and would repeat work.
