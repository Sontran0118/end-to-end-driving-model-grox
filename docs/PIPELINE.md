# PhysicalAI-AV project pipeline

This project turns selected camera frames and LiDAR returns into sparse 3-D occupancy labels. The camera supplies the image; LiDAR supplies the measured returns used as the label source.

```text
metadata tables + devkit
        |
        v
selected clips -> keyframes + camera matches -> rectified photos
        |                                      |
        +-> calibrated, time-compensated LiDAR-+
                         |
                         v
             voxelization + vehicle filtering
                         |
                         v
                 OCC occupancy grids
```

## Six project steps

| Step | Script | Main inputs | Main outputs |
|---|---|---|---|
| 1. Select data | `tools/download_whole_chunks.py` plus the recorded clip-selection work | `clips.csv`, metadata, `features.csv`, local devkit | selected chunk files in `data/raw/`, `data/raw/manifest.json` |
| 2. Keyframes | `tools/keyframes_matches.py` | LiDAR and camera ZIP members | `data/processed/keyframes.parquet`, `data/processed/matches.parquet` |
| 3. Camera preparation | `tools/rectify_camera.py`, `tools/make_contact_sheet.py` | camera video, timestamps, offline intrinsics, matches | rectified JPEGs, per-clip K JSON, rectification reports and contact sheet |
| 4. Coordinate and alignment checks | `tools/transforms.py`, `tools/test_transforms.py`, `tools/lidar_alignment.py`, `tools/overlay_checks.py`, `tools/timing_check.py` | calibration, egomotion, LiDAR, rectified photos | transform tests and review overlays under `outputs/` |
| 5. Vehicle-box proposal | `tools/vehicle_box_proposal.py` | vehicle dimensions, LiDAR, extrinsics | proposal plots and inside/shell counts |
| 6. Occupancy labels and QA | `tools/build_labels.py`, `tools/labels_check.py`, `tools/labels_qa.py` | keyframes, matches, LiDAR, egomotion, calibration, vehicle dimensions | `data/labels/`, label manifest, QA plots/contact sheet |

## Conventions

- Rig axes are X forward, Y left, Z up. The exact physical placement of the rig origin relative to the car body and road is **UNVERIFIED** in the public material used here.
- The camera helper uses OpenCV axes: X right, Y down, Z forward. LiDAR uses a Cartesian sensor frame.
- Quaternions are ordered `(qx, qy, qz, qw)`. SciPy applies the active rotation convention.
- `T_target_from_source` maps coordinates expressed in `source` into `target`.
- Offline sensor extrinsics are used as sensor-to-rig transforms (`T_ego_from_sensor`). Egomotion poses are rig/ego-to-clip-local-anchor (`T_anchor_from_ego`). These meanings are stated in `tools/transforms.py`.
- Translations are metres and timestamps are integer microseconds on the clip-relative common clock. The keyframe code treats `spin_start_timestamp` as the LiDAR reference timestamp, so it represents scan start in this pipeline. Camera frame phase (start, middle, or end) remains **UNVERIFIED**.
- Label-grid constants are `VOXEL=0.4`, `SHAPE=(200,100,16)`, and `ORIGIN=(0,-20,-1)` in the rig frame. The own-vehicle margin is `0.20 m`. The grid therefore spans 80 m in X, 40 m in Y, and 6.4 m in Z; its top coordinate is `z=5.4 m` in the rig frame.

## Key decisions

- Whole chunks were used because the fallback download path obtains the listed ZIPs and Parquets once in `data/raw/`; the script does not extract whole archives. Individual-member saving was not used for the final download.
- Only `camera_front_wide_120fov` is rectified and matched. Rectification uses HFOV 120 degrees and one shared pinhole K (`fx=fy≈369.504`, `cx=640`, `cy=192`) for 1280x384 output.
- LiDAR points are motion-compensated to camera-frame time using interpolated translation and rotation. Translation is linear, rotation uses slerp, and extrapolation is rejected.
- The vehicle box is built per clip from `vehicle_dimensions` and expanded by 0.20 m on all sides. The margin is a conservative separation allowance; its physical adequacy is not independently validated.
- Labels are OCC-only sparse-return labels. Unobserved cells remain `UNKNOWN`; the pipeline does not ray-cast FREE space. `FREE` is imported in the label builder but not assigned.

## Verification record

Measured by code:

- Transform tests cover inverse identity, orthonormal rotations with determinant 1, a known point, exact-pose interpolation, and midpoint interpolation.
- Keyframe/match generation recorded 1,850 keyframes and 7,400 target matches for 50 clips.
- Rectification decoded 40 unique frames per clip, 2,000 total, and wrote a per-clip K JSON.
- Label QA loaded 1,850 NPZ files through `Grid` with zero recorded validation failures; the manifest had 1,850 rows and zero missing paths before the later manifest-column clarification.
- The label QA run recorded NPZ keys, dtypes, shapes, split counts of 35/7/8 clips, and a 50-clip voxel-dot contact sheet.

Reviewed by eye:

- LiDAR alignment overlays and the highway-sign A/B timing check were saved for visual review; the code does not claim that alignment is correct.
- Three checkpoint clips were used for label overlay review. The user was asked to judge the images, so visual correctness is not a measured result.

## Known limitations and next steps

- There is no FREE class or ray casting; UNKNOWN includes space not hit by a retained return.
- A single LiDAR spin can leave striping and does not describe all surfaces at one instant.
- The configured grid ceiling is 5.4 m above its origin and comes from viewer tests, not a confirmed teammate specification.
- Front-of-vehicle returns can remain near the conservative box boundary; shell counts were recorded rather than used to alter labels.
- Only 3 of 50 clips were visually checked in the alignment/label review workflow. Per-clip calibration quality was not independently certified.
- The current code and manifest disagree after the manifest clarification: the code in `tools/build_labels.py` still writes `n_occ`, while the saved manifest uses `n_points_in_grid` and `n_occ_voxels`. `tools/labels_qa.py` still expects `n_occ`. This is an audit item, not silently repaired here.

## Reproduce from a fresh clone

The exact script entry points are:

```powershell
python tools/download_whole_chunks.py
python tools/keyframes_matches.py --all --write
python tools/rectify_camera.py --all
python tools/test_transforms.py
python tools/lidar_alignment.py
python tools/overlay_checks.py
python tools/timing_check.py
python tools/vehicle_box_proposal.py
python tools/build_labels.py
python tools/labels_check.py
python tools/labels_qa.py
python tools/make_contact_sheet.py
```

The first command requires the dataset license and Hugging Face authentication; never place the token in a file or command transcript. The scripts assume the project dependencies and local devkit are already installed. The precise installation command/version set is **UNVERIFIED from these scripts**; use the project setup instructions rather than guessing a version.

## Glossary

- **Clip:** one recorded sequence identified by `clip_id`.
- **Chunk:** the archive grouping containing files for many clips.
- **Spin:** one LiDAR sweep, represented by a spin index and timestamps.
- **Keyframe:** every fifth eligible LiDAR spin; the first eligible index is 15 because three earlier history spins are required.
- **History frame:** one of the three earlier spins at configured offsets `[-5,-10,-15]`.
- **Rig:** the vehicle-fixed coordinate frame joining sensors.
- **Extrinsics:** sensor-to-rig calibration transforms.
- **Egomotion:** time-varying rig-to-local-anchor poses.
- **Rectification:** remapping a wide-angle image into a pinhole image.
- **f-theta:** the camera model whose polynomial maps ray angle to pixel radius in the source image; the exact source parameter units are supplied by the model data.
- **Voxel:** a rectangular 3-D cell addressed by integer `(ix,iy,iz)`.
- **OCC/FREE/UNKNOWN:** viewer states 2/1/0. This pipeline writes OCC and leaves other cells UNKNOWN.
- **Motion compensation:** transforming returns from their individual LiDAR times to one camera-frame time.

## Fifteen understanding checks

1. **What identifies a clip?** `clip_id`; its chunk is obtained from the metadata/devkit mapping.
2. **Why read ZIP members instead of extracting archives?** It avoids whole-archive extraction while retaining random member access.
3. **Which LiDAR timestamp is used as reference?** `spin_start_timestamp`.
4. **Why does the first keyframe start at 15?** Offsets -5, -10, and -15 must all have valid earlier spins.
5. **How are camera matches found?** `searchsorted` checks the two neighboring timestamps and chooses the smallest absolute difference.
6. **What does the rectification K describe?** The 1280x384 pinhole output image.
7. **What does an extrinsic transform do?** It maps a sensor point into the rig frame.
8. **What does an egomotion pose do?** It maps rig coordinates into the clip-local anchor frame at a timestamp.
9. **Why compensate motion?** Each LiDAR return has a time; compensation places returns at camera time.
10. **How is a voxel index calculated?** `floor((point-origin)/voxel)` component by component.
11. **What makes a cell OCC?** At least one retained point lands in it.
12. **What happens to a cell with no return?** It remains UNKNOWN; no FREE ray is written.
13. **Why remove an own-vehicle box?** Returns from the recording vehicle should not become scene occupancy.
14. **What does `n_points_in_grid` mean?** Valid point returns inside the grid before duplicate points are collapsed into occupied voxels.
15. **What remains to validate?** Camera frame phase, physical rig-origin placement, and broader visual calibration quality.

## Audit

- Code/docs disagreement: `build_labels.py` and `labels_qa.py` use the old `n_occ` name, while the saved manifest uses `n_points_in_grid` and `n_occ_voxels`.
- Dead or unused code: `make_zip_map` in `keyframes_matches.py` is defined but not used by its run path; `FREE` is imported by `build_labels.py` but not assigned.
- TODO/fragile areas: the physical rig origin is not documented precisely; camera timestamp phase is unresolved; source f-theta parameter units depend on metadata; hard-coded review clip/spin choices and output paths are easy to drift.
- Inventory note: no other project Python scripts were found outside `tools/`; `occgrid.py` and `viewer/` were intentionally excluded from this documentation set.
