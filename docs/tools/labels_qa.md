# `tools/labels_qa.py`

## Purpose and place

Performs a read-only QA pass over saved occupancy labels and builds a 50-clip voxel-dot contact sheet. It is after label generation and does not recompute labels.

## Inputs and outputs

It reads every `data/labels/**/*.npz`, `labels_manifest.parquet`, `clips.csv`, and rectified images. It writes `outputs/labels_qa/contact_sheet_voxel_dots.jpg`.

## Run

```powershell
python tools/labels_qa.py
```

There are no CLI arguments.

## Walkthrough

The loader constructs `Grid` for every NPZ so validation runs, reports one file’s keys/dtypes/shapes, and checks manifest paths. The contact-sheet path chooses one middle keyframe per clip, applies viewer `surface_mask`, computes voxel centers, drops camera depth ≤0.5 m, draws 2-pixel height-colored dots, blends once, resizes thumbnails, and labels them.

## Math example

If a center projects to camera depth `z=10` and camera x `1`, its horizontal pixel is about `369.504/10+640=676.95`; if it is within 1280x384 it can be drawn.

## Constants and conventions

The sheet uses four thumbnails per row and the fixed rectified K. It chooses the middle keyframe after sorting each clip’s rows. OCC/surface semantics come from the viewer rather than this script.

## Checks and failure modes

The recorded QA run found 1,850 NPZs, zero Grid failures, 1,850 manifest rows, and zero missing paths; prior split counts were 35/7/8 clips. It recorded NPZ key shapes and generated the contact sheet. Current fragility: the script still references `manifest.n_occ` and `record.n_occ`, while the saved manifest was later renamed to `n_points_in_grid` plus `n_occ_voxels`; rerunning may fail until code is updated.

## Explain it back

1. **Does it write labels?** No.
2. **What does Grid construction test?** Shape/types/invariants required by the viewer model.
3. **What does the contact sheet show?** One mid-clip surface-voxel dot overlay per clip.
4. **Why drop depth ≤0.5 m?** To avoid drawing invalid/too-near projections.
5. **What should be fixed before rerunning?** The stale `n_occ` manifest references.
