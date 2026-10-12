# `tools/make_contact_sheet.py`

## Purpose and place

Builds a thumbnail contact sheet of rectified frames sorted by brightness. It is a lightweight review step after rectification.

## Inputs and outputs

It reads `data/processed/rectification_report.json` and rectified `after_*.jpg` files. It writes `data/processed/outputs/contact_sheet.jpg` and `data/processed/rectification_summary.csv`.

## Run

```powershell
python tools/make_contact_sheet.py
```

There are no CLI arguments.

## Walkthrough

`main` loads the report, sorts clip rows by `per_clip_mean_brightness`, chooses the first matching rectified JPEG, resizes it to the configured thumbnail size, adds a short clip id and brightness label, and saves the sheet plus a compact CSV summary.

## Math example

If a clip’s report brightness is 42.5 and its first rectified image is found, the thumbnail label is the short clip id followed by `42.5`; the script does not recompute brightness from pixels.

## Constants and conventions

The sheet uses 256x96 thumbnails and five columns. Input/output locations are constants. Sorting is ascending, so the darkest clips appear first.

## Checks and failure modes

Missing report rows or matching images cause an error or an incomplete sheet. The script does not validate calibration or image geometry.

## Explain it back

1. **What determines order?** Reported per-clip mean brightness.
2. **What image is selected?** The first matching `after_*.jpg`.
3. **Does it change source images?** No.
4. **How many columns?** Five.
5. **What accompanies the JPEG?** A CSV summary.
