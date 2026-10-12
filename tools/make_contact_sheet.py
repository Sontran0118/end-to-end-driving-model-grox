"""Build a brightness-sorted contact sheet from existing rectified frames.

This reads the existing rectification report and JPEG outputs; it does not
decode videos or rerun the rectification pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT = PROCESSED / "rectification_report.json"
SHEET = PROCESSED / "outputs" / "contact_sheet.jpg"
SUMMARY = PROCESSED / "rectification_summary.csv"
THUMB_W, THUMB_H = 256, 96
COLS = 5


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = sorted(report["per_clip_mean_brightness"], key=lambda row: row["mean_brightness"])
    out_dir = SHEET.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = 255 * np.ones((((len(rows) + COLS - 1) // COLS) * (THUMB_H + 24), COLS * THUMB_W, 3), dtype="uint8")
    summary_rows = []
    for i, row in enumerate(rows):
        clip_id = row["clip_id"]
        images = sorted((PROCESSED / "images" / clip_id).glob("after_*.jpg"))
        if not images:
            raise FileNotFoundError(f"No rectified image for {clip_id}")
        image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {images[0]}")
        thumb = cv2.resize(image, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        x = (i % COLS) * THUMB_W
        y = (i // COLS) * (THUMB_H + 24)
        sheet[y : y + THUMB_H, x : x + THUMB_W] = thumb
        label = f"{clip_id[:8]}  {row['mean_brightness']:.2f}"
        cv2.putText(sheet, label, (x + 4, y + THUMB_H + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        summary_rows.append({
            "clip_id": clip_id,
            "mean_brightness": row["mean_brightness"],
            "outside_fraction": row["geometry"]["output_at_hfov_120"]["outside_fraction"],
            "largest_no_border_hfov_degrees": row["geometry"]["largest_hfov_without_borders_degrees"],
        })
    if not cv2.imwrite(str(SHEET), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {SHEET}")
    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({
        "contact_sheet": str(SHEET.relative_to(ROOT)),
        "summary_csv": str(SUMMARY.relative_to(ROOT)),
        "clips": len(rows),
        "bytes": SHEET.stat().st_size,
        "size": [COLS * THUMB_W, ((len(rows) + COLS - 1) // COLS) * (THUMB_H + 24)],
    }, indent=2))


if __name__ == "__main__":
    main()
