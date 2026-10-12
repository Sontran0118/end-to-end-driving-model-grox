# PhysicalAI AV sample

## Selection

- Source: `nvidia/PhysicalAI-Autonomous-Vehicles`, current metadata revision inspected on 2026-10-11.
- Selection uses only the local metadata tables in `data/metadata/` plus the devkit's `features.csv` rules.
- `clip_id` is the index in the metadata tables. NVIDIA's split is `clip_index.split`.
- Required features: `lidar_top_360fov`, `camera_front_wide_120fov`, `egomotion`, `camera_intrinsics.offline`, and `sensor_extrinsics.offline`.
- Offline calibration absence is excluded by requiring both offline calibration flags to be true.
- Platform: `hyperion_8`, chosen because it reaches the requested split-balanced sample with fewer available chunks than `hyperion_8.1`.
- Seed: 42. Sample: 50 clips, exactly 35 train / 7 val / 8 test.
- Chunks: train `41`, validation `2107`, test `2246`.
- The minimum possible number of chunks is three because the selected split quotas come from separate split-pure chunks.

## Time and country

The selected countries are United States (43) and Germany (7).

The selected `hour_of_day` distribution is:

| Hour | Count |
|---:|---:|
| 0 | 2 |
| 7 | 2 |
| 8 | 3 |
| 9 | 6 |
| 10 | 6 |
| 11 | 5 |
| 12 | 2 |
| 13 | 1 |
| 14 | 2 |
| 15 | 4 |
| 16 | 1 |
| 17 | 7 |
| 18 | 4 |
| 19 | 1 |
| 21 | 1 |
| 22 | 2 |
| 23 | 1 |

The NVIDIA schema documents `hour_of_day` only as the recording hour (0–23); it does not specify local time versus UTC. No `daytime`, `nighttime`, `weather`, or `clear` field is present, so no timezone or daylight rule was guessed.

Using only a clearly labeled provisional grouping of 19:00–06:00 as night-like, 7 of 50 clips (14%) fall in that group. This is not many. If desired, daytime-like replacements can be made without adding chunks: the same three chunks contain 78 train, 59 validation, and 79 test candidate clips at hours 07–18. The current sample is therefore retained unchanged.

## Download-size estimate

Whole selected feature archives (the devkit's `download_clip_features()` behavior):

| Feature | Chunk 41 | Chunk 2107 | Chunk 2246 | Total |
|---|---:|---:|---:|---:|
| LiDAR ZIP | 31,408,106,639 | 29,983,509,473 | 29,415,355,494 | 90,806,971,606 bytes / 84.571 GiB |
| Front-wide camera ZIP | 1,715,875,840 | 1,328,058,319 | 1,278,917,606 | 4,322,851,765 bytes / 4.026 GiB |
| Egomotion ZIP | 39,527,851 | 35,864,893 | 39,255,633 | 114,648,377 bytes / 0.107 GiB |
| Offline calibration Parquets | 243,677 | 169,578 | 220,857 | 634,112 bytes / 0.001 GiB |
| **Total** | 33,163,754,007 | 31,347,602,263 | 30,733,749,590 | **95,245,105,860 bytes / 88.70 GiB** |

For a clip-granular estimate, remote ZIP central directories were read with HTTP range requests; no LiDAR or video payload was downloaded. The selected members plus whole calibration Parquets are:

| Feature | Chunk 41 | Chunk 2107 | Chunk 2246 | Selected total |
|---|---:|---:|---:|---:|
| LiDAR selected members | 11,399,576,597 | 2,396,428,717 | 2,368,146,519 | 16,164,151,833 bytes / 15.054 GiB |
| Camera video selected members | 674,670,832 | 65,416,472 | 79,674,442 | 819,761,746 bytes / 0.763 GiB |
| Camera timestamp selected members | 333,982 | 66,720 | 76,160 | 476,862 bytes / 0.000 GiB |
| Egomotion selected members | 14,655,956 | 3,175,892 | 3,164,009 | 20,995,857 bytes / 0.020 GiB |
| Whole offline calibration Parquets | 243,677 | 169,578 | 220,857 | 634,112 bytes / 0.001 GiB |
| **Estimated total** | 12,089,481,044 | 2,465,257,379 | 2,451,281,987 | **17,006,020,410 bytes / 15.838 GiB** |

The per-chunk subtotal in the final row is intentionally not used as an exact transfer promise: it depends on range-request overhead and ZIP local headers. The reliable selected-total estimate is 17,006,020,410 bytes.

## Devkit behavior

The devkit's `download_clip_features()` maps clip IDs to their chunks and downloads the complete feature file for each needed chunk. It does not download individual ZIP members. `get_clip_feature(..., maybe_stream=True)` can stream a chunk file and then read the requested member, so a range-aware custom workflow can approach the selected-member estimate above. Calibration features are Parquet files and remain whole-chunk downloads.

No LiDAR or video feature files have been downloaded. The 120 GiB cap is recorded in `AGENTS.md`; every future download still requires a size report and explicit approval.
