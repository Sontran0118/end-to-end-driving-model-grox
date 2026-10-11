from physical_ai_av import PhysicalAIAVDatasetInterface

ds = PhysicalAIAVDatasetInterface()
print(ds)

CAM = ds.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
LIDAR = "lidar_top_360fov"

# Pick the first clip that has both the front wide camera and lidar.
has_both = ds.feature_presence[CAM] & ds.feature_presence[LIDAR]
print(f"{has_both.sum()} / {len(has_both)} clips have both {CAM} and {LIDAR}")

clip_id = ds.feature_presence.index[has_both][0]
print(f"{clip_id=} chunk={ds.get_clip_chunk(clip_id)}")

intr = ds.get_clip_feature(clip_id, "camera_intrinsics", maybe_stream=True)
extr = ds.get_clip_feature(clip_id, "sensor_extrinsics", maybe_stream=True)
dims = ds.get_clip_feature(clip_id, "vehicle_dimensions", maybe_stream=True)

cam = intr.camera_models[CAM]
print(f"{cam.width=} {cam.height=} principal_point={cam.principal_point}")
print("th2r:", cam.th2r)
print("r2th:", cam.r2th)

print("sensors:", sorted(extr.sensor_poses))
T_rig_cam = extr.sensor_poses[CAM]
T_rig_lidar = extr.sensor_poses[LIDAR]
print("camera origin in rig frame:", T_rig_cam.apply([0.0, 0.0, 0.0]))
print("lidar  origin in rig frame:", T_rig_lidar.apply([0.0, 0.0, 0.0]))

print(dims)

# --- video ---------------------------------------------------------------
import numpy as np
from PIL import Image

reader = ds.get_clip_feature(clip_id, CAM, maybe_stream=True)
print(f"fps={float(reader.fps):.3f} frames={len(reader.timestamps)}")
print("duration_s:", (reader.timestamps[-1] - reader.timestamps[0]) / 1e6)

frame_idxs = np.linspace(0, len(reader.timestamps) - 1, 6).astype(np.int64)
images = reader.decode_images_from_frame_indices(frame_idxs)
print("images:", images.shape, images.dtype)

for i, img in zip(frame_idxs, images):
    Image.fromarray(img).save(f"frame_{i:05d}.png")

# contact sheet: all 6 samples side by side, downscaled
h, w = images.shape[1:3]
sheet = np.concatenate(list(images), axis=1)
Image.fromarray(sheet).resize((sheet.shape[1] // 6, h // 6)).save("contact_sheet.png")

reader.close()
