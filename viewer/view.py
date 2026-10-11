"""Log a PhysicalAI-AV clip to Rerun: ego trajectory, front camera, lidar."""

import DracoPy
import numpy as np
import rerun as rr
from physical_ai_av import PhysicalAIAVDatasetInterface

N_KEYFRAMES = 40


def height_colors(pts, lo=-2.0, hi=4.0):
    """Colour points by height so structure is visible against the ground plane."""
    t = np.clip((pts[:, 2] - lo) / (hi - lo), 0, 1)
    return np.stack([t, 0.4 + 0.4 * t, 1 - t], axis=-1)


ds = PhysicalAIAVDatasetInterface()

CAM = ds.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
LIDAR = "lidar_top_360fov"

has_both = ds.feature_presence[CAM] & ds.feature_presence[LIDAR]
clip_id = ds.feature_presence.index[has_both][0]
print(f"{clip_id=} chunk={ds.get_clip_chunk(clip_id)}")

extr = ds.get_clip_feature(clip_id, "sensor_extrinsics", maybe_stream=True)
T_rig_cam = extr.sensor_poses[CAM]
T_rig_lidar = extr.sensor_poses[LIDAR]

ego = ds.get_clip_feature(clip_id, "egomotion", maybe_stream=True)
t0, t1 = ego.time_range
print(f"egomotion range: {t0} .. {t1} us ({(t1 - t0) / 1e6:.1f} s)")

reader = ds.get_clip_feature(clip_id, CAM, maybe_stream=True)
print(f"video: {len(reader.timestamps)} frames @ {float(reader.fps):.2f} fps")

# Keyframe timestamps, clamped inside the egomotion range (interpolators do not
# extrapolate -- outside the range you get NaN, not an error).
cam_ts = reader.timestamps
cam_ts = cam_ts[(cam_ts >= t0) & (cam_ts <= t1)]
keyframe_ts = cam_ts[np.linspace(0, len(cam_ts) - 1, N_KEYFRAMES).astype(np.int64)]

images, actual_ts = reader.decode_images_from_timestamps(keyframe_ts)
print(f"decoded {images.shape}")

# Lidar: inspect the schema before trusting any column names.
lidar_df = ds.get_clip_feature(clip_id, LIDAR, maybe_stream=True)["pointclouds"]
print("lidar columns:", lidar_df.columns.tolist())
print(lidar_df.dtypes)
print(lidar_df.head())


spin_starts = lidar_df["spin_start_timestamp"].to_numpy()
spin_ends = lidar_df["spin_end_timestamp"].to_numpy()
print(
    f"lidar: {len(lidar_df)} spins, {spin_starts[0]} .. {spin_ends[-1]} us, "
    f"{np.median(spin_ends - spin_starts) / 1000:.1f} ms/spin"
)


def decode_spin(i):
    """Decodes the Draco-compressed point cloud of spin `i` -> xyz [N,3] float32."""
    pc = DracoPy.decode(lidar_df["draco_encoded_pointcloud"].iloc[i])
    return np.asarray(pc.points, dtype=np.float32)


def spin_for(ts):
    """Index of the spin covering timestamp `ts` (microseconds)."""
    return int(np.clip(np.searchsorted(spin_starts, ts, side="right") - 1, 0, len(lidar_df) - 1))


print("first spin decodes to", decode_spin(0).shape)

rr.init("physicalai-av", spawn=True)

# Static: whole ego trajectory as one line strip in the anchor frame.
traj_ts = np.linspace(t0, t1, 500)
traj = np.stack([ego(t).pose.translation for t in traj_ts])
rr.log("world/trajectory", rr.LineStrips3D([traj], radii=0.1), static=True)

for ts, img in zip(actual_ts, images):
    rr.set_time("sensor", timestamp=float(ts) * 1e-6)

    pose = ego(ts).pose
    rr.log(
        "world/rig",
        rr.Transform3D(translation=pose.translation, mat3x3=pose.rotation.as_matrix()),
    )
    rr.log(
        "world/rig/lidar",
        rr.Transform3D(
            translation=T_rig_lidar.translation,
            mat3x3=T_rig_lidar.rotation.as_matrix(),
        ),
    )
    rr.log(
        "world/rig/cam_front",
        rr.Transform3D(
            translation=T_rig_cam.translation, mat3x3=T_rig_cam.rotation.as_matrix()
        ),
    )
    rr.log("world/rig/cam_front/image", rr.Image(img))

    # The spin covering this keyframe, in raw lidar coordinates.
    # Rerun composes world/rig * world/rig/lidar for us.
    pts = decode_spin(spin_for(ts))
    rr.log(
        "world/rig/lidar/points",
        rr.Points3D(pts, radii=0.03, colors=height_colors(pts)),
    )

reader.close()
print("done -- scrub the 'sensor' timeline in the viewer")
