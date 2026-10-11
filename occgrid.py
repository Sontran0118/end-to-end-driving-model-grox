from dataclasses import dataclass

import numpy as np

FORMAT_VERSION = 1

UNKNOWN, FREE, OCC = 0, 1, 2
STATE_NAMES = ["unknown", "free", "occupied"]


@dataclass(frozen=True)
class Grid:
    """Occupancy over a box of space, ego-frame at one instant.

    Axes are X forward, Y left, Z up (rig FLU). Index order is occ[ix, iy, iz].
    """

    occ: np.ndarray                  # uint8   [X, Y, Z]   {UNKNOWN, FREE, OCC}
    origin: np.ndarray               # float32 [3]         rig-frame corner of voxel [0,0,0]
    voxel: float                     #                     edge length, metres
    sem: np.ndarray | None = None    # uint8   [X, Y, Z]   class id, 0 = none
    flow: np.ndarray | None = None   # float16 [X, Y, 2]   BEV velocity, m/s
    pose: np.ndarray | None = None   # float32 [4, 4]      rig -> anchor

    def __post_init__(self):
        if self.occ.dtype != np.uint8 or self.occ.ndim != 3:
            raise ValueError(f"occ must be uint8 [X,Y,Z], got {self.occ.dtype} {self.occ.shape}")
        if self.occ.max(initial=0) > OCC:
            raise ValueError(f"occ has values above OCC={OCC}")
        if self.origin.shape != (3,):
            raise ValueError(f"origin must be [3], got {self.origin.shape}")
        if not self.voxel > 0:
            raise ValueError(f"voxel must be positive, got {self.voxel}")
        if self.sem is not None and self.sem.shape != self.occ.shape:
            raise ValueError(f"sem {self.sem.shape} != occ {self.occ.shape}")
        if self.flow is not None and self.flow.shape != self.occ.shape[:2] + (2,):
            raise ValueError(f"flow {self.flow.shape} != {self.occ.shape[:2] + (2,)}")
        if self.pose is not None and self.pose.shape != (4, 4):
            raise ValueError(f"pose must be [4,4], got {self.pose.shape}")
