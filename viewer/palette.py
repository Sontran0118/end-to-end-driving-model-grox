import numpy as np

_RAMP_STOPS = np.array([0.00, 0.35, 0.65, 1.00])

_RAMP_COLORS = np.array([
    [0.106, 0.165, 0.420],   # #1b2a6b  deep blue
    [0.169, 0.702, 0.753],   # #2bb3c0  cyan
    [0.478, 0.820, 0.318],   # #7ad151  green
    [0.992, 0.906, 0.145],   # #fde725  yellow
], np.float32)


CLASS_NAMES = ["none", "road", "marking", "footway", "vegetation", "building",
               "pole/sign", "car", "truck/bus", "person/rider", "other"]
CLASS_COLORS = np.array([
    [0.25, 0.25, 0.25],   # 0  none
    [0.35, 0.35, 0.38],   # 1  road
    [0.90, 0.90, 0.90],   # 2  marking
    [0.55, 0.45, 0.35],   # 3  footway
    [0.20, 0.70, 0.25],   # 4  vegetation
    [0.75, 0.55, 0.35],   # 5  building
    [0.85, 0.80, 0.20],   # 6  pole/sign
    [0.20, 0.45, 1.00],   # 7  car
    [0.10, 0.25, 0.70],   # 8  truck/bus
    [1.00, 0.20, 0.20],   # 9  person/rider
    [0.60, 0.60, 0.60],   # 10 other
], np.float32)
_MISSING = np.array([1.0, 0.0, 1.0], np.float32)   # magenta: unmapped class


def flat(centers, grid, idx):
    """Flatten a grid of class indices to a color image.

    Args:
        centers: (N, 3) array of RGB colors for each class.
        grid: (H, W) array of class indices.
        idx: (H, W) array of indices into centers.

    Returns:
        (H, W, 3) array of RGB colors.
    """


    return np.full((len(centers),3),0.7,np.float32)


def height_ramp(centers,grid, idx, lo=-1.0,hi=5.4):
    """Colour by world height, clamped to [lo, hi]."""
    t = np.clip((centers[:,2] -lo) / (hi - lo), 0.0, 1.0)

    rgb = np.stack(
        [np.interp(t, _RAMP_STOPS, _RAMP_COLORS[:,i]) for i in range(3)],
        axis=-1
    
    )
    return rgb.astype(np.float32)

def by_class(centers, grid, idx):
    """Colour by class."""
    if grid.sem is None:
        return flat(centers, grid, idx)

    cls = grid.sem[idx[:,0], idx[:,1], idx[:,2]]

    out = np.broadcast_to(_MISSING, (len(cls), 3)).copy()

    known = cls < len(CLASS_COLORS)
    out[known] = CLASS_COLORS[cls[known]]
    return out


