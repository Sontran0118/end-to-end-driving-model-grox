import numpy as np

from occgrid import Grid, OCC


def surface_mask(occ: np.ndarray) -> np.ndarray:
    """True for occupied voxels with at least one non-occupied face-neighbour.

    Interior voxels are invisible from every direction, so they can be dropped
    before rendering. Padding with False means grid-boundary voxels count as
    surface, which is correct -- you can see the outside of the volume.
    """
    b = occ == OCC
    p = np.pad(b, 1, constant_values=False)
    interior = (
        p[2:, 1:-1, 1:-1]
        & p[:-2, 1:-1, 1:-1]
        & p[1:-1, 2:, 1:-1]
        & p[1:-1, :-2, 1:-1]
        & p[1:-1, 1:-1, 2:]
        & p[1:-1, 1:-1, :-2]
    )
    return b & ~interior


def build_instances(grid: Grid, palette) -> np.ndarray:
    idx = np.argwhere(surface_mask(grid.occ))
    centers = grid.origin + (idx + 0.5) * grid.voxel
    colors = palette(centers, grid, idx)
    return np.ascontiguousarray(np.hstack([centers, colors]), dtype=np.float32)






