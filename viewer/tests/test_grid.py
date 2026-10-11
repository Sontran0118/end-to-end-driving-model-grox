import numpy as np
import pytest

from occgrid import Grid, OCC
from viewer.grid import build_instances, surface_mask


def flat(centers, grid, idx):
    """Minimal palette: every voxel mid-grey."""
    return np.full((len(centers), 3), 0.5, np.float32)


def test_single_voxel_is_surface():
    occ = np.zeros((5, 5, 5), np.uint8)
    occ[2, 2, 2] = OCC
    assert surface_mask(occ).sum() == 1


def test_solid_block_keeps_only_shell():
    occ = np.zeros((9, 9, 9), np.uint8)
    occ[2:7, 2:7, 2:7] = OCC          # 5x5x5 block, away from the boundary
    assert (occ == OCC).sum() == 125
    assert surface_mask(occ).sum() == 125 - 27     # minus the 3x3x3 core


def test_full_grid_boundary_is_surface():
    X, Y, Z = 4, 5, 6
    occ = np.full((X, Y, Z), OCC, np.uint8)
    expected = X * Y * Z - (X - 2) * (Y - 2) * (Z - 2)
    assert surface_mask(occ).sum() == expected


def test_no_wraparound():
    """A voxel at x=0 must not see x=X-1 as its neighbour."""
    occ = np.zeros((5, 1, 1), np.uint8)
    occ[0] = occ[4] = OCC
    assert surface_mask(occ).sum() == 2


def test_empty_grid():
    occ = np.zeros((4, 4, 4), np.uint8)
    assert surface_mask(occ).sum() == 0
    g = Grid(occ=occ, origin=np.zeros(3, np.float32), voxel=0.4)
    assert build_instances(g, flat).shape == (0, 6)


def test_center_position():
    """origin is a CORNER; instances must be at voxel CENTRES."""
    occ = np.zeros((3, 3, 3), np.uint8)
    occ[0, 0, 0] = OCC
    g = Grid(occ=occ, origin=np.array([0.0, -20.0, -1.0], np.float32), voxel=0.4)
    inst = build_instances(g, flat)
    assert inst.shape == (1, 6)
    np.testing.assert_allclose(inst[0, :3], [0.2, -19.8, -0.8], atol=1e-6)


def test_dtype_and_contiguity():
    """moderngl copies raw bytes -- wrong dtype or strides uploads garbage."""
    occ = np.zeros((5, 5, 5), np.uint8)
    occ[1:4, 1:4, 1:4] = OCC
    g = Grid(occ=occ, origin=np.zeros(3, np.float32), voxel=0.4)
    inst = build_instances(g, flat)
    assert inst.dtype == np.float32
    assert inst.flags.c_contiguous
    assert inst.shape[0] == surface_mask(occ).sum()
