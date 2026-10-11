import numpy as np
import pytest

from occgrid import Grid, OCC
from viewer.palette import (CLASS_COLORS, CLASS_NAMES, _MISSING, _RAMP_COLORS,
                            by_class, flat, height_ramp)

PALETTES = [flat, height_ramp, by_class]


def make_grid(sem=None):
    occ = np.zeros((3, 3, 3), np.uint8)
    occ[0, 0, 0] = occ[1, 1, 1] = OCC
    return Grid(occ=occ, origin=np.zeros(3, np.float32), voxel=0.4, sem=sem)


def test_ramp_hits_endpoints():
    centers = np.array([[0, 0, -1.0], [0, 0, 5.4]])
    rgb = height_ramp(centers, None, None)
    np.testing.assert_allclose(rgb[0], _RAMP_COLORS[0], atol=1e-5)
    np.testing.assert_allclose(rgb[1], _RAMP_COLORS[-1], atol=1e-5)


def test_ramp_clamps_outside_range():
    centers = np.array([[0, 0, -99.0], [0, 0, 99.0]])
    rgb = height_ramp(centers, None, None)
    np.testing.assert_allclose(rgb[0], _RAMP_COLORS[0], atol=1e-5)
    np.testing.assert_allclose(rgb[1], _RAMP_COLORS[-1], atol=1e-5)





def test_ramp_uses_z_not_x():
    """centers[:, 2] is height. Picking the wrong column is invisible on screen."""
    a = height_ramp(np.array([[0.0, 0.0, 5.4]]), None, None)
    b = height_ramp(np.array([[5.4, 0.0, -1.0]]), None, None)
    assert not np.allclose(a, b)


def test_by_class_looks_up_the_table():
    sem = np.zeros((3, 3, 3), np.uint8)
    sem[0, 0, 0] = 7    # car
    sem[1, 1, 1] = 4    # vegetation
    g = make_grid(sem)
    idx = np.array([[0, 0, 0], [1, 1, 1]])
    rgb = by_class(np.zeros((2, 3)), g, idx)
    np.testing.assert_allclose(rgb[0], CLASS_COLORS[7])
    np.testing.assert_allclose(rgb[1], CLASS_COLORS[4])


def test_by_class_unmapped_id_is_magenta_not_a_crash():
    sem = np.zeros((3, 3, 3), np.uint8)
    sem[0, 0, 0] = 99
    g = make_grid(sem)
    rgb = by_class(np.zeros((1, 3)), g, np.array([[0, 0, 0]]))
    np.testing.assert_allclose(rgb[0], _MISSING)


def test_by_class_without_sem_falls_back():
    g = make_grid(sem=None)
    rgb = by_class(np.zeros((2, 3)), g, np.array([[0, 0, 0], [1, 1, 1]]))
    assert rgb.shape == (2, 3)


def test_class_table_matches_names():
    assert len(CLASS_COLORS) == len(CLASS_NAMES)


@pytest.mark.parametrize("palette", PALETTES, ids=lambda p: p.__name__)
def test_palette_contract(palette):
    """Every palette must return [N,3] float32 in 0..1, interchangeably."""
    n = 5
    centers = np.random.default_rng(0).uniform(-10, 10, (n, 3))
    sem = np.zeros((3, 3, 3), np.uint8)
    sem[0, 0, 0] = 7
    g = make_grid(sem)
    idx = np.zeros((n, 3), np.int64)
    rgb = palette(centers, g, idx)
    assert rgb.shape == (n, 3)
    assert rgb.dtype == np.float32
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


@pytest.mark.parametrize("palette", PALETTES, ids=lambda p: p.__name__)
def test_palette_handles_zero_voxels(palette):
    g = make_grid(np.zeros((3, 3, 3), np.uint8))
    rgb = palette(np.zeros((0, 3)), g, np.zeros((0, 3), np.int64))
    assert rgb.shape == (0, 3)


def test_ramp_is_ordered_and_distinct():
    """Equal heights give equal colours; different heights give different ones,
    and the ends match the gradient's endpoints."""
    centers = np.zeros((20, 3))
    centers[:, 2] = np.linspace(-1.0, 5.4, 20)
    rgb = height_ramp(centers, None, None)

    np.testing.assert_allclose(rgb[0], _RAMP_COLORS[0], atol=1e-5)
    np.testing.assert_allclose(rgb[-1], _RAMP_COLORS[-1], atol=1e-5)

    steps = np.linalg.norm(np.diff(rgb, axis=0), axis=1)
    assert np.all(steps > 1e-4)        # every step changes the colour
