import numpy as np
import pytest

from viewer.camera import OrbitCamera, look_at, perspective


def test_look_at_rotation_is_orthonormal():
    """A view matrix's rotation block must be a pure rotation -- no scale, no mirror."""
    m = look_at([10.0, 0.0, 5.0], [0.0, 0.0, 0.0])
    R = m[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) > 0


def test_look_at_puts_eye_at_origin():
    """The camera becomes the origin of eye space, by construction."""
    eye = np.array([10.0, 4.0, 5.0])
    m = look_at(eye, [0.0, 0.0, 0.0])
    p = m @ np.append(eye, 1.0)
    np.testing.assert_allclose(p[:3], [0, 0, 0], atol=1e-5)


def test_look_at_puts_target_on_negative_z():
    """GL cameras look down -Z. Positive z here would mean the scene is behind you."""
    m = look_at([40.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    p = m @ np.array([20.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(p[:2], [0, 0], atol=1e-5)
    assert p[2] < 0
    np.testing.assert_allclose(p[2], -20.0, atol=1e-5)


def test_perspective_maps_near_and_far_to_ndc_bounds():
    m = perspective(45.0, 1.0, 0.1, 300.0)
    for z, expected in ((-0.1, -1.0), (-300.0, 1.0)):
        p = m @ np.array([0.0, 0.0, z, 1.0])
        np.testing.assert_allclose(p[2] / p[3], expected, atol=1e-4)


def test_perspective_shrinks_with_distance():
    """The whole point of a projection: same size object, further away, smaller."""
    m = perspective(45.0, 1.0, 0.1, 300.0)
    near = m @ np.array([5.0, 0.0, -10.0, 1.0])
    far = m @ np.array([5.0, 0.0, -50.0, 1.0])
    assert abs(near[0] / near[3]) > abs(far[0] / far[3])


def test_perspective_aspect_compensates_for_window_shape():
    """A wide window must not stretch the scene horizontally."""
    square = perspective(45.0, 1.0, 0.1, 300.0)
    wide = perspective(45.0, 2.0, 0.1, 300.0)
    assert wide[0, 0] == pytest.approx(square[0, 0] / 2.0)
    assert wide[1, 1] == pytest.approx(square[1, 1])


def test_eye_is_at_the_requested_distance():
    cam = OrbitCamera(target=(20.0, 0.0, 0.0), distance=40.0)
    assert np.linalg.norm(cam.eye - cam.target) == pytest.approx(40.0)


def test_target_projects_to_screen_centre():
    """End-to-end: whatever you aim at must land at NDC (0,0)."""
    cam = OrbitCamera(target=(20.0, 0.0, 0.0), distance=40.0)
    clip = cam.mvp(16 / 9) @ np.array([20.0, 0.0, 0.0, 1.0])
    ndc = clip[:3] / clip[3]
    np.testing.assert_allclose(ndc[:2], [0.0, 0.0], atol=1e-5)


def test_pitch_clamp_survives_repeated_orbit():
    """At +-pi/2 the view direction is parallel to up and look_at divides by zero."""
    cam = OrbitCamera()
    for _ in range(100):
        cam.orbit(0.0, 0.5)
    assert abs(cam.pitch) < np.pi / 2
    assert np.all(np.isfinite(cam.view()))

    for _ in range(200):
        cam.orbit(0.0, -0.5)
    assert abs(cam.pitch) < np.pi / 2
    assert np.all(np.isfinite(cam.view()))


def test_zoom_is_multiplicative_and_bounded():
    cam = OrbitCamera(distance=40.0)
    cam.zoom(0.5)
    assert cam.distance == pytest.approx(20.0)
    for _ in range(50):
        cam.zoom(0.5)
    assert cam.distance >= 1.0          # clamped, never zero or negative
