"""Focused mathematical tests for tools.transforms."""

import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from transforms import (
    camera_to_ego_transform,
    ego_t_to_ego_t_minus_k,
    interpolate_anchor_from_ego,
    lidar_to_ego_transform,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0, 10],
            "qx": [0.0, 0.0], "qy": [0.0, 0.0],
            "qz": [0.0, np.sin(np.pi / 4)], "qw": [1.0, np.cos(np.pi / 4)],
            "x": [0.0, 10.0], "y": [0.0, 2.0], "z": [0.0, 4.0],
        }
    )


class TransformTests(unittest.TestCase):
    def test_sensor_transform_known_point_and_inverse(self) -> None:
        row = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 1.0, "y": 2.0, "z": 3.0}
        transform = camera_to_ego_transform(row)
        point = transform @ np.array([4.0, 5.0, 6.0, 1.0])
        np.testing.assert_allclose(point, [5.0, 7.0, 9.0, 1.0])
        np.testing.assert_allclose(transform @ np.linalg.inv(transform), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(lidar_to_ego_transform(row), transform)

    def test_rotation_is_orthonormal_and_right_handed(self) -> None:
        transform = interpolate_anchor_from_ego(_df(), 5)
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertTrue(np.isclose(np.linalg.det(rotation), 1.0))

    def test_interpolation_and_exact_sample(self) -> None:
        df = _df()
        exact = interpolate_anchor_from_ego(df, 10)
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        expected[:3, 3] = [10.0, 2.0, 4.0]
        np.testing.assert_allclose(exact, expected, atol=1e-12)
        halfway = interpolate_anchor_from_ego(df, 5)
        np.testing.assert_allclose(halfway[:3, 3], [5.0, 1.0, 2.0], atol=1e-12)
        relative = ego_t_to_ego_t_minus_k(df, 10, 5)
        np.testing.assert_allclose(relative, np.linalg.inv(halfway) @ exact, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
