import numpy as np


class OrbitCamera:
    """Camera that swings around a fixed target point.

    State is spherical (yaw, pitch, distance) rather than a position, because
    orbiting is then just adding to an angle. `eye` converts back on demand.
    """

    PITCH_LIMIT = np.pi / 2 - 0.02

    def __init__(self, target=(20.0, 0.0, 0.0), distance=40.0,
                 yaw=np.pi, pitch=0.4, fov_y=45.0, near=0.1, far=300.0):
        self.target = np.asarray(target, np.float64)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(np.clip(pitch, -self.PITCH_LIMIT, self.PITCH_LIMIT))
        self.fov_y = float(fov_y)
        self.near = float(near)
        self.far = float(far)

    @property
    def eye(self):
        """Spherical -> cartesian around target. sin(pitch) is z: FLU is Z-up."""
        cp = np.cos(self.pitch)
        offset = np.array([cp * np.cos(self.yaw),
                           cp * np.sin(self.yaw),
                           np.sin(self.pitch)])
        return self.target + self.distance * offset

    def view(self):
        return look_at(self.eye, self.target)

    def proj(self, aspect):
        return perspective(self.fov_y, aspect, self.near, self.far)

    def mvp(self, aspect):
        return self.proj(aspect) @ self.view()

    def orbit(self, dyaw, dpitch):
        """Mouse drag. Pitch must stay off the poles or look_at divides by zero."""
        self.yaw += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch,
                                   -self.PITCH_LIMIT, self.PITCH_LIMIT))

    def zoom(self, factor):
        """Scroll. Multiplicative, so each click moves a constant fraction."""
        self.distance = float(np.clip(self.distance * factor, 1.0, 500.0))

    def pan(self, dx, dy):
        """Shift the target in the camera's own screen plane."""
        m = self.view()
        right, up = m[0, :3], m[1, :3]
        self.target = self.target + (right * dx + up * dy) * self.distance * 0.001



def look_at(eye, target, up = (0.0, 0.0, 1.0)):
    """Compute a camera view matrix.

    Args:
        eye: (3,) array of camera position.
        target: (3,) array of camera target.
        up: (3,) array of camera up direction.

    Returns:
        (4, 4) view matrix.
    """
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m





def perspective(fov_y_deg, aspect, near, far):
    """Standard GL projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

