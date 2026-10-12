import numpy as np


from occgrid import Grid, OCC,UNKNOWN,FREE  



def corridor(shape = (100,50,16), voxel = 0.4):
    """Ground plane, two walls, and a box vehicle ahead. The main dev scene."""
    occ = np.full(shape, FREE, np.uint8)
    occ[:, :, 0] = OCC                      # road surface
    occ[:, 0, :8] = OCC                     # right wall
    occ[:, -1, :8] = OCC                    # left wall
    occ[50:62, 22:28, 1:5] = OCC            # a vehicle 20 m ahead
    occ[:, :, 12:] = UNKNOWN                # nothing observed up high
    return Grid(
        occ=occ,
        origin=np.array([0.0, -10.0, -1.0], np.float32),
        voxel=voxel,
    )


def single_voxel(shape=(5, 5, 5), voxel=0.4):
    """One cube at the centre. The smallest renderable thing."""
    occ = np.full(shape, FREE, np.uint8)
    occ[2, 2, 2] = OCC
    return Grid(occ=occ, origin=np.zeros(3, np.float32), voxel=voxel)


def ground_plane(shape=(100, 50, 16), voxel=0.4):
    """Flat road only. Checks orientation and the height ramp."""
    occ = np.full(shape, FREE, np.uint8)
    occ[:, :, 0] = OCC
    return Grid(occ=occ, origin=np.array([0.0, -10.0, -1.0], np.float32), voxel=voxel)


def checkerboard(shape=(20, 20, 20), voxel=0.4):
    """Every occupied voxel is isolated -- worst case for surface extraction."""
    occ = np.full(shape, FREE, np.uint8)
    occ[::2, ::2, ::2] = OCC
    return Grid(occ=occ, origin=np.zeros(3, np.float32), voxel=voxel)


def random_fill(shape=(100, 50, 16), voxel=0.4, p=0.1, seed=0):
    """Random occupancy for perf and stress testing."""
    rng = np.random.default_rng(seed)
    occ = np.full(shape, FREE, np.uint8)
    occ[rng.random(shape) < p] = OCC
    return Grid(occ=occ, origin=np.array([0.0, -10.0, -1.0], np.float32), voxel=voxel)


def driving(t=0.0, shape=(100, 50, 16), voxel=0.4):
    """One ego-frame keyframe of a drive at time t seconds.

    Everything is expressed relative to the car, so static world features
    slide toward the ego as t advances -- that is what makes the playback
    read as motion. The lead vehicle moves slowly because it is travelling
    in the same direction.
    """
    occ = np.full(shape, FREE, np.uint8)
    occ[:, :, 0] = OCC                       # road
    occ[:, :, 12:] = UNKNOWN                 # unobserved above

    ego_speed = 8.0                          # m/s
    s = ego_speed * t / voxel                # how far the world has slid, in cells

    # Kerbs: continuous, so sliding them is invisible. Lamp posts are not.
    occ[:, 0, :4] = OCC
    occ[:, -1, :4] = OCC
    for post in range(-2, 12):
        x = int(post * 15.0 / voxel - s)     # every 15 m
        if 0 <= x < shape[0]:
            occ[x:x+2, 1:3, :10] = OCC
            occ[x:x+2, -3:-1, :10] = OCC

    # Lead vehicle, 25 m ahead, pulling away at 2 m/s relative.
    lead = int((25.0 + 2.0 * t) / voxel)
    if 0 <= lead < shape[0] - 12:
        occ[lead:lead+12, 22:28, 1:5] = OCC

    # Oncoming vehicle in the other lane, closing fast.
    onc = int((70.0 - 14.0 * t) / voxel)
    if 0 <= onc < shape[0] - 12:
        occ[onc:onc+12, 10:16, 1:5] = OCC

    return Grid(occ=occ, origin=np.array([0.0, -10.0, -1.0], np.float32), voxel=voxel)


def drive_sequence(n=120, fps=10.0, **kw):
    """n keyframes of `driving`, one per 1/fps second."""
    return [driving(t=i / fps, **kw) for i in range(n)]







