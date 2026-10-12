"""GL rendering of voxel instances. No window -- app.py owns that.

Targets GLSL 410 core: macOS's ceiling, well under the Jetson's 4.6, so the
same shaders run on both.
"""

from pathlib import Path

import moderngl
import numpy as np

SHADER_DIR = Path(__file__).parent / "shaders"

_FACES = [
    ((+1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, +1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, +1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]


def cube_mesh():
    """Unit cube spanning [-0.5, +0.5] -> [36, 6] float32 of pos+normal."""
    verts = []
    for n, a, b in _FACES:
        n, a, b = (np.array(v, np.float32) for v in (n, a, b))
        c = n * 0.5
        quad = [c - a/2 - b/2, c + a/2 - b/2, c + a/2 + b/2, c - a/2 + b/2]
        for i in (0, 1, 2, 0, 2, 3):
            verts.append(np.concatenate([quad[i], n]))
    return np.array(verts, np.float32)
class VoxelRenderer:
    """Draws instanced cubes. One draw call regardless of voxel count."""

    def __init__(self, ctx, max_instances=200_000):
        self.ctx = ctx
        self.max_instances = max_instances
        self.n = 0

        self.prog = ctx.program(
            vertex_shader=(SHADER_DIR / "voxel.vert").read_text(),
            fragment_shader=(SHADER_DIR / "voxel.frag").read_text(),
        )

        self.cube_vbo = ctx.buffer(cube_mesh().tobytes())
        self.inst_vbo = ctx.buffer(reserve=max_instances * 6 * 4)

        self.vao = ctx.vertex_array(
            self.prog,
            [
                (self.cube_vbo, "3f 3f", "in_pos", "in_normal"),
                (self.inst_vbo, "3f 3f/i", "in_center", "in_color"),
            ],
        )

    def upload(self, instances):
        """instances: [N, 6] float32, contiguous. Writes a subrange, never reallocates."""
        n = len(instances)
        if n > self.max_instances:
            raise ValueError(f"{n} instances exceeds max {self.max_instances}")
        if n:
            self.inst_vbo.write(instances.tobytes())
        self.n = n

    def draw(self, mvp, voxel_size, clear=(0.04, 0.055, 0.08)):
        self.ctx.clear(*clear, depth=1.0)
        if self.n == 0:
            return
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.prog["mvp"].write(np.ascontiguousarray(mvp.T, np.float32).tobytes())
        self.prog["voxel_size"].value = float(voxel_size)
        self.vao.render(instances=self.n)
