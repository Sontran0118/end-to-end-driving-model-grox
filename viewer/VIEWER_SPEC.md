# Occupancy viewer — build spec

Real-time 3-D occupancy renderer. Develop on macOS (M3 Pro), deploy to Jetson Orin Nano.

**Target: OpenGL 4.1 core** (`#version 410 core`) — macOS's ceiling, well under Orin's 4.6,
so one codebase runs on both.

**Principle: the viewer never imports `physical_ai_av`.** Its only input is a voxel array.
That keeps it testable against synthetic grids today, before any real labels exist, and
means a rendering bug can never be confused with a label bug.

---

## 0. Data contract

Every subsystem below agrees on this. Nothing else crosses module boundaries.

```python
# paiav/spec.py  -- shared by labeller, trainer, and viewer
UNKNOWN, FREE, OCC = 0, 1, 2

@dataclass(frozen=True)
class Grid:
    occ:    np.ndarray   # uint8  [X, Y, Z]      {UNKNOWN, FREE, OCC}
    origin: np.ndarray   # float32 [3]           world coords of corner of voxel [0,0,0]
    voxel:  float                                # edge length, metres
    sem:    np.ndarray | None = None   # uint8   [X, Y, Z]  class id, 0 = none
    flow:   np.ndarray | None = None   # float16 [X, Y, 2]  BEV velocity m/s
```

**Axes: X forward, Y left, Z up** (rig FLU, matching `sensor_extrinsics`). Index order is
`occ[ix, iy, iz]`.

> Note: `l2d_lewm/train_occ6.py` uses `(Xo, Yo, Zo)` with **Y as height**. Do not copy that
> convention here — it will silently transpose everything. Convert at the boundary.

World position of a voxel centre:

```python
center = origin + (np.array([ix, iy, iz]) + 0.5) * voxel
```

---

## 1. `viewer/grid.py` — surface extraction and instance packing

The only module with non-trivial logic. Pure NumPy, no GL.

### `surface_mask(occ) -> np.ndarray`

| | |
|---|---|
| **in** | `occ` uint8 `[X,Y,Z]` |
| **out** | bool `[X,Y,Z]`, True where occupied **and** ≥1 of its 6 face-neighbours is not occupied |
| **why** | interior voxels are invisible; dropping them is a 5-10× instance reduction |

**Do not use `np.roll`.** It wraps, so a voxel at `x=0` would see `x=X-1` as its neighbour.
Pad instead:

```python
b = occ == OCC
p = np.pad(b, 1, constant_values=False)
interior = (p[2:,1:-1,1:-1] & p[:-2,1:-1,1:-1] &
            p[1:-1,2:,1:-1] & p[1:-1,:-2,1:-1] &
            p[1:-1,1:-1,2:] & p[1:-1,1:-1,:-2])
return b & ~interior
```

Padding with `False` makes grid-boundary voxels surface voxels, which is correct — you can
see the outside of the volume.

### `build_instances(grid, palette) -> np.ndarray`

| | |
|---|---|
| **in** | `Grid`, a colouring callable |
| **out** | float32 `[N, 6]`, C-contiguous: `[cx, cy, cz, r, g, b]` |
| **invariant** | `N == surface_mask(grid.occ).sum()` |
| **budget** | ≤ 2 ms for a 200×100×16 grid |

Interleaved, one row per instance, directly uploadable as a vertex buffer.

### Tests — `tests/test_grid.py`

| Test | Assertion |
|---|---|
| single voxel | `surface_mask.sum() == 1` |
| solid 5×5×5 block in a larger grid | `sum == 125 - 27` (shell only) |
| entirely-occupied grid | `sum == X*Y*Z - (X-2)(Y-2)(Z-2)` — boundary is surface |
| **no wraparound** | occupied at `x=0` and `x=X-1`, nothing between → both are surface |
| empty grid | `sum == 0`, `build_instances` returns shape `(0, 6)` |
| position correctness | voxel `[0,0,0]` → centre `origin + voxel/2`, exactly |
| dtype/contiguity | `float32`, `flags.c_contiguous` |

---

## 2. `viewer/palette.py` — colour

### `height_ramp(z, lo, hi) -> [N,3] float32`

Tesla-like gradient, clamped outside `[lo, hi]`. Control points:

```
0.00  #1b2a6b  deep blue     (ground)
0.35  #2bb3c0  cyan
0.65  #7ad151  green
1.00  #fde725  yellow        (tall structures)
```

### `class_colors(sem) -> [N,3] float32`

Lookup table over the 11 classes already defined in `l2d_lewm/make_labels_tesla.py`
(`S_NAMES`: none/road/marking/footway/vegetation/building/pole/car/truck/person/other).
**Reuse that mapping** — don't invent a second class set.

### Tests
- monotonic in z across the ramp; endpoints exact
- `z < lo` and `z > hi` clamp, no NaN, no out-of-`[0,1]` values
- `class_colors` covers every id in `S_NAMES`; unknown id → visible magenta, not a crash

---

## 3. `viewer/shaders/` — GLSL 410

### `voxel.vert`
| in | per-vertex `cube_pos` vec3 (unit cube, 36 verts), per-vertex `face_normal` vec3; per-instance `center` vec3, `color` vec3; uniforms `mvp` mat4, `voxel_size` float |
| out | `v_color` vec3, `v_normal` vec3, `v_world` vec3, `v_bary` vec2 |

`gl_Position = mvp * vec4(center + cube_pos * voxel_size, 1.0)`

### `voxel.frag`
Four effects, in this order — each is a visible step up:

1. **Face shading** (biggest readability win): brightness from `abs(normal)` — top `1.0`,
   sides `0.75`, front/back `0.55`. Without it the volume reads as a flat blob.
2. **Edge darkening**: darken the outer 5% of each face via `v_bary`. This is what makes
   individual voxels legible rather than mush.
3. **Distance fog**: `mix(color, bg, smoothstep(fog_near, fog_far, depth))`. Reads as
   uncertainty with range — honest for a monocular model.
4. **Background** `#0a0e14`.

### Tests
- shaders compile and link in a standalone context (catches GLSL errors in CI)
- uniform names present in the linked program's member list

---

## 4. `viewer/camera.py`

```python
class OrbitCamera:
    target: np.ndarray   # [3] look-at point, metres
    distance: float
    yaw: float; pitch: float      # radians, pitch clamped to (-pi/2+eps, pi/2-eps)
    fov_y: float = 45.0           # degrees
    near: float = 0.1; far: float = 300.0

    def view(self) -> np.ndarray       # [4,4] float32
    def proj(self, aspect) -> np.ndarray
    def mvp(self, aspect) -> np.ndarray
    def orbit(self, dyaw, dpitch); def zoom(self, factor); def pan(self, dx, dy)
```

Also `FollowCamera` for playback: target tracks the ego position, yaw tracks heading with
exponential smoothing (`alpha ≈ 0.1`) so it doesn't jitter.

### Tests
- `view()` upper-left 3×3 is orthonormal (`R @ R.T ≈ I`, `det ≈ +1`)
- camera at `+X` looking at origin maps origin to NDC `(0,0)` within 1e-5
- `proj` maps `z=near → -1`, `z=far → +1` (GL depth convention)
- pitch clamp holds under repeated `orbit()` calls — no gimbal flip
- `mvp == proj @ view` exactly

---

## 5. `viewer/renderer.py` — moderngl core, no window

```python
class VoxelRenderer:
    def __init__(self, ctx: moderngl.Context, max_instances: int = 200_000)
    def upload(self, instances: np.ndarray) -> None   # [N,6] float32, N <= max
    def draw(self, mvp, params: RenderParams) -> None
```

- Cube VBO (36 verts, position + normal + bary) uploaded **once** at init
- Instance VBO **pre-allocated** to `max_instances * 24` bytes; `upload` writes a subrange.
  Never reallocate per frame — that's the classic stutter source.
- One `ctx.vertex_array` with the instance buffer bound at divisor 1
- `draw` issues exactly **one** `vao.render(instances=N)` call
- Enables depth test, back-face culling

Must work with `moderngl.create_standalone_context()` — no window dependency. `app.py` is
the only module allowed to import `moderngl_window`.

### Tests
- `upload` with `N > max_instances` raises `ValueError`, doesn't corrupt
- `upload` with `N == 0` then `draw` is a no-op, no GL error
- `ctx.error == "GL_NO_ERROR"` after a full upload+draw cycle
- **golden image**: render the synthetic scene to a 256×256 FBO, read pixels, assert
  non-background fraction within ±2% of a recorded value, and that the ground plane
  occupies the lower half. Catches regressions without brittle exact-pixel matching.
- **perf**: 50 k instances, 100 frames, assert mean frame time < 16 ms

---

## 6. `viewer/source.py` — upload abstraction

The one thing that must exist now so Jetson doesn't require a rewrite.

```python
class InstanceSource(Protocol):
    def update(self, grid: Grid) -> int:     # returns instance count
        ...
```

- **`NumpySource`** (macOS, and Orin fallback): `build_instances` → `renderer.upload`.
- **`CudaSource`** (Orin, later): GL buffer registered via `cudaGraphicsGLRegisterBuffer`;
  surface extraction and instance packing run as CUDA kernels writing straight into it.
  Zero copy, never touches the CPU — the full inference→pixels path stays on-device.

### Tests
- `NumpySource.update` return value equals `surface_mask(...).sum()`
- a `FakeRenderer` recording `upload()` calls confirms the source never reallocates

---

## 7. `viewer/synthetic.py` — test scenes

Needed **first**, so everything above is developed and tested before real labels exist.

| Generator | Purpose |
|---|---|
| `single_voxel()` | smallest renderable thing |
| `ground_plane(shape)` | `occ[:, :, 0] = OCC` — checks orientation and the height ramp |
| `corridor()` | ground + two walls + a box "vehicle" ahead — the main dev scene |
| `checkerboard()` | worst case for surface extraction: every voxel is a surface voxel |
| `random_fill(p)` | perf/stress |

### Tests
- every generator returns a valid `Grid` (dtype, shape, values ⊆ {0,1,2})
- `checkerboard` → `surface_mask.sum() == (occ == OCC).sum()`

---

## 8. `viewer/app.py` — window and playback

```
python -m viewer.app --synthetic corridor
python -m viewer.app --shards out/shards/*.npz --fps 10
```

- loads a `.npz` sequence, plays on a timeline
- keys: `space` play/pause, `←/→` step, `c` orbit↔follow, `g` toggle ground grid,
  `f` toggle fog, `1/2/3` colour by height / semantics / flow
- HUD: frame index, instance count, frame time ms

### Tests
`--headless --frames 10 --out /tmp/x.png` runs the full loop under EGL and writes a PNG.
This is the end-to-end smoke test and the only one that exercises `moderngl_window`.

---

## Build order

Each step ends with something you can see or a test that passes.

| # | Deliverable | Done when |
|---|---|---|
| 1 | `spec.py` + `synthetic.py` | generators return valid `Grid`s, tests green |
| 2 | `grid.py` | all surface-extraction tests pass, incl. no-wraparound |
| 3 | `palette.py` | ramp tests pass |
| 4 | `camera.py` | matrix tests pass |
| 5 | `renderer.py` + shaders | **a single white cube appears on screen** |
| 6 | instanced colours + face shading | corridor scene renders and reads as 3-D |
| 7 | edge darkening + fog + background | it looks like the reference |
| 8 | `app.py` | orbit with the mouse, scrub the timeline |
| 9 | `source.py` | `NumpySource` in place, `CudaSource` stubbed |
| 10 | real shards | point it at the label pipeline's output |

**Step 5 is the risk milestone.** Everything before it is NumPy you can unit-test; everything
after is incremental shader work. Get one cube on screen before writing any more logic.

---

## Jetson notes (deferred, not forgotten)

- Orin supports GL 4.6; nothing here needs more than 4.1. No shader changes on deploy.
- Headless: `moderngl.create_standalone_context(backend="egl")`. Already required by the
  renderer tests, so it's exercised from day one.
- `CudaSource` via `cudaGraphicsGLRegisterBuffer` — only worth it if profiling shows the
  CPU upload is the bottleneck. At 8 k instances it will not be.
- Real constraint on Orin is the **network**, not the renderer: 8 GB shared across model,
  viewer, and OS. Plan on TensorRT FP16, fewer channels, possibly the 100×100×10 grid.
