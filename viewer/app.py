"""Interactive viewer window. The only module that imports moderngl_window.

    python -m viewer.app
    python -m viewer.app --scene corridor --palette height_ramp
    python -m viewer.app --label label_000.npz --palette by_class

Keys: 1/2/3 palette, Q/W/E/R/T/Y scene, SPACE play/pause, LEFT/RIGHT step.
"""

import argparse
import sys

import moderngl_window as mglw
import numpy as np

from occgrid import Grid
from viewer import palette as palettes
from viewer import synthetic
from viewer.camera import OrbitCamera
from viewer.grid import build_instances
from viewer.renderer import VoxelRenderer

SCENES = {
    "single_voxel": synthetic.single_voxel,
    "ground_plane": synthetic.ground_plane,
    "corridor": synthetic.corridor,
    "checkerboard": synthetic.checkerboard,
    "random_fill": synthetic.random_fill,
    "driving": synthetic.driving,
}

PALETTES = {
    "flat": palettes.flat,
    "height_ramp": palettes.height_ramp,
    "by_class": palettes.by_class,
}


def load_label(path):
    """Load a Grid from a .npz written by paiav.make_label."""
    d = np.load(path)
    return Grid(
        occ=d["occ"],
        origin=d["origin"],
        voxel=float(d["voxel"]),
        sem=d["sem"] if "sem" in d else None,
        pose=d["pose"] if "pose" in d else None,
    )


class ViewerWindow(mglw.WindowConfig):
    gl_version = (4, 1)
    title = "occupancy viewer"
    window_size = (1280, 800)
    resizable = True

    scene_name = "corridor"
    palette_name = "height_ramp"
    label_path = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = VoxelRenderer(self.ctx)
        self.palette = PALETTES[self.palette_name]

        # frames is a list of Grids: one entry for a static scene or a label,
        # many for a playable sequence. on_render advances through it.
        self.playing = False
        self.fps = 10.0
        self._acc = 0.0

        if self.label_path:
            self.frames = [load_label(self.label_path)]
            self.camera = OrbitCamera(target=(28.0, 0.0, 0.0), distance=62.0, pitch=0.55)
            print(f"loaded {self.label_path}")
        elif self.scene_name == "driving":
            self.frames = synthetic.drive_sequence(n=120)
            self.camera = OrbitCamera(target=(20.0, 0.0, 0.0), distance=45.0, pitch=0.45)
            self.playing = True
        else:
            self.frames = [SCENES[self.scene_name]()]
            self.camera = OrbitCamera(target=(20.0, 0.0, 0.0), distance=45.0, pitch=0.45)

        self.set_frame(0)

    def set_frame(self, i):
        self.frame_i = i % len(self.frames)
        self.grid = self.frames[self.frame_i]
        self.refresh()

    def refresh(self):
        self.renderer.upload(build_instances(self.grid, self.palette))

    def frame_view(self):
        """Point the camera at the middle of the current grid and back off to fit."""
        extent = np.array(self.grid.occ.shape) * self.grid.voxel
        self.camera.target = self.grid.origin + extent / 2.0
        self.camera.distance = float(np.linalg.norm(extent)) * 0.8
        print(f"target={self.camera.target} distance={self.camera.distance:.1f}")

    def load_scene(self, name):
        self.frames = (synthetic.drive_sequence(n=120) if name == "driving"
                       else [SCENES[name]()])
        self.playing = name == "driving"
        self.set_frame(0)
        print(f"{name}: {self.renderer.n} instances, {len(self.frames)} frame(s)")

    def on_render(self, time, frametime):
        if self.playing and len(self.frames) > 1:
            self._acc += frametime
            if self._acc >= 1.0 / self.fps:
                self._acc = 0.0
                self.set_frame(self.frame_i + 1)
        aspect = self.wnd.viewport_size[0] / self.wnd.viewport_size[1]
        self.renderer.draw(self.camera.mvp(aspect), self.grid.voxel)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.camera.orbit(-dx * 0.005, dy * 0.005)

    ZOOM_STEP = 0.85

    def on_mouse_scroll_event(self, x_offset, y_offset):
        # Mac trackpads often report a two-finger swipe on the x axis, so take
        # whichever axis actually moved. A zero delta must not fall through to
        # the else branch, or idle events would slowly zoom out.
        delta = y_offset if abs(y_offset) >= abs(x_offset) else x_offset
        if delta == 0.0:
            return
        self.camera.zoom(self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP)

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        keys = self.wnd.keys

        if key == keys.SPACE:
            self.playing = not self.playing
            return
        if key in (keys.EQUAL, keys.UP):
            self.camera.zoom(self.ZOOM_STEP)
            return
        if key in (keys.MINUS, keys.DOWN):
            self.camera.zoom(1.0 / self.ZOOM_STEP)
            return
        if key == keys.F:
            self.frame_view()
            return
        if key == keys.LEFT:
            self.playing = False
            self.set_frame(self.frame_i - 1)
            return
        if key == keys.RIGHT:
            self.playing = False
            self.set_frame(self.frame_i + 1)
            return

        for k, name in zip((keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3), PALETTES):
            if key == k:
                self.palette = PALETTES[name]
                self.refresh()
                print(f"palette: {name}")
                return

        for k, name in zip((keys.Q, keys.W, keys.E, keys.R, keys.T, keys.Y), SCENES):
            if key == k:
                self.load_scene(name)
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="corridor", choices=SCENES)
    ap.add_argument("--palette", default="height_ramp", choices=PALETTES)
    ap.add_argument("--label", default=None, help="path to a .npz from paiav.make_label")
    # parse_known_args, not parse_args: moderngl_window reads sys.argv itself
    # and has its own flags (-wnd, -fs, ...). Ours are consumed here and the
    # rest handed on, so both parsers see only what they understand.
    args, rest = ap.parse_known_args()
    ViewerWindow.scene_name = args.scene
    ViewerWindow.palette_name = args.palette
    ViewerWindow.label_path = args.label
    sys.argv = [sys.argv[0]] + rest
    mglw.run_window_config(ViewerWindow)


if __name__ == "__main__":
    main()
