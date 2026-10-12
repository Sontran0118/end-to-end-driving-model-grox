#version 410 core

// Per-vertex: the unit cube, uploaded once.
in vec3 in_pos;        // corner offsets in [-0.5, +0.5]
in vec3 in_normal;     // which way this face points

// Per-instance: one row of the [N, 6] buffer.
in vec3 in_center;     // voxel centre, metres
in vec3 in_color;      // rgb 0..1

uniform mat4 mvp;
uniform float voxel_size;

out vec3 v_color;
out vec3 v_normal;

void main() {
    v_color = in_color;
    v_normal = in_normal;
    gl_Position = mvp * vec4(in_center + in_pos * voxel_size, 1.0);
}
