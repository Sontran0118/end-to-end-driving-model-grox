#version 410 core

in vec3 v_color;
in vec3 v_normal;

out vec4 f_color;

void main() {
    vec3 n = abs(v_normal);
    float shade = n.z * 1.0 + (n.x + n.y) * 0.65;   // top bright, sides dim
    f_color = vec4(v_color * shade, 1.0);
}

