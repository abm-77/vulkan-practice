#version 450

#extension GL_ARB_separate_shader_objects: enable

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;

layout (location = 0) out vec4 out_color;

out gl_PerVertex {
	vec4 gl_Position;
};

void main () {
	gl_Position = vec4(in_position, 1.0);
	out_color = in_color;
}