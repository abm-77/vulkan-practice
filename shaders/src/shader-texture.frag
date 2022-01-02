#version 450

#extension GL_ARB_separate_shader_objects: enable

layout (binding = 1) uniform sampler2D tex_sampler;

layout (location = 0) in vec4 frag_color;
layout (location = 1) in vec2 frag_tex_coords;

layout (location = 0) out vec4 out_color;

void main () {
	out_color = texture(tex_sampler, frag_tex_coords);
}
