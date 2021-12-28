use ash::vk::{self};


use std::{
	cmp::max,
	ffi::CString,
	path::Path,
	ptr, mem::swap,
};

use super::*;

pub fn create_render_pass(
	device: &ash::Device,
	surface_format: vk::Format,
) -> vk::RenderPass {
	let color_attachment = vk::AttachmentDescription {
		format: surface_format,
		flags: vk::AttachmentDescriptionFlags::empty(),
		samples: vk::SampleCountFlags::TYPE_1,
		load_op: vk::AttachmentLoadOp::CLEAR,
		store_op: vk::AttachmentStoreOp::STORE,
		stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
		stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
		initial_layout: vk::ImageLayout::UNDEFINED,
		final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
	};

	let color_attachment_ref = vk::AttachmentReference {
		attachment: 0,
		layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	};

	let subpasses = [vk::SubpassDescription {
		color_attachment_count: 1,
		p_color_attachments: &color_attachment_ref,
		p_depth_stencil_attachment: ptr::null(),
		flags: vk::SubpassDescriptionFlags::empty(),
		pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
		input_attachment_count: 0,
		p_resolve_attachments: ptr::null(),
		p_input_attachments: ptr::null(),
		preserve_attachment_count: 0,
		p_preserve_attachments: ptr::null(),
	}];

	let render_pass_attachments = [color_attachment];

	let subpass_dependencies = [vk::SubpassDependency {
		src_subpass: vk::SUBPASS_EXTERNAL,
		dst_subpass: 0,
		src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
		dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
		src_access_mask: vk::AccessFlags::empty(),
		dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
		dependency_flags: vk::DependencyFlags::empty(),
	}];

	let render_pass_create_info = vk::RenderPassCreateInfo {
		s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
		flags: vk::RenderPassCreateFlags::empty(),
		p_next: ptr::null(),
		attachment_count: render_pass_attachments.len() as u32,
		p_attachments: render_pass_attachments.as_ptr(),
		subpass_count: subpasses.len() as u32,
		p_subpasses: subpasses.as_ptr(),
		dependency_count: subpass_dependencies.len() as u32,
		p_dependencies: subpass_dependencies.as_ptr(),
	};

	unsafe {
		device
			.create_render_pass(&render_pass_create_info, None)
			.expect("Failed to create render pass!")
	}
}
pub fn create_graphics_pipeline(
	device: &ash::Device, 
	render_pass: vk::RenderPass,
	swapchain_extent: vk::Extent2D,
	ubo_set_layout: vk::DescriptorSetLayout,
)  -> (vk::Pipeline, vk::PipelineLayout) {
	let vert_shader_code =
		tools::read_shader_code(Path::new("shaders/spv/shader-uniform-buffer.vert.spv"));
	let frag_shader_code =
		tools::read_shader_code(Path::new("shaders/spv/shader-uniform-buffer.frag.spv"));

	let vert_shader_module = create_shader_module(device, vert_shader_code);
	let frag_shader_module = create_shader_module(device, frag_shader_code);

	let main_function_name = CString::new("main").unwrap();

	let shader_stages = [
		// Vertex Shader
		vk::PipelineShaderStageCreateInfo {
			s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
			p_next: ptr::null(),
			flags: vk::PipelineShaderStageCreateFlags::empty(),
			module: vert_shader_module,
			p_name: main_function_name.as_ptr(),
			p_specialization_info: ptr::null(),
			stage: vk::ShaderStageFlags::VERTEX,
		},
		// Fragment Shader
		vk::PipelineShaderStageCreateInfo {
			s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
			p_next: ptr::null(),
			flags: vk::PipelineShaderStageCreateFlags::empty(),
			module: frag_shader_module,
			p_name: main_function_name.as_ptr(),
			p_specialization_info: ptr::null(),
			stage: vk::ShaderStageFlags::FRAGMENT,
		},
	];

	let binding_description = Vertex::get_binding_descriptions();
	let attribute_description = Vertex::get_attribute_descriptions();

	let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineVertexInputStateCreateFlags::empty(),
		vertex_attribute_description_count: attribute_description.len() as u32,
		p_vertex_attribute_descriptions: attribute_description.as_ptr(),
		vertex_binding_description_count: binding_description.len() as u32,
		p_vertex_binding_descriptions: binding_description.as_ptr(),
	};

	let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
		p_next: ptr::null(),
		primitive_restart_enable: vk::FALSE,
		topology: vk::PrimitiveTopology::TRIANGLE_LIST,
	};

	let viewports = [vk::Viewport {
		x: 0.0,
		y: 0.0,
		width: swapchain_extent.width as f32,
		height: swapchain_extent.height as f32,
		min_depth: 0.0,
		max_depth: 1.0,
	}];

	let scissors = [vk::Rect2D {
		offset: vk::Offset2D { x: 0, y: 0 },
		extent: swapchain_extent,
	}];

	let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineViewportStateCreateFlags::empty(),
		scissor_count: scissors.len() as u32,
		p_scissors: scissors.as_ptr(),
		viewport_count: viewports.len() as u32,
		p_viewports: viewports.as_ptr(),
	};

	let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineRasterizationStateCreateFlags::empty(),
		depth_clamp_enable: vk::FALSE,
		cull_mode: vk::CullModeFlags::BACK,
		front_face: vk::FrontFace::CLOCKWISE,
		line_width: 1.0,
		polygon_mode: vk::PolygonMode::FILL,
		rasterizer_discard_enable: vk::FALSE,
		depth_bias_clamp: 0.0,
		depth_bias_constant_factor: 0.0,
		depth_bias_enable: vk::FALSE,
		depth_bias_slope_factor: 0.0,
	};

	let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineMultisampleStateCreateFlags::empty(),
		rasterization_samples: vk::SampleCountFlags::TYPE_1,
		sample_shading_enable: vk::FALSE,
		min_sample_shading: 0.0,
		p_sample_mask: ptr::null(),
		alpha_to_one_enable: vk::FALSE,
		alpha_to_coverage_enable: vk::FALSE,
	};

	let stencil_state = vk::StencilOpState {
		fail_op: vk::StencilOp::KEEP,
		pass_op: vk::StencilOp::KEEP,
		depth_fail_op: vk::StencilOp::KEEP,
		compare_op: vk::CompareOp::ALWAYS,
		compare_mask: 0,
		write_mask: 0,
		reference: 0,
	};

	let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
		depth_test_enable: vk::FALSE,
		depth_write_enable: vk::FALSE,
		depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
		depth_bounds_test_enable: vk::FALSE,
		stencil_test_enable: vk::FALSE,
		front: stencil_state,
		back: stencil_state,
		max_depth_bounds: 1.0,
		min_depth_bounds: 0.0,
	};

	let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
		blend_enable: vk::TRUE,
		color_write_mask: 
			vk::ColorComponentFlags::R | 
			vk::ColorComponentFlags::G | 
			vk::ColorComponentFlags::B |
			vk::ColorComponentFlags::A,
		src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
		dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
		color_blend_op: vk::BlendOp::ADD,
		src_alpha_blend_factor: vk::BlendFactor::ONE,
		dst_alpha_blend_factor: vk::BlendFactor::ZERO,
		alpha_blend_op: vk::BlendOp::ADD,
	}];

	let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
		s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineColorBlendStateCreateFlags::empty(),
		logic_op_enable: vk::FALSE,
		logic_op: vk::LogicOp::COPY,
		attachment_count: color_blend_attachment_states.len() as u32,
		p_attachments: color_blend_attachment_states.as_ptr(),
		blend_constants: [0.0, 0.0, 0.0, 0.0],
	};

	let set_layouts = [ubo_set_layout];

	let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
		s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineLayoutCreateFlags::empty(),
		set_layout_count: set_layouts.len() as u32,
		p_set_layouts: set_layouts.as_ptr(),
		push_constant_range_count: 0,
		p_push_constant_ranges: ptr::null(),
	};

	let pipeline_layout = unsafe {
		device
			.create_pipeline_layout(&pipeline_layout_create_info, None)
			.expect("Failed to create pipeline layout!")
	};

	let graphics_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
		s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineCreateFlags::empty(),
		stage_count: shader_stages.len() as u32,
		p_stages: shader_stages.as_ptr(),
		p_vertex_input_state: &vertex_input_state_create_info,
		p_input_assembly_state: &vertex_input_assembly_state_info,
		p_tessellation_state: ptr::null(),
		p_viewport_state: &viewport_state_create_info,
		p_rasterization_state: &rasterization_state_create_info,
		p_multisample_state: &multisample_state_create_info,
		p_depth_stencil_state: &depth_state_create_info,
		p_color_blend_state: &color_blend_state,
		p_dynamic_state: ptr::null(),
		layout: pipeline_layout,
		render_pass: render_pass,
		subpass: 0,
		base_pipeline_handle: vk::Pipeline::null(),
		base_pipeline_index: -1,
	}];

	let graphics_pipelines = unsafe {
		device
			.create_graphics_pipelines(
				vk::PipelineCache::null(),
				 &graphics_pipeline_create_infos,
				 None
			)
			.expect("Failed to create Graphics Pipeline!")
	};

	unsafe {
		device.destroy_shader_module(vert_shader_module, None);
		device.destroy_shader_module(frag_shader_module, None);
	}
	
	(graphics_pipelines[0], pipeline_layout)
}

pub fn create_framebuffers(
	device: &ash::Device,
	render_pass: vk::RenderPass,
	image_views: &Vec<vk::ImageView>,
	swapchain_extent: vk::Extent2D,
) -> Vec<vk::Framebuffer> {
	let mut framebuffers = vec![];

	for &image_view in image_views.iter() {
		let attachments = [image_view];

		let framebuffer_create_info = vk::FramebufferCreateInfo {
			s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
			p_next: ptr::null(),
			flags: vk::FramebufferCreateFlags::empty(),
			render_pass: render_pass,
			attachment_count: attachments.len() as u32,
			p_attachments: attachments.as_ptr(),
			width: swapchain_extent.width, 
			height: swapchain_extent.height, 
			layers: 1,
		};

		let framebuffer = unsafe {
			device
				.create_framebuffer(&framebuffer_create_info, None)
				.expect("Failed to create Framebuffer!")
		};

		framebuffers.push(framebuffer);
	}

	framebuffers
}

pub fn create_command_pool(
	device: &ash::Device,
	queue_families: &QueueFamilyIndices,
) -> vk::CommandPool {
	let command_pool_create_info = vk::CommandPoolCreateInfo {
		s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::CommandPoolCreateFlags::empty(),
		queue_family_index: queue_families.graphics_family.unwrap(),
	};

	unsafe {
		device
			.create_command_pool(&command_pool_create_info, None)
			.expect("Failed to create Command Pool!")
	}
}

pub fn create_command_buffers(
	device: &ash::Device,
	command_pool: vk::CommandPool,
	graphics_pipeline: vk::Pipeline,
	framebuffers: &Vec<vk::Framebuffer>,
	render_pass: vk::RenderPass,
	surface_extent: vk::Extent2D,
	vertex_buffer: vk::Buffer,
	index_buffer: vk::Buffer,
	pipeline_layout: vk::PipelineLayout,
	descriptor_sets: &Vec<vk::DescriptorSet>,
) -> Vec<vk::CommandBuffer> {
	let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
		s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
		p_next: ptr::null(),
		command_buffer_count: framebuffers.len() as u32,
		command_pool: command_pool,
		level: vk::CommandBufferLevel::PRIMARY,
	};

	let command_buffers = unsafe {
		device
			.allocate_command_buffers(&command_buffer_allocate_info)
			.expect("Failed to allocate Command Buffers!")
	};

	for (i, &command_buffer) in command_buffers.iter().enumerate() {
		let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            p_inheritance_info: ptr::null(),
            flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        };

		unsafe {
			device
				.begin_command_buffer(command_buffer, &command_buffer_begin_info)
				.expect("Failed to begin Command Buffer at beginning!")
		} 

		let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass: render_pass,
            framebuffer: framebuffers[i],
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: surface_extent,
            },
        };

		unsafe {
			device.cmd_begin_render_pass(
				command_buffer,
				&render_pass_begin_info,
				vk::SubpassContents::INLINE,
			);

			device.cmd_bind_pipeline(
				command_buffer,
				vk::PipelineBindPoint::GRAPHICS,
				graphics_pipeline,
			);

			let vertex_buffers = [vertex_buffer];
			let offsets = [0_u64];
			let descriptor_sets_to_bind = [descriptor_sets[i]];

			device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
			device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);
			device.cmd_bind_descriptor_sets(
			 	command_buffer, 
				vk::PipelineBindPoint::GRAPHICS, 
				pipeline_layout, 
				0, 
				&descriptor_sets_to_bind, 
				&[]
			);

			device.cmd_draw_indexed(command_buffer, RECT_INDICES_DATA.len() as u32, 1, 0, 0, 0);

			device.cmd_end_render_pass(command_buffer);

			device
				.end_command_buffer(command_buffer)
				.expect("Failed to record Command Buffer at ending!");
		}

	}

	command_buffers
}

pub fn create_sync_objects(device: &ash::Device, max_frame_in_flight: usize) -> SyncObjects {
	let mut sync_objects = SyncObjects {
		image_available_semaphores: vec![],
		render_finished_semaphores: vec![],
		in_flight_fences: vec![],
	};

	let semaphore_create_info = vk::SemaphoreCreateInfo {
		s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::SemaphoreCreateFlags::empty(),
	};

	let fence_create_info = vk::FenceCreateInfo {
		s_type: vk::StructureType::FENCE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::FenceCreateFlags::SIGNALED,
	};

	for _ in 0..max_frame_in_flight {
		unsafe {
			let image_available_semaphore = device
				.create_semaphore(&semaphore_create_info, None)
				.expect("Failed to create Semaphore Object!");
			let render_finished_semaphore = device
				.create_semaphore(&semaphore_create_info, None)
				.expect("Failed to create Semaphore Object!");
			let in_flight_fence = device
				.create_fence(&fence_create_info, None)
				.expect("Failed to create Fence Object!");
			
				sync_objects.image_available_semaphores.push(image_available_semaphore);
				sync_objects.render_finished_semaphores.push(render_finished_semaphore);
				sync_objects.in_flight_fences.push(in_flight_fence);

		}
	}

	sync_objects
}

pub fn create_vertex_buffer<T>(
	device: &ash::Device,
	device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
	command_pool: vk::CommandPool,
	submit_queue: vk::Queue,
	data: &[T],
) -> (vk::Buffer, vk::DeviceMemory) {
	let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

	let (staging_buffer, staging_buffer_memory) = create_buffer(
		device,
		buffer_size,
		vk::BufferUsageFlags::TRANSFER_SRC,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		&device_memory_properties,
	);	

	unsafe {
		let data_ptr = device
			.map_memory(
				staging_buffer_memory, 
				0, 
				buffer_size, 
				vk::MemoryMapFlags::empty()
			)
			.expect("Failed to Map Memory") as *mut T;

		data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
		device.unmap_memory(staging_buffer_memory);
	}

		let (vertex_buffer, vertex_buffer_memory) = create_buffer(
			device,
			buffer_size,
			vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
			&device_memory_properties,
		);

		copy_buffer(
			device,
			submit_queue,
			command_pool,
			staging_buffer,
			vertex_buffer,
			buffer_size,
		);

		unsafe {
			device.destroy_buffer(staging_buffer, None);
			device.free_memory(staging_buffer_memory, None);
		}

		(vertex_buffer, vertex_buffer_memory)
}

pub fn create_index_buffer(
	device: &ash::Device,
	device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
	command_pool: vk::CommandPool,
	submit_queue: vk::Queue,
	data: &[u32],
) -> (vk::Buffer, vk::DeviceMemory) {
	let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

	let (staging_buffer, staging_buffer_memory) = create_buffer(
		device,
		buffer_size,
		vk::BufferUsageFlags::TRANSFER_SRC,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		&device_memory_properties,
	);	

	unsafe {
		let data_ptr = device
			.map_memory(
				staging_buffer_memory, 
				0, 
				buffer_size, 
				vk::MemoryMapFlags::empty()
			)
			.expect("Failed to Map Memory") as *mut u32;

		data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
		device.unmap_memory(staging_buffer_memory);
	}

		let (index_buffer, index_buffer_memory) = create_buffer(
			device,
			buffer_size,
			vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
			&device_memory_properties,
		);

		copy_buffer(
			device,
			submit_queue,
			command_pool,
			staging_buffer,
			index_buffer,
			buffer_size,
		);

		unsafe {
			device.destroy_buffer(staging_buffer, None);
			device.free_memory(staging_buffer_memory, None);
		}

		(index_buffer, index_buffer_memory)
}

