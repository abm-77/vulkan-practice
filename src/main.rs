use vulkan_tutorial::{
	utility, 
	utility::{
		constants::*, 
		share, 
		structures::*,
		window::*,
	}
};


use ash::vk;

use cgmath::{Deg, Matrix4, Vector3};

use std::{
	ptr,
	path::Path,
};


const TEXTURE_PATH: &'static str = "textures/texture.jpg";

struct App {
	window: winit::window::Window,

	instance: ash::Instance,
	_entry: ash::Entry,
	surface_loader: ash::extensions::khr::Surface,
	surface: vk::SurfaceKHR,
	debug_utils_loader: ash::extensions::ext::DebugUtils,
	debug_messenger: vk::DebugUtilsMessengerEXT,

	physical_device: vk::PhysicalDevice,
	device: ash::Device, // logical device

	queue_family: QueueFamilyIndices,
	graphics_queue: vk::Queue,
	present_queue: vk::Queue,

	swapchain_loader: ash::extensions::khr::Swapchain,
	swapchain: vk::SwapchainKHR,
	swapchain_images: Vec<vk::Image>,
	swapchain_format: vk::Format,
	swapchain_extent: vk::Extent2D,
	swapchain_imageviews: Vec<vk::ImageView>,
	swapchain_framebuffers: Vec<vk::Framebuffer>,

	render_pass: vk::RenderPass,
	pipeline_layout: vk::PipelineLayout,
	ubo_layout: vk::DescriptorSetLayout,
	graphics_pipeline: vk::Pipeline,

	texture_image: vk::Image,
	texture_image_memory: vk::DeviceMemory,
	texture_sampler: vk::Sampler,
	texture_image_view: vk::ImageView,

	vertex_buffer: vk::Buffer,
	vertex_buffer_memory: vk::DeviceMemory,
	index_buffer: vk::Buffer,
	index_buffer_memory: vk::DeviceMemory,

	uniform_transform: UniformBufferObject,
	uniform_buffers: Vec<vk::Buffer>,
	uniform_buffers_memory: Vec<vk::DeviceMemory>,

	descriptor_pool: vk::DescriptorPool,
	descriptor_sets: Vec<vk::DescriptorSet>,

	command_pool: vk::CommandPool,
	command_buffers: Vec<vk::CommandBuffer>,

	image_available_semaphores: Vec<vk::Semaphore>,
	render_finished_semaphores: Vec<vk::Semaphore>,
	in_flight_fences: Vec<vk::Fence>,
	current_frame: usize,

	is_frame_buffer_resized: bool,
}

impl App {
	pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> App {
		// window
		let window = utility::window::init_window(event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

		// init vulkan 
	 	let entry = ash::Entry::new();
		let instance = share::create_instance(
			&entry, 
			WINDOW_TITLE, 
			VALIDATION.is_enable, 
			&VALIDATION.required_validation_layers.to_vec()
		);

		// init debug
		let (debug_utils_loader, debug_messenger) = 
			utility::debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
		
		// surface information
		let surface_info = share::create_surface(&entry, &instance, &window, WINDOW_WIDTH, WINDOW_HEIGHT);

		// get physical device
		let physical_device = share::pick_physical_device(&instance, &surface_info, &DEVICE_EXTENSIONS);
		let physical_device_memory_properties = 
			unsafe { instance.get_physical_device_memory_properties(physical_device) };

		// get logical device
		let (logical_device, family_indices) = share::create_logical_device(
			&instance, 
			physical_device, 
			&VALIDATION,
			&DEVICE_EXTENSIONS,
			&surface_info,
		);

		// get queues
		let graphics_queue = 
			unsafe { logical_device.get_device_queue(family_indices.graphics_family.unwrap(), 0)};
		let present_queue = 
			unsafe { logical_device.get_device_queue(family_indices.present_family.unwrap(), 0)};

		// swapchain
		let swapchain_info = share::create_swapchain(
			&instance,
			&logical_device,
			physical_device,
			&window,
			&surface_info,
			&family_indices,
		);

		let swapchain_imageviews = share::v1::create_image_views(
			&logical_device,
			swapchain_info.swapchain_format,
			&swapchain_info.swapchain_images,
		);

		// graphics pipeline
		let render_pass = share::v1::create_render_pass(&logical_device, swapchain_info.swapchain_format);
		let ubo_layout = share::v1::create_descriptor_set_layout(&logical_device);
		let (graphics_pipeline, pipeline_layout) = share::v1::create_graphics_pipeline(
			&logical_device, 
			render_pass,
			swapchain_info.swapchain_extent,
			ubo_layout,
		);

		let swapchain_framebuffers = share::v1::create_framebuffers(
			&logical_device,
			render_pass,
			&swapchain_imageviews,
			swapchain_info.swapchain_extent,
		);

		// buffers
		let command_pool = share::v1::create_command_pool(&logical_device, &family_indices);

		let (texture_image, texture_memory) = share::v1::create_texture_image(
			&logical_device,
			command_pool,
			graphics_queue,
			&physical_device_memory_properties,
			&Path::new(TEXTURE_PATH),
		);

		let texture_image_view = share::v1::create_texture_image_view(&logical_device, texture_image, 1);
		let texture_sampler = share::v1::create_texture_sampler(&logical_device);

		let (vertex_buffer, vertex_buffer_memory) = utility::share::v1::create_vertex_buffer(
			&logical_device,
			&physical_device_memory_properties,
			command_pool,
			graphics_queue,
			&RECT_VERTICES_DATA,
		);

		let (index_buffer, index_buffer_memory) = utility::share::v1::create_index_buffer(
			&logical_device,
			&physical_device_memory_properties,
			command_pool,
			graphics_queue,
			&RECT_INDICES_DATA,
		);

		let (uniform_buffers, unfiform_buffers_memory) = share::v1::create_uniform_buffers(
			&logical_device,
			&physical_device_memory_properties,
			swapchain_info.swapchain_images.len(),
		);

		let descriptor_pool = share::v1::create_descriptor_pool(&logical_device, swapchain_info.swapchain_images.len());
		let descriptor_sets = share::v1::create_descriptor_sets(
			&logical_device,
			descriptor_pool,
			ubo_layout,
			&uniform_buffers,
			texture_image_view,
			texture_sampler,
			swapchain_info.swapchain_images.len(),
		);

		let command_buffers = share::v1::create_command_buffers(
			&logical_device,
			command_pool,
			graphics_pipeline,
			&swapchain_framebuffers,
			render_pass,
			swapchain_info.swapchain_extent,
			vertex_buffer,
			index_buffer,
			pipeline_layout,
			&descriptor_sets,
		);

		// synchronization
		let sync_objects = share::v1::create_sync_objects(&logical_device, MAX_FRAMES_IN_FLIGHT);
		
		App { 
			window: window,
			_entry: entry, 
			instance: instance, 
			debug_utils_loader: debug_utils_loader, 
			debug_messenger: debug_messenger,
			surface: surface_info.surface,
			surface_loader: surface_info.surface_loader,
			physical_device,
			device: logical_device,
			queue_family: family_indices,
			graphics_queue: graphics_queue,
			present_queue: present_queue,

			swapchain_loader: swapchain_info.swapchain_loader,
			swapchain: swapchain_info.swapchain,
			swapchain_images: swapchain_info.swapchain_images,
			swapchain_format: swapchain_info.swapchain_format,
			swapchain_extent: swapchain_info.swapchain_extent,
			swapchain_imageviews: swapchain_imageviews,
			swapchain_framebuffers: swapchain_framebuffers,

			render_pass: render_pass,
			pipeline_layout: pipeline_layout,
			ubo_layout: ubo_layout,
			graphics_pipeline: graphics_pipeline,

			texture_image: texture_image,
			texture_image_view: texture_image_view,
			texture_sampler: texture_sampler,
			texture_image_memory: texture_memory,

			vertex_buffer: vertex_buffer,
			vertex_buffer_memory: vertex_buffer_memory,
			index_buffer: index_buffer,
			index_buffer_memory: index_buffer_memory,

			uniform_transform: UniformBufferObject {
				model: cgmath::prelude::SquareMatrix::identity(),
				view: cgmath::prelude::SquareMatrix::identity(),
				proj: cgmath::prelude::SquareMatrix::identity(),
			},
			uniform_buffers: uniform_buffers,
			uniform_buffers_memory: unfiform_buffers_memory,

			descriptor_pool: descriptor_pool,
            descriptor_sets: descriptor_sets,

			command_pool: command_pool,
			command_buffers: command_buffers,

			image_available_semaphores: sync_objects.image_available_semaphores,
			render_finished_semaphores: sync_objects.render_finished_semaphores,
			in_flight_fences: sync_objects.in_flight_fences,
			current_frame: 0,

			is_frame_buffer_resized: false,
		}
	}

	fn update_uniform_buffer(&mut self, current_image: usize, delta_time: f32) {
		self.uniform_transform.model =
            Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Deg(90.0) * delta_time)
                * self.uniform_transform.model;

        let ubos = [self.uniform_transform.clone()];

        let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(
                        self.uniform_buffers_memory[current_image],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to Map Memory") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
	}
}

impl Drop for App {
	fn drop(&mut self) {
		unsafe {
			self.cleanup_swapchain();

			self.device.destroy_image(self.texture_image, None);
			self.device.free_memory(self.texture_image_memory, None);

			self.device.destroy_descriptor_set_layout(self.ubo_layout, None);

			for i in 0..MAX_FRAMES_IN_FLIGHT {
				self.device
					.destroy_semaphore(self.image_available_semaphores[i], None);
				self.device
					.destroy_semaphore(self.render_finished_semaphores[i], None);
				self.device
					.destroy_fence(self.in_flight_fences[i], None);
			}

			self.device.destroy_buffer(self.vertex_buffer, None);
			self.device.free_memory(self.vertex_buffer_memory, None);

			self.device.destroy_buffer(self.index_buffer, None);
			self.device.free_memory(self.index_buffer_memory, None);

			self.device.destroy_sampler(self.texture_sampler, None);
			self.device.destroy_image_view(self.texture_image_view, None);
			
			self.device.destroy_command_pool(self.command_pool, None);

			self.device.destroy_device(None);
			self.surface_loader.destroy_surface(self.surface, None);

			if VALIDATION.is_enable {
				self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
			}
			self.instance.destroy_instance(None);
		}
	}
}

impl VulkanApp for App {
	fn draw_frame (&mut self, delta_time: f32) {
		let wait_fences = [self.in_flight_fences[self.current_frame]];

		unsafe {
			self.device
				.wait_for_fences(&wait_fences, true, std::u64::MAX)
				.expect("Failed to wait for fence!");
		}

		let (image_index, _is_sub_optimal) = unsafe {
			let result = self.swapchain_loader
				.acquire_next_image(
					self.swapchain,
					std::u64::MAX, 
					self.image_available_semaphores[self.current_frame],
					vk::Fence::null()
			);

			match result {
				Ok(image_index) => image_index,
				Err(vk_result) => match vk_result {
					vk::Result::ERROR_OUT_OF_DATE_KHR => {
						self.recreate_swapchain();
						return;
					},
					_ => panic!("Failed to acquire Swap Chain Image!"),
				},
			}
		};

		self.update_uniform_buffer(image_index as usize, delta_time);

		let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
		let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
		let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

		let submit_infos = [vk::SubmitInfo {
			s_type: vk::StructureType::SUBMIT_INFO,
			p_next: ptr::null(),
			wait_semaphore_count: wait_semaphores.len() as u32,
			p_wait_semaphores: wait_semaphores.as_ptr(),
			p_wait_dst_stage_mask: wait_stages.as_ptr(),
			command_buffer_count: 1,
			p_command_buffers: &self.command_buffers[image_index as usize],
			signal_semaphore_count: signal_semaphores.len() as u32,
			p_signal_semaphores: signal_semaphores.as_ptr(),
		}];

		unsafe {
			self.device
				.reset_fences(&wait_fences)
				.expect("Failed to reset Fence!");

			self.device
				.queue_submit(
					self.graphics_queue,
					&submit_infos,
					self.in_flight_fences[self.current_frame],
				)
				.expect("Failed to execture queue submit.");
		}

		let swapchains = [self.swapchain];

		let present_info = vk::PresentInfoKHR {
			s_type: vk::StructureType::PRESENT_INFO_KHR,
			p_next: ptr::null(),
			wait_semaphore_count: signal_semaphores.len() as u32,
			p_wait_semaphores: signal_semaphores.as_ptr(),
			swapchain_count: swapchains.len() as u32,
			p_swapchains: swapchains.as_ptr(),
			p_image_indices: &image_index,
			p_results: ptr::null_mut(),
		};

		let result = unsafe {
			self.swapchain_loader
				.queue_present(self.present_queue, &present_info)
		};

		let is_resized = match result {
			Ok(_) => self.is_frame_buffer_resized,
			Err(vk_result) => match vk_result {
				vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
				_ => panic!("Failed to execute queue present!"),	
			},
		};

		if is_resized {
			self.is_frame_buffer_resized = false;
			self.recreate_swapchain();
		}

		self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	fn recreate_swapchain(&mut self) {
		let surface_info = SurfaceInfo {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
            screen_width: WINDOW_WIDTH,
            screen_height: WINDOW_HEIGHT,
        };	

		// TODO(bryson): see if there is a bettwe way to do this?
		// don't recreate swapchain if window was minimized
		let dimensionless_window = {
			let size = self.window.inner_size();
			size.width == 0 || size.height == 0
		};
		if dimensionless_window { return; } 


		unsafe {
			self.device
				.device_wait_idle()
				.expect("Failed to wait device idle!")
		};
		self.cleanup_swapchain();

		let swapchain_info = share::create_swapchain (
			&self.instance,
			&self.device,
			self.physical_device,
			&self.window,
			&surface_info,
			&self.queue_family,
		);

		self.swapchain_loader = swapchain_info.swapchain_loader;
		self.swapchain = swapchain_info.swapchain;
		self.swapchain_images = swapchain_info.swapchain_images;
		self.swapchain_format = swapchain_info.swapchain_format;
		self.swapchain_extent = swapchain_info.swapchain_extent;

		self.swapchain_imageviews = share::v1::create_image_views(
			&self.device,
			self.swapchain_format,
			&self.swapchain_images,
		);
		self.render_pass = share::v1::create_render_pass(&self.device, self.swapchain_format);
		let (graphics_pipeline, pipeline_layout) = share::v1::create_graphics_pipeline(
			&self.device, 
			self.render_pass, 
			swapchain_info.swapchain_extent,
			self.ubo_layout,
		);
		self.graphics_pipeline = graphics_pipeline;
		self.pipeline_layout = pipeline_layout;
		
		self.swapchain_framebuffers = share::v1::create_framebuffers(
			&self.device,
			self.render_pass,
			&self.swapchain_imageviews,
			self.swapchain_extent,
		);

		let mem_props = unsafe {
			self.instance.get_physical_device_memory_properties(self.physical_device)
		};

		let (uniform_buffers, uniform_buffers_memory) = share::v1::create_uniform_buffers(
			&self.device, 
			&mem_props,
			self.swapchain_images.len()
		);
		self.uniform_buffers = uniform_buffers;
		self.uniform_buffers_memory = uniform_buffers_memory;

		self.descriptor_pool = share::v1::create_descriptor_pool(&self.device, self.swapchain_images.len());
		self.descriptor_sets = share::v1::create_descriptor_sets(
			&self.device,
			self.descriptor_pool,
			self.ubo_layout,
			&self.uniform_buffers,
			self.texture_image_view,
			self.texture_sampler,
			self.swapchain_images.len()
		);

		self.command_buffers = share::v1::create_command_buffers(
			&self.device,
			self.command_pool,
			self.graphics_pipeline,
			&self.swapchain_framebuffers,
			self.render_pass,
			self.swapchain_extent,
			self.vertex_buffer,
			self.index_buffer,
			self.pipeline_layout,
			&self.descriptor_sets,
		);
	}

	fn cleanup_swapchain(&mut self) {
		unsafe {
			self.device
				.free_command_buffers(self.command_pool, &self.command_buffers);
			for &framebuffer in self.swapchain_framebuffers.iter() {
				self.device.destroy_framebuffer(framebuffer, None);
			}
			self.device.destroy_pipeline(self.graphics_pipeline, None);
			self.device.destroy_pipeline_layout(self.pipeline_layout, None);
			self.device.destroy_render_pass(self.render_pass, None);
			for &imageview in self.swapchain_imageviews.iter() {
				self.device.destroy_image_view(imageview, None);
			}
			self.swapchain_loader.destroy_swapchain(self.swapchain, None);

			for i in 0..self.uniform_buffers.len() {
				self.device.destroy_buffer(self.uniform_buffers[i], None);
				self.device.free_memory(self.uniform_buffers_memory[i], None);
			}

			self.device.destroy_descriptor_pool(self.descriptor_pool, None);
		}
	}

	 fn wait_device_idle(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

	fn resize_framebuffer(&mut self) {
		self.is_frame_buffer_resized = true;
	}

	fn window_ref(&self) -> &winit::window::Window {
		&self.window
	}
}

fn main() {
	let program_proc = ProgramProc::new();
	let vulkan_app = App::new(&program_proc.event_loop);
	program_proc.main_loop(vulkan_app);
}
