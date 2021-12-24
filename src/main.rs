use vulkan_tutorial::{
	utility, 
	utility::{
		constants::*, 
		debug::*, 
		share, 
	}
};

use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState},
    event_loop::{ControlFlow, EventLoop},
    window::{Window},
};

use ash::vk;

use std::{
	ffi::CString,
	ptr,
	os::raw::c_char,
	collections::HashSet,
};

struct VulkanApp {
	instance: ash::Instance,
	_entry: ash::Entry,
	surface_loader: ash::extensions::khr::Surface,
	surface: vk::SurfaceKHR,
	debug_utils_loader: ash::extensions::ext::DebugUtils,
	debug_messenger: vk::DebugUtilsMessengerEXT,

	_physical_device: vk::PhysicalDevice,
	device: ash::Device, // logical device

	_graphics_queue: vk::Queue,
	_present_queue: vk::Queue,

	swapchain_loader: ash::extensions::khr::Swapchain,
	swapchain: vk::SwapchainKHR,
	_swapchain_images: Vec<vk::Image>,
	_swapchain_format: vk::Format,
	_swapchain_extent: vk::Extent2D,
	swapchain_imageviews: Vec<vk::ImageView>,
}

impl VulkanApp {
	pub fn new(window: &Window) -> VulkanApp {
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
		let surface_info = share::create_surface(&entry, &instance, &window);

		// get physical device
		let physical_device = share::pick_physical_device(&instance, &surface_info);

		// get logical device
		let (logical_device, family_indices) = share::create_logical_device(
			&instance, 
			physical_device, 
			&VALIDATION,
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
			&surface_info,
			&family_indices,
		);

		let swapchain_imageviews = VulkanApp::create_image_views(
			&logical_device,
			swapchain_info.swapchain_format,
			&swapchain_info.swapchain_images,
		);

		VulkanApp { 
			_entry: entry, 
			instance: instance, 
			debug_utils_loader: debug_utils_loader, 
			debug_messenger: debug_messenger,
			surface: surface_info.surface,
			surface_loader: surface_info.surface_loader,
			_physical_device: physical_device,
			device: logical_device,
			_graphics_queue: graphics_queue,
			_present_queue: present_queue,

			swapchain_loader: swapchain_info.swapchain_loader,
			swapchain: swapchain_info.swapchain,
			_swapchain_images: swapchain_info.swapchain_images,
			_swapchain_format: swapchain_info.swapchain_format,
			_swapchain_extent: swapchain_info.swapchain_extent,
			swapchain_imageviews: swapchain_imageviews,
		}
	}

	fn draw_frame (&mut self) {

	}

	fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {
		event_loop.run(move |event, _, control_flow| {
			match event {
				Event::WindowEvent {event, ..} => {
					match event {
						WindowEvent::CloseRequested => {
							*control_flow = ControlFlow::Exit
						},
						WindowEvent::KeyboardInput { input, ..} => {
							match input {
								KeyboardInput { virtual_keycode, state, .. } => {
									match (virtual_keycode, state) {
										(Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
											dbg!();
											*control_flow = ControlFlow::Exit
										},
										_ => {},
									}
								},
							}
						},
						_ => {},
					}
				},
				Event::MainEventsCleared => {
					window.request_redraw();
				},
				Event::RedrawRequested(_window_id) => {
					self.draw_frame();
				},
				_ => (),
			}
		});
	}

}

impl Drop for VulkanApp {
	fn drop(&mut self) {
		unsafe {
			for &imageview in self.swapchain_imageviews.iter() {
				self.device.destroy_image_view(imageview, None);
			}
			self.swapchain_loader.destroy_swapchain(self.swapchain, None);
			self.device.destroy_device(None);
			self.surface_loader.destroy_surface(self.surface, None);
			if VALIDATION.is_enable {
				self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
			}
			self.instance.destroy_instance(None);
		}
	}
}

fn main() {
    let event_loop = EventLoop::new();
    let window = utility::window::init_window(&event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);
	let vulkan_app = VulkanApp::new(&window);
	vulkan_app.main_loop(event_loop, window);
}
