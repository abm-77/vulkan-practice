use vulkan_tutorial::{
	utility, 
	utility::{constants::*, debug::ValidationInfo, share, structures::*}
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
	_swapchain_extent: vk::Exten2D,
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
		let surface_info = VulkanApp::create_surface(&entry, &instance, &window);

		// get physical device
		let physical_device = VulkanApp::pick_physical_device(&instance, &surface_info);

		// get logical device
		let (logical_device, family_indices) = VulkanApp::create_logical_device(
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
		let swapchain_info = VulkanApp::create_swapchain(
			&instance,
			&logical_device,
			physical_device,
			&surface_info,
			&family_indices,
		);

		VulkanApp { 
			instance: instance, 
			_entry: entry, 
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
			_swapchain_extent: swapchain_info.swapchain_exten,
		}
	}

	fn create_surface (
		entry: &ash::Entry,
		instance: &ash::Instance,
		window: &Window
	) -> SurfaceInfo {
		let surface = unsafe {
			utility::platforms::create_surface(entry, instance, window)
				.expect("Failed to create window")
		};

		let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

		SurfaceInfo {
			surface_loader,
			surface,
		}
	}

	fn pick_physical_device(instance: &ash::Instance, surface_info: &SurfaceInfo) -> vk::PhysicalDevice {
		let physical_devices = unsafe {
			instance
				.enumerate_physical_devices()
				.expect("Failed to enumerate Physical Devices!")
		};

		let result = physical_devices.iter().find(|physical_device| {
			VulkanApp::is_physical_device_suitable(instance, **physical_device, surface_info)
		});

		match result {
			Some(physical_device) => *physical_device,
			None => panic!("Failed to find a suitable GPU!"),
		}
	}

	fn is_physical_device_suitable (
		instance: &ash::Instance, 
		physical_device: vk::PhysicalDevice,
		surface_info: &SurfaceInfo,
	) -> bool {
		let _device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
		let _device_features = unsafe { instance.get_physical_device_features(physical_device) };
		let indices = VulkanApp::find_queue_family(instance, physical_device, surface_info);
		return indices.is_complete();
	}

	fn find_queue_family(
		instance: &ash::Instance, 
		physical_device: vk::PhysicalDevice,
		surface_info: &SurfaceInfo,
	) -> QueueFamilyIndices {
		let queue_families = unsafe {instance.get_physical_device_queue_family_properties(physical_device)};

		let mut queue_family_indices = QueueFamilyIndices::new(); 

		let mut index = 0;
		for queue_family in queue_families.iter() {
			if queue_family.queue_count > 0 
				&& queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
					queue_family_indices.graphics_family = Some(index);
			}

			let is_present_support = unsafe {
				surface_info
					.surface_loader
					.get_physical_device_surface_support(physical_device, index as u32, surface_info.surface).unwrap()
			};

			if queue_family.queue_count > 0 && is_present_support {
				queue_family_indices.present_family = Some(index);
			}

			if queue_family_indices.is_complete() {
				break;
			}

			index += 1;
		}

		queue_family_indices
	}

	fn check_device_extension_support(
		instance: &ash::Instance,
		physical_device: vk::PhysicalDevice,
	) -> bool {
		let available_extensions = unsafe {
			instance
				.enumerate_device_extension_properties(physical_device)
				.expect("Failed to get device exetension properties.")
		};

		let mut available_extension_names = vec![];

		println!("\tAvailable Device Extensions: ");
		for extension in available_extensions.iter() {
			let extension_name = utility::tools::vk_to_string(&extension.extension_name);
			println!("\t\tName: {}, Version: {}", extension_name, extension.spec_version);
			available_extension_names.push(extension_name);
		}

		let mut required_extensions = HashSet::new();
		for extension in DEVICE_EXTENSIONS.names.iter() {
			required_extensions.insert(extension.to_string());
		}

		for extension_name in available_extension_names.iter() {
			required_extensions.remove(extension_name);
		}

		required_extensions.is_empty()
	}

	fn query_swapchain_support(
		physical_device: vk::PhysicalDevice,
		surface_info: &SurfaceInfo,
	) -> SwapChainSupportDetail {
		unsafe {
			let capabilities = surface_info
				.surface_loader
				.get_physical_device_surface_capabilities(physical_device, surface_info.surface)
				.expect("Failed to query for surface capabilities.");
			let formats = surface_info
				.surface_loader
				.get_physical_device_surface_formats(physical_device, surface_info.surface)	
				.expect("Failed to query for surface formats.");
			let present_modes = surface_info
				.surface_loader
				.get_physical_device_surface_present_modes(physical_device, surface_info.surface)
				.expect("Failed to query for surface prestn mode.");

			SwapChainSupportDetail {
				capabilities,
				formats,
				present_modes
			}
		}
	}

	fn create_swapchain(
		instance: &ash::Instance,
		device: &ash::Device,
		physical_device: vk::PhysicalDevice,
		surface_info: &SurfaceInfo,
		queue_family: &QueueFamilyIndices,
	) -> SwapchainInfo {
		let swapchain_support = VulkanApp::query_swapchain_support(physical_device, surface_info);

		let surface_format = VulkanApp::choose_swapchain_format(physical_device, surface_info);
		let present_mode = VulkanApp::choose_swapchain_format(&swapchain_formats);
		let extent = Vulkan::choose_swapchain_extent(&swapchain_support.capabilities);

		let image_count = swapchain_support.capabilities.min_image_count + 1;
		let image_count = if swapchain_support.capabilities.max_image_count > 0 {
			image_count.min(swapchain_support.capabilities.max_image_count)
		} else{
			image_count
		};

		let (image_sharing_mode, queue_family_index_count, queue_familiy_indices) =
			if queue_family.graphics_family != queue_family.present_family {
				(
					vk::SharingMode::EXCLUSIVE,
					2,
					vec![
						queue_family.graphics_family.unwrap(),
						queue_family.present_family.unwrap(),
					],
				)
			} else {
				(vk::SharingMode::EXCLUSIVE, 0, vec![])
		};

		let swapchain_create_info = vk::SwapchainCreateInfoKHR {
			s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
			p_next: ptr::null(),
			flags: vk::SwapchainCreateFlagsKHR::empty(),
			surface: surface_info.surface,
			min_image_count: image_count,
			image_color_space: surface_format.color_space,
			image_format: surface_format.format,
			image_extent: extent,
			image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
			image_sharing_mode,
			p_queue_family_indices: queue_familiy_indices.as_ptr(),
			queue_family_index_count,
			pre_transform: swapchain_support.capabilities.current_transform,
			composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
			present_mode,
			clipped: vk::TRUE,
			old_swapchain: vk::SwapchainKHR::null(),
			image_array_layers: 1,
		};

		let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
		let swapchain = unsafe {
			swapchain_loader
				.create_swapchain(&swapchain_create_info, None)
				.expect("Failed to create Swapchain!")
		};
		let swapchain_images = unsafe {
			swapchain_loader
				.get_swapchain_images(swapchain)
				.expect("Failed to get Swapchain Images.")
		};

		SwapChainInfo {
			swapchain_loader,
			swapchain,
			swapchain_format: surface_format.format,
			swapchain_extent: extent,
			swapchain_images
		}
	}


	fn create_logical_device(
		instance: &ash::Instance,
		physical_device: vk::PhysicalDevice,
		validation: &ValidationInfo,
		surface_info: &SurfaceInfo,
	) -> (ash::Device, QueueFamilyIndices) {
		let indices = VulkanApp::find_queue_family(instance, physical_device, surface_info);

		let mut unique_queue_families = HashSet::new();
		unique_queue_families.insert(indices.graphics_family.unwrap());
		unique_queue_families.insert(indices.present_family.unwrap());

		let queue_priorities = [1.0_f32];
		let mut queue_create_infos = vec![];

		for &queue_family in unique_queue_families.iter() {
			let queue_create_info = vk::DeviceQueueCreateInfo {
				s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
				p_next: ptr::null(),
				flags: vk::DeviceQueueCreateFlags::empty(),
				queue_family_index: queue_family,
				p_queue_priorities: queue_priorities.as_ptr(),
				queue_count: queue_priorities.len() as u32,
			};
			queue_create_infos.push(queue_create_info);
		}

		let physical_device_features = vk::PhysicalDeviceFeatures {
			..Default::default() // by defualt, enable no features
		};

		let required_validation_layer_raw_names: Vec<CString> = validation
			.required_validation_layers.iter()
			.map(|layer_name| CString::new(*layer_name).unwrap())
			.collect();

		let enable_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
			.iter()
			.map(|layer_name| layer_name.as_ptr())
			.collect();
		
		let device_create_info = vk::DeviceCreateInfo {
			s_type: vk::StructureType::DEVICE_CREATE_INFO,
			p_next: ptr::null(),
			flags: vk::DeviceCreateFlags::empty(),
			queue_create_info_count: queue_create_infos.len() as u32,
			p_queue_create_infos: queue_create_infos.as_ptr(),
			enabled_layer_count: if validation.is_enable {
				enable_layer_names.len()
			} else {
				0
			} as u32,
			pp_enabled_layer_names: if validation.is_enable {
				enable_layer_names.as_ptr()
			} else {
				ptr::null()
			},
			enabled_extension_count: 0,
			pp_enabled_extension_names: ptr::null(),
			p_enabled_features: &physical_device_features,
		};

		let logical_device: ash::Device = unsafe {
			instance
				.create_device(physical_device, &device_create_info, None)
				.expect("Failed to create Logical Device!")
		};

		(logical_device, indices)
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
