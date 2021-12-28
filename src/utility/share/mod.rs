pub mod v1;

use ash::{Entry, Instance, Device, vk};
use winapi::um::cfgmgr32::CM_DELETE_CLASS_ONLY;

use std::{
	ffi::CString, 
	os::raw::{c_char, c_void}, 
	path::Path, 
	ptr
};

use crate::utility::{
	constants::*, 
	debug, 
	platforms,
	structures::*,
	tools,
};

pub fn create_instance (
	entry: &Entry,
	window_title: &str,
	is_enable_debug: bool,
	required_validation_layers: &Vec<&str>,
) -> ash::Instance {
	if is_enable_debug && debug::check_validation_layer_support(entry, required_validation_layers) == false {
		panic!("Validation layers requested, but not available!");	
	}

	let app_name = CString::new(window_title).unwrap();
	let engine_name = CString::new("Vulkan Engine").unwrap();
	let app_info = vk::ApplicationInfo {
		s_type: vk::StructureType::APPLICATION_INFO,
		p_next: ptr::null(),
		p_application_name: app_name.as_ptr(),
		application_version: APPLICATION_VERSION,
		p_engine_name: engine_name.as_ptr(),
		engine_version: ENGINE_VERSION,
		api_version: API_VERSION,
	};

	let debug_utils_create_info = debug::populate_debug_messenger_create_info();

	let extension_names = platforms::required_extension_names();

	let required_validation_layer_raw_names: Vec<CString> = required_validation_layers
		.iter()
		.map(|layer_name| CString::new(*layer_name).unwrap())
		.collect();
	
	let layer_names: Vec<*const i8> = required_validation_layer_raw_names
		.iter()
		.map(|layer_name| layer_name.as_ptr())
		.collect();

	let create_info  = vk::InstanceCreateInfo {
		s_type: vk::StructureType::INSTANCE_CREATE_INFO,
		p_next: if VALIDATION.is_enable {
			&debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
		} else {
			ptr::null()
		},
		flags: vk::InstanceCreateFlags::empty(),
		p_application_info: &app_info,
		pp_enabled_layer_names: if is_enable_debug {
			layer_names.as_ptr()
		} else {
			ptr::null()
		},
		enabled_layer_count: if is_enable_debug {
			layer_names.len()
		} else {
			0
		} as u32,
		pp_enabled_extension_names: extension_names.as_ptr(),
		enabled_extension_count: extension_names.len() as u32,
	};

	let instance: ash::Instance = unsafe {
		entry
			.create_instance(&create_info, None)
			.expect("Failed to create instance!")
	};

	instance
}

pub fn create_surface (
	entry: &ash::Entry,
	instance: &ash::Instance,
	window: &winit::window::Window,
	surface_width: u32,
	surface_height: u32,
) -> SurfaceInfo {
	let surface = unsafe {
		platforms::create_surface(entry, instance, window)
			.expect("Failed to create window")
	};

	let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

	SurfaceInfo {
		surface_loader,
		surface,
		screen_width: surface_width,
		screen_height: surface_height,
	}
}

pub fn pick_physical_device(
	instance: &ash::Instance, 
	surface_info: &SurfaceInfo,
	required_device_extensions: &DeviceExtension,
) -> vk::PhysicalDevice {
	let physical_devices = unsafe {
		instance
			.enumerate_physical_devices()
			.expect("Failed to enumerate Physical Devices!")
	};

	let result = physical_devices.iter().find(|physical_device| {
		is_physical_device_suitable(instance, **physical_device, surface_info, required_device_extensions)
	});

	match result {
		Some(p_physical_device) => *p_physical_device,
		None => panic!("Failed to find a suitable GPU!"),
	}
}

pub fn is_physical_device_suitable (
	instance: &ash::Instance, 
	physical_device: vk::PhysicalDevice,
	surface_info: &SurfaceInfo,
	required_device_extensions: &DeviceExtension,
) -> bool {
	let _device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
	let _device_features = unsafe { instance.get_physical_device_features(physical_device) };
	let indices = find_queue_family(instance, physical_device, surface_info);

	let is_queue_family_supported= indices.is_complete();
	let is_device_extension_supported = 
		check_device_extension_support(instance, physical_device, required_device_extensions);
	let is_swapchain_supported= if is_device_extension_supported {
		let swapchain_support = query_swapchain_support(physical_device, surface_info);
		!swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
	} else {
		false
	};

	return 	is_queue_family_supported &&
			is_device_extension_supported &&
			is_swapchain_supported;

}

pub fn find_queue_family(
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

pub fn check_device_extension_support(
	instance: &ash::Instance,
	physical_device: vk::PhysicalDevice,
	device_extensions: &DeviceExtension,
) -> bool {
	let available_extensions = unsafe {
		instance
			.enumerate_device_extension_properties(physical_device)
			.expect("Failed to get device exetension properties.")
	};

	let mut available_extension_names = vec![];

	println!("\tAvailable Device Extensions: ");
	for extension in available_extensions.iter() {
		let extension_name = tools::vk_to_string(&extension.extension_name);
		println!("\t\tName: {}, Version: {}", extension_name, extension.spec_version);
		available_extension_names.push(extension_name);
	}

	let mut required_extensions = std::collections::HashSet::new();
	for extension in device_extensions.names.iter() {
		required_extensions.insert(extension.to_string());
	}

	for extension_name in available_extension_names.iter() {
		required_extensions.remove(extension_name);
	}

	required_extensions.is_empty()
}

pub fn query_swapchain_support(
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

pub fn create_swapchain(
	instance: &ash::Instance,
	device: &ash::Device,
	physical_device: vk::PhysicalDevice,
	window: &winit::window::Window,
	surface_info: &SurfaceInfo,
	queue_family: &QueueFamilyIndices,
) -> SwapChainInfo {
	let swapchain_support = query_swapchain_support(physical_device, surface_info);

	let surface_format = choose_swapchain_format(&swapchain_support.formats);
	let present_mode = choose_swapchain_present_mode(&swapchain_support.present_modes);
	let extent = choose_swapchain_extent(&swapchain_support.capabilities, window);

	let image_count = swapchain_support.capabilities.min_image_count + 1;
	let image_count = if swapchain_support.capabilities.max_image_count > 0 {
		image_count.min(swapchain_support.capabilities.max_image_count)
	} else{
		image_count
	};

	let (image_sharing_mode, queue_family_index_count, queue_familiy_indices) =
		if queue_family.graphics_family != queue_family.present_family {
			(
				vk::SharingMode::CONCURRENT,
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

pub fn choose_swapchain_format(
	available_formats: &Vec<vk::SurfaceFormatKHR>
) -> vk::SurfaceFormatKHR {
	// check if list contains most widely used R8G8B8A8 format with nonlinear color space
	for available_format in available_formats {
		if available_format.format == vk::Format::B8G8R8A8_SRGB
			&& available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
		{
			return available_format.clone();
		}
	}

	// return the first format from the list
	return available_formats.first().unwrap().clone();
}

pub fn choose_swapchain_present_mode(
	available_present_modes: &Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
	for &available_present_mode in available_present_modes.iter() {
		if available_present_mode == vk::PresentModeKHR:: MAILBOX {
			return available_present_mode;
		}
	}
	vk::PresentModeKHR::FIFO
}

pub fn choose_swapchain_extent(
	capabilities: &vk::SurfaceCapabilitiesKHR,
	window: &winit::window::Window,
) -> vk::Extent2D {
	if capabilities.current_extent.width != u32::MAX {
		capabilities.current_extent
	} else {
		use tools::clamp;
		
		let window_size = window.inner_size();

		vk::Extent2D {
			width: clamp(
				window_size.width,
				capabilities.min_image_extent.width,
				capabilities.max_image_extent.width,
			),
			height: clamp(
				window_size.height,
				capabilities.min_image_extent.height,
				capabilities.max_image_extent.height,
			),
		}
	}
}

pub fn create_logical_device(
	instance: &ash::Instance,
	physical_device: vk::PhysicalDevice,
	validation: &debug::ValidationInfo,
	device_extensions: &DeviceExtension,
	surface_info: &SurfaceInfo,
) -> (ash::Device, QueueFamilyIndices) {
	let indices = find_queue_family(instance, physical_device, surface_info);

	let mut unique_queue_families = std::collections::HashSet::new();
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
	
	let enable_extension_names = device_extensions.get_extensions_raw_names();

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
		enabled_extension_count: enable_extension_names.len() as u32,
		pp_enabled_extension_names: enable_extension_names.as_ptr(),
		p_enabled_features: &physical_device_features,
	};

	let logical_device: ash::Device = unsafe {
		instance
			.create_device(physical_device, &device_create_info, None)
			.expect("Failed to create Logical Device!")
	};

	(logical_device, indices)
}
pub fn create_image_views(
	device: &ash::Device,
	surface_format: vk::Format,
	images: &Vec<vk::Image>
) -> Vec<vk::ImageView> {
	let mut swapchain_imageviews = vec![];

	for &image in images.iter() {
		let imageview_create_info = vk::ImageViewCreateInfo {
			s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
			p_next: ptr::null(),
			flags: vk::ImageViewCreateFlags::empty(),
			view_type: vk::ImageViewType::TYPE_2D,
			format: surface_format,
			components: vk::ComponentMapping {
				r: vk::ComponentSwizzle::IDENTITY,
				g: vk::ComponentSwizzle::IDENTITY,
				b: vk::ComponentSwizzle::IDENTITY,
				a: vk::ComponentSwizzle::IDENTITY,
			},
			subresource_range: vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			},
			image: image,
		};

		let imageview = unsafe {
			device
				.create_image_view(&imageview_create_info, None)
				.expect("Failed to create Image View!")
		};
		swapchain_imageviews.push(imageview);
	}
	swapchain_imageviews
}

pub fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> vk::ShaderModule {
	let shader_module_create_info = vk::ShaderModuleCreateInfo {
		s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::ShaderModuleCreateFlags::empty(),
		code_size: code.len(),
		p_code: code.as_ptr() as *const u32,
	};

	unsafe {
		device
			.create_shader_module(&shader_module_create_info, None)
			.expect("Failed to create Shader Module!")
	}
}

pub fn find_memory_type(
	type_filter: u32,
	required_properties: vk::MemoryPropertyFlags,
	mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> u32 {
	for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
		if (type_filter & (1 << i)) > 0 &&
			memory_type.property_flags.contains(required_properties){
			return i as u32;
		}
	}

	panic!("Failed to find suitable memory type!")
}

pub fn create_buffer(
	device: &ash::Device,
	size: vk::DeviceSize,
	usage: vk::BufferUsageFlags,
	required_memory_properties: vk::MemoryPropertyFlags,
	device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> (vk::Buffer, vk::DeviceMemory) {
	let buffer_create_info = vk::BufferCreateInfo {
		s_type: vk::StructureType::BUFFER_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::BufferCreateFlags::empty(),
		size: size,
		usage: usage,
		sharing_mode: vk::SharingMode::EXCLUSIVE,
		queue_family_index_count: 0,
		p_queue_family_indices: ptr::null(),

	};

	let buffer = unsafe {
		device
			.create_buffer(&buffer_create_info, None)
			.expect("Could not create Vertex Buffer!")
	};

	let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
	let memory_type = find_memory_type(
		mem_requirements.memory_type_bits,
		required_memory_properties,
		device_memory_properties,
	);

	let allocate_info = vk::MemoryAllocateInfo {
		s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
		p_next: ptr::null(),
		allocation_size: mem_requirements.size,
		memory_type_index: memory_type,
	};

	let buffer_memory = unsafe {
		device
			.allocate_memory(&allocate_info, None)
			.expect("Failed to allocate Vertex Buffer memory!")
	};

	unsafe {
		device
			.bind_buffer_memory(buffer, buffer_memory, 0)
			.expect("Failed to bind buffer!");
	}

	(buffer, buffer_memory)
}

fn copy_buffer(
	device: &ash::Device,
	submit_queue: vk::Queue,
	command_pool: vk::CommandPool,
	src_buffer: vk::Buffer,
	dst_buffer: vk::Buffer,
	size: vk::DeviceSize,

) {
	let command_buffer = begin_single_time_command(device, command_pool);
	let copy_regions = [vk::BufferCopy {
		src_offset: 0,
		dst_offset: 0,
		size,
	}];

	unsafe {
		device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);
	}

	end_single_time_command(device, command_pool, submit_queue, command_buffer);

}

pub fn begin_single_time_command(
	device: &ash::Device,
	command_pool: vk::CommandPool,
) -> vk::CommandBuffer {
	let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
		s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
		p_next: ptr::null(),
		command_buffer_count: 1,
		command_pool,
		level: vk::CommandBufferLevel::PRIMARY,
	};

	let command_buffer = unsafe {
		device
			.allocate_command_buffers(&command_buffer_allocate_info)
			.expect("Failed to allocate Command Bufer!")
	}[0];

	let begin_info = vk::CommandBufferBeginInfo {
		s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
		p_next: ptr::null(),
		flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
		p_inheritance_info: ptr::null(),
	};

	unsafe {
		device
			.begin_command_buffer(command_buffer, &begin_info)
			.expect("Failed to begin Command Buffer!");
	}

	command_buffer
}

pub fn end_single_time_command(
	device: &ash::Device,
	command_pool: vk::CommandPool,
	submit_queue: vk::Queue,
	command_buffer: vk::CommandBuffer,
) {

	unsafe {
		device
			.end_command_buffer(command_buffer)
			.expect("Could not end Command Buffer!");
	}

	let buffers_to_submit = [command_buffer];

	let submit_info = [vk::SubmitInfo {
		s_type: vk::StructureType::SUBMIT_INFO,
		p_next: ptr::null(),
		wait_semaphore_count: 0,
		p_wait_semaphores: ptr::null(),
		p_wait_dst_stage_mask: ptr::null(),
		command_buffer_count: 1,
		p_command_buffers: &command_buffer,
		signal_semaphore_count: 0,
		p_signal_semaphores: ptr::null(),
	}];


	unsafe {
		device
			.queue_submit(submit_queue, &submit_info, vk::Fence::null())
			.expect("Failed to Submit Queue.");
		device
			.queue_wait_idle(submit_queue)
			.expect("Failed to wait Queue idle");

		device.free_command_buffers(command_pool, &buffers_to_submit);
	}
}