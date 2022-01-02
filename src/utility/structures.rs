use ash::vk;

use cgmath::{
	Matrix4,
};

pub struct DeviceExtension {
	pub names: [&'static str; 1],
}

pub struct SurfaceInfo {
	pub surface_loader: ash::extensions::khr::Surface,
	pub surface: vk::SurfaceKHR,
	pub screen_width: u32,
	pub screen_height: u32,
}

pub struct SwapChainInfo {
	pub swapchain_loader: ash::extensions::khr::Swapchain,
	pub swapchain: vk::SwapchainKHR,
	pub swapchain_images: Vec<vk::Image>,
	pub swapchain_format: vk::Format,
	pub swapchain_extent: vk::Extent2D,
}

pub struct SwapChainSupportDetail {
	pub capabilities: vk::SurfaceCapabilitiesKHR,
	pub formats: Vec<vk::SurfaceFormatKHR>,
	pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct QueueFamilyIndices {
	pub graphics_family: Option<u32>,
	pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
	pub fn new() -> QueueFamilyIndices {
		QueueFamilyIndices {
			graphics_family: None,
			present_family: None,
		}
	}
	pub fn is_complete(&self) -> bool {
		self.graphics_family.is_some() &&
		self.present_family.is_some()
	}
}

pub struct SyncObjects {
	pub image_available_semaphores: Vec<vk::Semaphore>,
	pub render_finished_semaphores: Vec<vk::Semaphore>,
	pub in_flight_fences: Vec<vk::Fence>,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Vertex {
	pub pos: [f32; 3],
	pub color: [f32; 4],
	pub tex_coord: [f32; 2],
}

impl Vertex {
	pub fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
		[vk::VertexInputBindingDescription {
			binding: 0,
			stride: std::mem::size_of::<Self>() as u32,
			input_rate: vk::VertexInputRate::VERTEX,
		}]
	}

	pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
		[
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 0,
				format: vk::Format::R32G32B32_SFLOAT,
				offset: 0,
			},
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 1,
				format: vk::Format::R32G32B32A32_SFLOAT,
				offset: std::mem::size_of::<[f32; 3]>() as u32,
			},
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 2,
				format: vk::Format::R32G32_SFLOAT,
				offset: std::mem::size_of::<[f32; 7]>() as u32,
			},
		]
	}
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct UniformBufferObject {
	pub model: Matrix4<f32>,
	pub view: Matrix4<f32>,
	pub proj: Matrix4<f32>,
}

pub const RECT_VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.75, -0.75, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        tex_coord: [1.0, 0.0],
    },
    Vertex {
        pos: [0.75, -0.75, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        tex_coord: [0.0, 0.0],
    },
    Vertex {
        pos: [0.75, 0.75, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        tex_coord: [0.0, 1.0],
    },
    Vertex {
        pos: [-0.75, 0.75, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        tex_coord: [1.0, 1.0],
    },
];
pub const RECT_INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];