use crate::utility::debug::ValidationInfo;
use crate::utility::structures::DeviceExtension;

use ash::vk::make_api_version;

use std::os::raw::c_char;

pub const APPLICATION_VERSION: u32 = make_api_version(0, 1, 0, 0);
pub const ENGINE_VERSION: u32 = make_api_version(0, 1, 0, 0);
pub const API_VERSION: u32 = make_api_version(0, 1, 0, 92);

pub const WINDOW_TITLE: &'static str = "Window";
pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;

pub const VALIDATION: ValidationInfo = ValidationInfo {
	is_enable: true,
	required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

pub const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension{
	names: ["VK_KHR_swapchain"],
};


