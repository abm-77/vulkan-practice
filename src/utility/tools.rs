use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;

// Converts from raw vk string to rust string
pub fn vk_to_string(raw_string_array: &[c_char]) -> String {
	let raw_string = unsafe {
		let pointer = raw_string_array.as_ptr();
		CStr::from_ptr(pointer)
	};

	raw_string
		.to_str()
		.expect("Failed to conver vulkan raw string.")
		.to_owned()
}

// reads shader code as bytes from file
pub fn read_shader_code(shader_path: &Path) -> Vec<u8> {
	use std::fs::File;
	use std::io::Read;
	
	let spv_file = File::open(shader_path).expect(&format!("Failed to find spv file at {:?}", shader_path));
	let byte_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

	byte_code
}