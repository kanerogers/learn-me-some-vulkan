use std::ffi::{CStr, CString};

use ash::{
    extensions::{ext, khr},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::{self},
    Device, Entry, Instance,
};
use winit::window::Window;

use crate::{QueueFamilyIndices, swap_chain::SwapChainSupportDetails};

pub struct VulkanContext {
    pub device: Device,
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
    pub surface: vk::SurfaceKHR,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub debug_utils: Option<ext::DebugUtils>,
    pub surface_loader: khr::Surface,
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            // WARNING: self.command_pool is now invalid!

            self.debug_messenger.map(|m| {
                self.debug_utils.as_ref().map(|d| {
                    d.destroy_debug_utils_messenger(m, None);
                })
            });

            self.surface_loader.destroy_surface(self.surface, None);
            // WARNING: self.surface is now invalid!

            self.device.destroy_device(None);
            // WARNING: self.device is now invalid!

            self.instance.destroy_instance(None);
            // WARNING: self.instance is now invalid!
        }
    }
}

impl VulkanContext {
    pub fn new(window: &Window) -> Self {
        let (instance, entry, debug_utils, debug_messenger) = unsafe { Self::init_vulkan(&window) };
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };
        let (physical_device, indices) = pick_physical_device(&instance, &entry, surface);
        let (device, graphics_queue, present_queue) =
            unsafe { create_logical_device(&instance, physical_device, indices.clone()) };
        let command_pool = create_command_pool(indices.clone(), &device);
        let surface_loader = khr::Surface::new(&entry, &instance);

        Self {
            device,
            entry,
            instance,
            physical_device,
            command_pool,
            surface,
            graphics_queue,
            present_queue,
            debug_utils,
            debug_messenger,
            surface_loader,
        }
    }

    pub unsafe fn init_vulkan(
        window: &Window,
    ) -> (
        Instance,
        Entry,
        Option<ext::DebugUtils>,
        Option<vk::DebugUtilsMessengerEXT>,
    ) {
        let app_name = CString::new("Hello Triangle").unwrap();
        let entry = Entry::new().unwrap();
        let extension_names = get_required_extensions_for_window(&window, &entry);
        let layer_names = get_layer_names(&entry);

        let mut debug_messenger_info = get_debug_messenger_create_info();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .api_version(vk::make_version(1, 0, 0));
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names)
            .push_next(&mut debug_messenger_info);

        let instance = entry.create_instance(&create_info, None).unwrap();

        let (debug_utils, messenger) =
            setup_debug_messenger(&entry, &instance, &debug_messenger_info);

        (instance, entry, debug_utils, messenger)
    }
}

// Surface
fn get_required_extensions_for_window(window: &Window, entry: &Entry) -> Vec<*const i8> {
    let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();

    let supported_extensions = entry
        .enumerate_instance_extension_properties()
        .expect("Unable to enumerate instance extension properties")
        .iter()
        .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
        .collect::<Vec<_>>();

    let mut extension_names_raw = Vec::new();
    for extension in surface_extensions {
        assert!(
            supported_extensions.contains(&extension),
            "Unsupported extension: {:?}",
            extension
        );
        extension_names_raw.push(extension.as_ptr())
    }

    if should_add_validation_layers() {
        let debug_utils = ext::DebugUtils::name();
        extension_names_raw.push(debug_utils.as_ptr())
    }

    return extension_names_raw;
}

// Physical Device
fn pick_physical_device(
    instance: &Instance,
    entry: &Entry,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, QueueFamilyIndices) {
    unsafe {
        let devices = instance.enumerate_physical_devices().unwrap();
        let mut devices = devices
            .into_iter()
            .map(|d| get_suitability(d, instance, entry, surface))
            .collect::<Vec<_>>();
        devices.sort_by_key(|i| i.0);

        let (_, indices, device) = devices.remove(0);
        (device, indices)
    }
}

/// Gets a device's suitability. Lower score is bettter.
unsafe fn get_suitability(
    device: vk::PhysicalDevice,
    instance: &Instance,
    entry: &Entry,
    surface: vk::SurfaceKHR,
) -> (i8, QueueFamilyIndices, vk::PhysicalDevice) {
    let properties = instance.get_physical_device_properties(device);
    let indices = QueueFamilyIndices::find_queue_families(instance, device, entry, surface);
    let has_extension_support = check_device_extension_support(instance, device);
    let swap_chain_adequate = check_swap_chain_adequate(instance, entry, surface, device);
    let has_graphics_family = indices.graphics_family.is_some();

    let mut suitability = 0;
    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        suitability -= 5;
    }

    let suitable = has_extension_support && swap_chain_adequate && has_graphics_family;

    if suitable {
        suitability -= 1
    }

    (suitability, indices, device)
}

fn check_swap_chain_adequate(
    instance: &Instance,
    entry: &Entry,
    surface: vk::SurfaceKHR,
    device: vk::PhysicalDevice,
) -> bool {
    let swap_chain_support_details =
        SwapChainSupportDetails::query_swap_chain_support(entry, instance, device, surface);
    !swap_chain_support_details.surface_formats.is_empty()
        && !swap_chain_support_details.present_modes.is_empty()
}

fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(device)
            .expect("Unable to get extension properties")
    }
    .iter()
    .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
    .collect::<Vec<_>>();

    let required_extension = khr::Swapchain::name();

    extensions.contains(&required_extension)
}


#[cfg(debug_assertions)]
fn should_add_validation_layers() -> bool {
    true
}

#[cfg(not(debug_assertions))]
fn should_add_validation_layers() -> bool {
    false
}

// Debug Messenger

fn get_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    let message_severity = 
    // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(message_severity)
        .message_type(message_type)
        .pfn_user_callback(Some(debug_messenger_callback))
}

#[cfg(debug_assertions)]
fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
    info: &vk::DebugUtilsMessengerCreateInfoEXT,
) -> (Option<ext::DebugUtils>, Option<vk::DebugUtilsMessengerEXT>) {
    let debug_utils = ext::DebugUtils::new(entry, instance);

    let messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(info, None)
            .unwrap()
    };

    (Some(debug_utils), Some(messenger))
}

#[cfg(not(debug_assertions))]
fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> (Option<ext::DebugUtils>, Option<vk::DebugUtilsMessengerEXT>) {
    (None, None)
}

fn get_layer_names(entry: &Entry) -> Vec<*const i8> {
    let mut validation_layers_raw = Vec::new();
    if !should_add_validation_layers() {
        return validation_layers_raw;
    };

    let validation_layers = get_validation_layers();
    let supported_layers = entry
        .enumerate_instance_layer_properties()
        .expect("Unable to enumerate instance layer properties")
        .iter()
        .map(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) })
        .collect::<Vec<_>>();

    for layer in validation_layers {
        assert!(
            supported_layers.contains(&layer),
            "Unsupported layer: {:?}",
            layer
        );
        validation_layers_raw.push(layer.as_ptr())
    }

    return validation_layers_raw;
}

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_messenger_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    println!(
        "[VULKAN]: {:?}",
        CStr::from_ptr((*p_callback_data).p_message)
    );
    return vk::FALSE;
}

#[cfg(debug_assertions)]
fn get_validation_layers() -> Vec<&'static CStr> {
    let validation_layer = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
    return vec![validation_layer];
}

#[cfg(not(debug_assertions))]
fn get_validation_layers() -> Vec<&'static CStr> {
    return Vec::new();
}
// Logical Device
unsafe fn create_logical_device<'a>(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    indices: QueueFamilyIndices,
) -> (Device, vk::Queue, vk::Queue) {
    let required_extensions = vec![khr::Swapchain::name()];

    // TODO: Portability
    // let extensions = portability_extensions();
    // if has_portability(instance, physical_device) {
    //     let mut extensions = extensions.iter().map(|i| i.as_c_str()).collect();
    //     required_extensions.append(&mut extensions);
    // }
    let required_extensions_raw = required_extensions
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();
    let queue_priorities = [1.0];
    let graphics_queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_priorities(&queue_priorities)
        .queue_family_index(indices.graphics_family.unwrap())
        .build();

    let mut queue_create_infos = vec![graphics_queue_create_info];

    if !indices.are_same() {
        let present_queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_priorities(&queue_priorities)
            .queue_family_index(indices.present_family.unwrap())
            .build();
        queue_create_infos.push(present_queue_create_info);
    }

    let physical_device_features = vk::PhysicalDeviceFeatures::builder();

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos[..])
        .enabled_extension_names(&required_extensions_raw)
        .enabled_features(&physical_device_features);

    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .unwrap();

    let graphics_queue = device.get_device_queue(indices.graphics_family.unwrap(), 0);
    let present_queue = device.get_device_queue(indices.present_family.unwrap(), 0);

    (device, graphics_queue, present_queue)
}

// Command Buffers/Pools
fn create_command_pool(
    queue_family_indices: QueueFamilyIndices,
    device: &Device,
) -> vk::CommandPool {
    let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.graphics_family.unwrap());

    unsafe {
        device
            .create_command_pool(&pool_info, None)
            .expect("Unable to create command pool")
    }
}