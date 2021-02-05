#![allow(dead_code)]
#![allow(unused_variables)]

use ash::{
    extensions::khr,
    extensions::ext,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use std::{
    ffi:: { CStr, CString},
    cmp
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use byte_slice_cast::AsSliceOf;

#[derive(Clone, Debug)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn find_queue_families(
        instance: &Instance,
        device: vk::PhysicalDevice,
        entry: &Entry,
        surface_khr: vk::SurfaceKHR,
    ) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        };
        let surface = khr::Surface::new(entry, instance);

        for (i, queue) in unsafe { instance
            .get_physical_device_queue_family_properties(device) }
            .iter()
            .enumerate()
        {
            let i = i as u32;
            if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }
            if unsafe { surface
                .get_physical_device_surface_support(device, i, surface_khr) }
                .unwrap()
            {
                indices.present_family = Some(i);
            }
            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }

    fn are_same(&self) -> bool {
        self.is_complete() && self.graphics_family == self.present_family
    }
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>
}

impl SwapChainSupportDetails {
    fn query_swap_chain_support(instance: &Instance, entry: &Entry, device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> SwapChainSupportDetails {
        let surface_ext = khr::Surface::new(entry, instance);
        let capabilities = unsafe { surface_ext.get_physical_device_surface_capabilities(device, surface).expect("unable to get capabilities") };
        let surface_formats = unsafe { surface_ext.get_physical_device_surface_formats(device, surface).expect("unable to get surface formats") };
        let present_modes = unsafe { surface_ext.get_physical_device_surface_present_modes(device, surface).expect("unable to get present modes") };

        SwapChainSupportDetails {
            capabilities,
            surface_formats,
            present_modes
        }
    }
}

struct HelloTriangleApplication {
    _entry: Entry,
    instance: Instance,
    debug_utils: Option<ext::DebugUtils>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    _physical_device: vk::PhysicalDevice,
    device: Device,
    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
    swap_chain: vk::SwapchainKHR,
    _swap_chain_images: Vec<vk::Image>,
    _swap_chain_image_format: vk::Format,
    _swap_chain_extent: vk::Extent2D,
    swap_chain_image_views: Vec<vk::ImageView>,
}

impl HelloTriangleApplication {
    pub fn new(window: &Window) -> HelloTriangleApplication {
        let (instance, entry, debug_utils, debug_messenger) = unsafe { Self::init_vulkan(window) };
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };
        let (physical_device, indices) = pick_physical_device(&instance, &entry, surface);
        let (device, graphics_queue, present_queue) =
            unsafe { create_logical_device(&instance, physical_device, indices) };
        let (swap_chain, format, extent) = create_swap_chain(&instance, &entry, physical_device, surface, window, &device);
        let mut swap_chain_images = get_swap_chain_images(&instance, &device, swap_chain);
        let swap_chain_image_views = create_image_views(&mut swap_chain_images, format, &device);

        HelloTriangleApplication {
            instance,
            _entry: entry,
            debug_utils,
            debug_messenger,
            _physical_device: physical_device,
            device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            swap_chain,
            _swap_chain_images: swap_chain_images,
            _swap_chain_image_format: format,
            _swap_chain_extent: extent,
            swap_chain_image_views
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

    pub fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
            .with_title("Hello, Triangle")
            .with_inner_size(LogicalSize::new(600, 600))
            .build(event_loop)
            .unwrap()
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            for view in self.swap_chain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            // self.swap_chain_image_views will now be empty

            let swapchain = khr::Swapchain::new(&self.instance, &self.device);
            swapchain.destroy_swapchain(self.swap_chain, None);
            // WARNING: self.swap_chain is now invalid!

            self.debug_messenger.map(|m| {
                self.debug_utils.as_ref().map(|d| {
                    d.destroy_debug_utils_messenger(m, None);
                })
            });
            // WARNING: self.debug_messenger is now invalid!

            self.device.destroy_device(None);
            // WARNING: self.device is now invalid!

            self.instance.destroy_instance(None);
            // WARNING: self.instance is now invalid!
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = HelloTriangleApplication::init_window(&event_loop);
    let app = HelloTriangleApplication::new(&window);
    main_loop(event_loop, window, app);
}

fn main_loop(event_loop: EventLoop<()>, window: Window, mut _app: HelloTriangleApplication) -> () {
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => {
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        }
    });
}

// Graphics Pipeline
fn create_graphics_pipeline(device: &Device) { 
    let vert_shader_code = include_bytes!("./shaders/shader.vert.spv");
    let frag_shader_code = include_bytes!("./shaders/shader.frag.spv");

    let vertex_shader_module = create_shader_module(device, vert_shader_code);
    let frag_shader_module = create_shader_module(device, frag_shader_code);
    let name = CString::new("main").unwrap();

    let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(name.as_c_str());

    let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(name.as_c_str());
    
    let shader_stages = [vertex_shader_stage_info, frag_shader_stage_info]; 


    // Cleanup
    unsafe { device.destroy_shader_module(vertex_shader_module, None) } ;
    unsafe { device.destroy_shader_module(frag_shader_module, None) } ;
}

fn create_shader_module(device: &Device, bytes: &[u8]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::builder()
        .code(bytes.as_slice_of::<u32>().unwrap());

    unsafe { device.create_shader_module(&create_info, None).expect("Unable to create shader module") }
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
    let required_extensions_raw = required_extensions.iter().map(|e| e.as_ptr()).collect::<Vec<_>>();
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

// Swap Chain
fn create_swap_chain (instance: &Instance, entry: &Entry, physical_device: vk::PhysicalDevice, surface: vk::SurfaceKHR, window: &Window, logical_device: &Device) -> (vk::SwapchainKHR, vk::Format, vk::Extent2D) {
    let swap_chain_support = SwapChainSupportDetails::query_swap_chain_support(instance, entry, physical_device, surface);

    let surface_format = choose_swap_surface_format(swap_chain_support.surface_formats);
    let present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
    let extent = choose_swap_extent(swap_chain_support.capabilities, window);

    let image_count = swap_chain_support.capabilities.min_image_count + 1;
    let image_count = if swap_chain_support.capabilities.max_image_count > 0 && image_count > swap_chain_support.capabilities.max_image_count {
        swap_chain_support.capabilities.max_image_count
    } else {
        image_count
    };

    let indices = QueueFamilyIndices::find_queue_families(instance, physical_device, entry, surface);
    let indices_are_same = indices.are_same();
    let image_sharing_mode = if indices_are_same { vk::SharingMode::EXCLUSIVE } else { vk::SharingMode::CONCURRENT };
    let queue_family_indices = if indices_are_same { Vec::new() } else { vec![indices.graphics_family.unwrap(), indices.present_family.unwrap()] };

    let create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(swap_chain_support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null())
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

    let swapchain = khr::Swapchain::new(instance, logical_device);
    let swapchain = unsafe { swapchain.create_swapchain(&create_info, None) }.expect("Unable to create Swapchain");
    (swapchain, surface_format.format, extent)
}

fn choose_swap_surface_format(formats: Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    for available_format in &formats { 
        if available_format.format == vk::Format::B8G8R8A8_SRGB && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            return *available_format
        }
    }

    return *formats.first().unwrap();
}

fn choose_swap_present_mode(available_present_modes: Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
    for available_present_mode in &available_present_modes {
        if available_present_mode == &vk::PresentModeKHR::MAILBOX {
            return *available_present_mode
        }
    }
    return *available_present_modes.first().unwrap();
}

fn choose_swap_extent(capabilities: vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX { return capabilities.current_extent }
    let physical_size = window.inner_size();
    let width = cmp::max(capabilities.min_image_extent.width, cmp::min(capabilities.max_image_extent.width, physical_size.width));
    let height = cmp::max(capabilities.max_image_extent.height, cmp::min(capabilities.min_image_extent.height, physical_size.height));

    vk::Extent2D {
        width,
        height
    }
}

fn get_swap_chain_images(instance: &Instance, device: &Device, swap_chain: vk::SwapchainKHR) -> Vec<vk::Image> {
    let swap_chain_ext = khr::Swapchain::new(instance, device);
    unsafe { swap_chain_ext.get_swapchain_images(swap_chain).expect("Unable to get swapchain images") }
}

fn create_image_views(swap_chain_images: &mut Vec<vk::Image>, format: vk::Format, device: &Device) -> Vec<vk::ImageView> {
    let image = swap_chain_images.get(0).unwrap().clone();
    let components = vk::ComponentMapping::builder()
        .r(vk::ComponentSwizzle::ONE)
        .g(vk::ComponentSwizzle::ONE)
        .b(vk::ComponentSwizzle::ONE)
        .a(vk::ComponentSwizzle::ONE)
        .build();

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(components)
        .subresource_range(subresource_range);

    let image_view = unsafe { device.create_image_view(&create_info, None).expect("Unable to get image view") };

    vec![image_view]
}


// Physical Device
fn pick_physical_device(
    instance: &Instance,
    entry: &Entry,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, QueueFamilyIndices) {
    unsafe {
        let devices = instance.enumerate_physical_devices().unwrap();
        let mut devices = devices.into_iter().map(|d| {
            get_suitability(d, instance, entry, surface)
        }).collect::<Vec<_>>();
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

    if suitable { suitability -= 1 }

    (suitability, indices, device)
}

fn check_swap_chain_adequate(instance: &Instance, entry: &Entry, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> bool {
    let swap_chain_support_details = SwapChainSupportDetails::query_swap_chain_support(instance, entry, device, surface);
    !swap_chain_support_details.surface_formats.is_empty() && !swap_chain_support_details.present_modes.is_empty()
}

fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let extensions = unsafe { instance.enumerate_device_extension_properties(device).expect("Unable to get extension properties") }
        .iter()
        .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
        .collect::<Vec<_>>();

        let required_extension = khr::Swapchain::name();

        extensions.contains(&required_extension)
}

// TODO: Portability?
// fn has_portability(instance: &Instance, device: vk::PhysicalDevice) -> bool {
//     let extensions = unsafe { instance.enumerate_device_extension_properties(device).expect("Unable to get extension properties") }
//         .iter()
//         .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
//         .collect::<Vec<_>>();

//         let portability_extension = CString::new("VK_KHR_portability_subset").unwrap();
//         extensions.contains(&portability_extension.as_c_str())
// } 

// fn portability_extensions() -> Vec<CString> {
//     vec![
//         CString::new("VK_KHR_portability_subset").unwrap(),
//         CString::new("VK_KHR_get_physical_device_properties2").unwrap()
//     ]
// }

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

#[cfg(debug_assertions)]
fn should_add_validation_layers() -> bool {
    true
}

#[cfg(not(debug_assertions))]
fn should_add_validation_layers() -> bool {
    false
}