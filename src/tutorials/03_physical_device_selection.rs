use ash::{
    extensions::ext,
    version::{EntryV1_0, InstanceV1_0},
    vk, Entry, Instance,
};
use std::ffi::{CStr, CString};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct HelloTriangleApplication {
    _entry: Entry,
    instance: Instance,
    debug_utils: Option<ext::DebugUtils>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    physical_device: vk::PhysicalDevice,
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
}

impl HelloTriangleApplication {
    pub fn new(window: &Window) -> HelloTriangleApplication {
        let (instance, _entry, debug_utils, debug_messenger) = unsafe { Self::init_vulkan(window) };
        println!("Picking a physical device..");
        let physical_device = pick_physical_device(&instance);

        HelloTriangleApplication {
            instance,
            _entry,
            debug_utils,
            debug_messenger,
            physical_device,
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
        let extension_names = get_required_extensions(&window, &entry);
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
            self.debug_messenger.map(|m| {
                self.debug_utils.as_ref().map(|d| {
                    d.destroy_debug_utils_messenger(m, None);
                })
            });
            self.instance.destroy_instance(None);
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

fn get_required_extensions(window: &Window, entry: &Entry) -> Vec<*const i8> {
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

fn get_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    let message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(message_severity)
        .message_type(message_type)
        .pfn_user_callback(Some(debug_messenger_callback))
}

unsafe fn find_queue_families(
    instance: &Instance,
    device: vk::PhysicalDevice,
) -> QueueFamilyIndices {
    let mut graphics_family = None;

    for (i, queue) in instance
        .get_physical_device_queue_family_properties(device)
        .iter()
        .enumerate()
    {
        if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_family = Some(i as u32);
            break;
        }
    }

    QueueFamilyIndices { graphics_family }
}

fn pick_physical_device(instance: &Instance) -> vk::PhysicalDevice {
    unsafe {
        println!("Getting devices..");
        let devices = instance.enumerate_physical_devices().unwrap();
        println!("Devices: {:?}", devices);
        for device in devices {
            println!("Checking if {:?} is suitable..", device);
            if is_device_suitable(device, instance) {
                return device;
            }
        }
        panic!("Failed to find a suitable device");
    }
}

unsafe fn is_device_suitable(device: vk::PhysicalDevice, instance: &Instance) -> bool {
    let properties = instance.get_physical_device_properties(device);
    let features = instance.get_physical_device_features(device);
    let indices = find_queue_families(&instance, device);
    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        && features.geometry_shader == vk::TRUE
        && indices.graphics_family.is_some()
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
    // println!(
    //     "[VULKAN]: {:?}",
    //     CStr::from_ptr((*p_callback_data).p_message)
    // );
    return vk::FALSE;
}
