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
    instance: Instance,
}

impl HelloTriangleApplication {
    pub fn new(window: &Window) -> HelloTriangleApplication {
        HelloTriangleApplication {
            instance: unsafe { Self::init_vulkan(window) },
        }
    }

    pub unsafe fn init_vulkan(window: &Window) -> Instance {
        let app_name = CString::new("Hello Triangle").unwrap();
        let entry = Entry::new().unwrap();
        let extension_names = get_required_extensions(&window, &entry);
        let layer_names = get_layer_names(&entry);

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .api_version(vk::make_version(1, 0, 0));
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        entry.create_instance(&create_info, None).unwrap()
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
            .with_title("Hello, Triangle")
            .with_inner_size(LogicalSize::new(600, 600))
            .build(event_loop)
            .unwrap()
    }

    fn main_loop(self, event_loop: EventLoop<()>, window: Window) -> () {
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                _ => (),
            }
        });
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = HelloTriangleApplication::init_window(&event_loop);
    let app = HelloTriangleApplication::new(&window);
    app.main_loop(event_loop, window);
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
