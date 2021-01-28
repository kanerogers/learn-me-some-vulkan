use ash::{
    version::{EntryV1_0, InstanceV1_0},
    vk, Entry, Instance,
};
use std::ffi::{CStr, CString};
use std::ptr;
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
        let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();

        let entry = Entry::new().unwrap();
        let supported_extensions = entry
            .enumerate_instance_extension_properties()
            .expect("Unable to enumerate instance extension properties")
            .iter()
            .map(|e| CStr::from_ptr(e.extension_name.as_ptr()))
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

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .api_version(vk::make_version(1, 0, 0));
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names_raw);

        entry.create_instance(&create_info, None).unwrap()
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
            .with_title("Hello, Triangle")
            .with_inner_size(LogicalSize::new(600, 600))
            .build(event_loop)
            .unwrap()
    }

    fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) -> () {
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
    let mut app = HelloTriangleApplication::new(&window);
    app.main_loop(event_loop, window);
}
