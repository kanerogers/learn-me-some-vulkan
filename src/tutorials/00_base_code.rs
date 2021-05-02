use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct HelloTriangleApplication {}

impl HelloTriangleApplication {
    pub fn new() -> HelloTriangleApplication {
        HelloTriangleApplication {}
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

fn main() {
    let event_loop = EventLoop::new();
    let window = HelloTriangleApplication::init_window(&event_loop);
    let app = HelloTriangleApplication::new();
    app.main_loop(event_loop, window);
}
