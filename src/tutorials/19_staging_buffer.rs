#![allow(dead_code)]
#![allow(unused_variables)]

use ash::{
    extensions::khr,
    extensions::ext,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use cgmath::{Vector2, Vector3, vec2, vec3};
use memoffset::offset_of;
use std::{cmp, ffi:: { CStr, CString}, mem::size_of};
use winit::{dpi::{LogicalSize, PhysicalSize}, event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::{Window, WindowBuilder}};
use byte_slice_cast::AsSliceOf;

const MAX_FRAMES_IN_FLIGHT:usize = 2;

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

struct Vertex {
    pos: Vector2<f32>,
    colour: Vector3<f32>,
}

impl Vertex {
    fn new(pos: Vector2<f32>, colour: Vector3<f32>) -> Self { Self { pos, colour } }
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }
    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let position_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32)
            .build();

        let colour_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, colour) as u32)
            .build();

        [position_attribute, colour_attribute]
    }
}

struct VulkanContext {
    device: Device,
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    surface: vk::SurfaceKHR,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_utils: Option<ext::DebugUtils>,
}

impl VulkanContext {
    fn new(window: &Window) -> Self { 
        let (instance, entry, debug_utils, debug_messenger) = unsafe { Self::init_vulkan(&window) };
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };
        let (physical_device, indices) = pick_physical_device(&instance, &entry, surface);
        let (device, graphics_queue, present_queue) =
            unsafe { create_logical_device(&instance, physical_device, indices.clone()) };
        let command_pool = create_command_pool(indices.clone(), &device);

        Self { 
            device, entry, instance, physical_device, command_pool, surface, graphics_queue, present_queue, debug_utils, debug_messenger
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

struct HelloTriangleApplication {
    context: VulkanContext,
    swap_chain_ext: khr::Swapchain,
    swap_chain: vk::SwapchainKHR,
    _swap_chain_images: Vec<vk::Image>,
    _swap_chain_image_format: vk::Format,
    _swap_chain_extent: vk::Extent2D,
    swap_chain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swap_chain_framebuffers: Vec<vk::Framebuffer>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<Option<vk::Fence>>,
    surface_loader: khr::Surface,
    current_frame: usize,
    framebuffer_resized: bool,
    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
}

impl HelloTriangleApplication {
    pub fn new(window: &Window) -> HelloTriangleApplication {
        let context = VulkanContext::new(&window);
        
        let (swap_chain_ext, swap_chain, format, extent) = create_swap_chain(&context, window);
        let mut swap_chain_images = get_swap_chain_images(&context, swap_chain);
        let swap_chain_image_views = create_image_views(&mut swap_chain_images, format, &context);
        let render_pass = create_render_pass(format, &context.device);
        let (pipeline_layout, pipeline) = create_graphics_pipeline(&context.device, extent, render_pass);
        let swap_chain_framebuffers = create_framebuffers(&swap_chain_image_views, &context.device, render_pass, extent);
        let vertices = vec![
            Vertex::new(vec2(-0.5, -0.5), vec3(1.0, 0.0, 1.0)),
            Vertex::new(vec2(0.5, -0.5), vec3(0.0, 1.0, 1.0)),
            Vertex::new(vec2(0.5, 0.5), vec3(1.0, 0.0, 1.0)),
        ];
        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(&context, &vertices);
        let vertex_count = vertices.len() as u32;
        let command_buffers = create_command_buffers(&context, &swap_chain_framebuffers, render_pass, extent, pipeline, vertex_buffer, vertex_count);
        let (image_available, render_finished, in_flight_fences, images_in_flight) = create_sync_objects(&context.device, swap_chain_image_views.len());
        let surface_loader = khr::Surface::new(&context.entry, &context.instance);

        HelloTriangleApplication {
            context,
            swap_chain_ext,
            swap_chain,
            _swap_chain_images: swap_chain_images,
            _swap_chain_image_format: format,
            _swap_chain_extent: extent,
            swap_chain_image_views,
            render_pass,
            pipeline_layout,
            pipeline,
            swap_chain_framebuffers,
            command_buffers,
            image_available_semaphores: image_available,
            render_finished_semaphores: render_finished,
            in_flight_fences,
            images_in_flight,
            surface_loader,
            current_frame: 0,
            framebuffer_resized: false,
            vertices,
            vertex_buffer,
            vertex_buffer_memory,
        }
    }


    pub fn recreate_swap_chain(&mut self, window: &Window) {
        unsafe { self.context.device.device_wait_idle().expect("Could not wait idle") };
        unsafe { self.cleanup_swap_chain() };

        let (swap_chain_ext, swap_chain, format, extent) = create_swap_chain(&self.context, window);
        self.swap_chain = swap_chain;
        let mut swap_chain_images = get_swap_chain_images(&self.context, swap_chain);
        self.swap_chain_image_views = create_image_views(&mut swap_chain_images, format, &self.context);
        self.render_pass = create_render_pass(format, &self.context.device);
        let (pipeline_layout, pipeline) = create_graphics_pipeline(&self.context.device, extent, self.render_pass);
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;
        self.swap_chain_framebuffers = create_framebuffers(&self.swap_chain_image_views, &self.context.device, self.render_pass, extent);
        let vertex_count = self.vertices.len() as u32;
        self.command_buffers = create_command_buffers(&self.context, &self.swap_chain_framebuffers, self.render_pass, extent, pipeline, self.vertex_buffer, vertex_count);
        self.framebuffer_resized = false;
    }


    pub fn draw_frame(&mut self, window: &Window) {
        let device = &self.context.device;
        let instance = &self.context.instance;

        let fence = self.in_flight_fences.get(self.current_frame).expect("Unable to get fence!");
        let fences = [*fence];

        unsafe { device.wait_for_fences(&fences, true, u64::MAX) }.expect("Unable to wait for fence");

        let image_available_semaphore = self.image_available_semaphores.get(self.current_frame).expect("Unable to get image_available semaphore for frame!");
        let render_finished_semaphore = self.render_finished_semaphores.get(self.current_frame).expect("Unable to get render_finished semaphore");

        let image_index = unsafe {
            match self.swap_chain_ext.acquire_next_image(self.swap_chain, u64::MAX, *image_available_semaphore, vk::Fence::null()) {
                Ok((index, _)) => index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swap_chain(&window),
                _ => panic!("Failed to acquire swap chain image!"),
            }
        };

        if let Some(image_in_flight_fence) = unsafe { self.images_in_flight.get_unchecked(image_index as usize) } { 
            let fences = [*image_in_flight_fence];
            unsafe { device.wait_for_fences(&fences, true, u64::MAX) }.expect("Unable to wait for image_in_flight_fence");
        }

        self.images_in_flight[image_index as usize] = Some(*fence);

        let wait_semaphores = [*image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let command_buffer = self.command_buffers.get(image_index as usize).unwrap();
        let command_buffers = [*command_buffer];

        let signal_semaphores = [*render_finished_semaphore];

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .build();

        let submits = [submit_info];
        
        self.images_in_flight[image_index as usize] = None;
        unsafe { device.reset_fences(&fences) }.expect("Unable to reset fences");
        unsafe { device.queue_submit(self.context.graphics_queue, &submits, *fence).expect("Unable to submit to queue") };

        let swap_chains = [self.swap_chain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swap_chains)
            .wait_semaphores(&signal_semaphores)
            .image_indices(&image_indices);

        unsafe { 
            match self.swap_chain_ext.queue_present(self.context.present_queue, &present_info) {
                Ok(false) => if self.framebuffer_resized { return self.recreate_swap_chain(&window) }
                Ok(true) => return self.recreate_swap_chain(&window),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swap_chain(&window),
                _ => panic!("Unable to present")
            }
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub unsafe fn cleanup_swap_chain(&mut self) {
        for framebuffer in self.swap_chain_framebuffers.drain(..) {
            self.context.device.destroy_framebuffer(framebuffer, None);
        }

        self.context.device.free_command_buffers(self.context.command_pool, &self.command_buffers);

        self.context.device.destroy_pipeline(self.pipeline, None);

        self.context.device.destroy_pipeline_layout(self.pipeline_layout, None);

        self.context.device.destroy_render_pass(self.render_pass, None);

        for view in self.swap_chain_image_views.drain(..) {
            self.context.device.destroy_image_view(view, None);
        }

        let swapchain = khr::Swapchain::new(&self.context.instance, &self.context.device);
        swapchain.destroy_swapchain(self.swap_chain, None);
    }

    pub fn resized(&mut self, new_size: PhysicalSize<u32>) {
        self.framebuffer_resized = true;
    }
}




impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            self.cleanup_swap_chain();
            // WARNING: self.pipeline is now invalid!
            // WARNING: self.pipeline_layout is now invalid!
            // self.swap_chain_image_views will now be empty
            // self.swap_chain_framebuffers will now be empty
            // WARNING: self.swap_chain is now invalid!

            self.context.device.destroy_buffer(self.vertex_buffer, None);
            self.context.device.free_memory(self.vertex_buffer_memory, None);

            for semaphore in self.render_finished_semaphores.drain(..) {
                self.context.device.destroy_semaphore(semaphore, None);
            }

            for semaphore in self.image_available_semaphores.drain(..) {
                self.context.device.destroy_semaphore(semaphore, None);
            }

            for fence in self.in_flight_fences.drain(..) {
                self.context.device.destroy_fence(fence, None);
            }

            self.context.device.destroy_command_pool(self.context.command_pool, None);
            // WARNING: self.command_pool is now invalid!

            self.context.debug_messenger.map(|m| {
                self.context.debug_utils.as_ref().map(|d| {
                    d.destroy_debug_utils_messenger(m, None);
                })
            });

            self.surface_loader.destroy_surface(self.context.surface, None);
            // WARNING: self.surface is now invalid!

            self.context.device.destroy_device(None);
            // WARNING: self.device is now invalid!

            self.context.instance.destroy_instance(None);
            // WARNING: self.instance is now invalid!
        }
    }
}

pub fn init_window(event_loop: &EventLoop<()>) -> Window {
    WindowBuilder::new()
        .with_title("Hello, Triangle")
        .with_inner_size(LogicalSize::new(600, 600))
        .build(event_loop)
        .unwrap()
}

fn main() {
    let event_loop = EventLoop::new();
    let window = init_window(&event_loop);
    let app = HelloTriangleApplication::new(&window);
    main_loop(event_loop, window, app);
}

// *************************
//        MAIN LOOP
// *************************

fn main_loop(event_loop: EventLoop<()>, window: Window, mut app: HelloTriangleApplication) -> () {
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(new_size) => app.resized(new_size),
                    _ => {}
                }
            }
            Event::LoopDestroyed => {
                println!("Exiting!");
                unsafe { app.context.device.device_wait_idle().expect("Failed to wait for device idle") }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => {
                app.draw_frame(&window);
            }
            _ => ()
        }
    });
}

fn find_memory_type(instance: &Instance, physical_device: &vk::PhysicalDevice, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
    let device_memory_properties = unsafe { instance.get_physical_device_memory_properties(*physical_device) };
    for i in 0..device_memory_properties.memory_type_count {
        let has_type = type_filter & (1 << i) != 0;
        let has_properties =  device_memory_properties.memory_types[i as usize].property_flags.contains(properties);
        if has_type && has_properties { return i }
    }

    panic!("Unable to find suitable memory type")
}

// Buffers
fn create_buffer(context: &VulkanContext, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::Buffer, vk::DeviceMemory) {
    let create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { context.device.create_buffer(&create_info, None).expect("Unable to create buffer") };

    let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };

    let memory_type = find_memory_type(&context.instance, &context.physical_device, requirements.memory_type_bits, properties);
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type)
        .build();
    
    let device_memory = unsafe { context.device.allocate_memory(&alloc_info, None).expect("Unable to allocate memory") };
    unsafe { context.device.bind_buffer_memory(buffer, device_memory, 0).expect("Unable to bind memory"); }


    (buffer, device_memory)
}

fn create_vertex_buffer(context: &VulkanContext, vertices: &Vec<Vertex>) -> (vk::Buffer, vk::DeviceMemory) {
    let size = (size_of::<Vertex>() * vertices.len()) as u64;
    let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
    let properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let (staging_buffer, staging_memory) = create_buffer(context, size, staging_usage, properties);
    unsafe { 
        let data = context.device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty()).expect("Unable to map memory");
        let data = data as *mut Vertex;
        std::ptr::copy_nonoverlapping(vertices.as_ptr(), data, vertices.len());
        context.device.unmap_memory(staging_memory)
    }

    let vertex_usage = vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(context, size, vertex_usage, properties);
    copy_buffer(context, staging_buffer, vertex_buffer, size);

    unsafe {
        context.device.destroy_buffer(staging_buffer, None);
        context.device.free_memory(staging_memory, None);
    }

    (vertex_buffer, vertex_buffer_memory)
}

fn copy_buffer(context: &VulkanContext, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize) {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(context.command_pool);

    let command_buffer = unsafe { context.device.allocate_command_buffers(&alloc_info).map(|mut b| b.pop().unwrap()).expect("Unable to allocate command buffer") };

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { context.device.begin_command_buffer(command_buffer, &begin_info).expect("Unable to begin command buffer") } 

    let copy_region = vk::BufferCopy::builder()
        .src_offset(0)
        .dst_offset(0)
        .size(size)
        .build();

    let regions = [copy_region];

    unsafe { 
        context.device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &regions);
        context.device.end_command_buffer(command_buffer).expect("Unable to end command buffer");
    }

    let command_buffers = &[command_buffer];

    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers)
        .build();

    let submit_info = &[submit_info];

    unsafe {
        context.device.queue_submit(context.graphics_queue, submit_info, vk::Fence::null()).expect("Unable to submit to queue");
        context.device.queue_wait_idle(context.graphics_queue).expect("Unable to wait idle");
        context.device.free_command_buffers(context.command_pool, command_buffers)
    }
}

// Semaphores
fn create_sync_objects(device: &Device, swapchain_images_size: usize) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<Option<vk::Fence>>) {
    let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut inflight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut images_in_flight = Vec::with_capacity(swapchain_images_size);

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);
    
    for _ in 0..MAX_FRAMES_IN_FLIGHT {

        let image_available = unsafe { device.create_semaphore(&semaphore_info, None).expect("Unable to create semaphore") };
        image_available_semaphores.push(image_available);

        let render_finished = unsafe { device.create_semaphore(&semaphore_info, None).expect("Unable to create semaphore") };
        render_finished_semaphores.push(render_finished);

        let in_flight_fence = unsafe { device.create_fence(&fence_info, None)}.expect("Unable to create fence!");
        inflight_fences.push(in_flight_fence);
    }

    for _ in 0..swapchain_images_size {
        images_in_flight.push(None);
    }

    (image_available_semaphores, render_finished_semaphores, inflight_fences, images_in_flight)
}

// Command Buffers/Pools
fn create_command_pool(queue_family_indices: QueueFamilyIndices, device: &Device) -> vk::CommandPool {
    let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.graphics_family.unwrap());

    unsafe { device.create_command_pool(&pool_info, None).expect("Unable to create command pool") }
}

fn create_command_buffers(context: &VulkanContext, swap_chain_framebuffers: &Vec<vk::Framebuffer>, render_pass: vk::RenderPass, extent: vk::Extent2D, graphics_pipeline: vk::Pipeline, vertex_buffer: vk::Buffer, vertex_count: u32) -> Vec<vk::CommandBuffer> {
    let device = &context.device;
    let command_pool = &context.command_pool;

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swap_chain_framebuffers.len() as u32);
    
    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info).expect("Unable to allocate frame_buffers") };

    for (command_buffer, framebuffer) in command_buffers.iter().zip(swap_chain_framebuffers) {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe { device.begin_command_buffer(*command_buffer, &begin_info).expect("Unable to begin command buffer"); }
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0},
                extent,
            };
        
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0]
            }
        };

        let clear_colors = [clear_color];

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(*framebuffer)
            .render_area(render_area)
            .clear_values(&clear_colors);

        let vertex_buffers = [vertex_buffer];
        let offsets = [0];

        unsafe { 
            device.cmd_begin_render_pass(*command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline);
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &vertex_buffers, &offsets);
            device.cmd_draw(*command_buffer, vertex_count, 1, 0, 0);
            device.cmd_end_render_pass(*command_buffer);
            device.end_command_buffer(*command_buffer).expect("Unable to record command buffer!");
        }
    }

    command_buffers
}

// Graphics Pipeline
fn create_graphics_pipeline(device: &Device, extent: vk::Extent2D, render_pass: vk::RenderPass) -> (vk::PipelineLayout, vk::Pipeline) { 
    let vert_shader_code = include_bytes!("./shaders/shader.vert.spv");
    let frag_shader_code = include_bytes!("./shaders/shader.frag.spv");

    let vertex_shader_module = create_shader_module(device, vert_shader_code);
    let frag_shader_module = create_shader_module(device, frag_shader_code);
    let name = CString::new("main").unwrap();

    let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(name.as_c_str())
        .build();

    let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(name.as_c_str())
        .build();
    
    let shader_stages = [vertex_shader_stage_info, frag_shader_stage_info]; 

    let binding_description = [Vertex::get_binding_description()];
    let attribute_descriptions = Vertex::get_attribute_descriptions();

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_descriptions);
        
    let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(extent.width as f32)
        .height(extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)
        .build();

    let viewports = [viewport];
    
    let offset = vk::Offset2D { x: 0, y: 0}; 
    let scissor = vk::Rect2D::builder()
        .offset(offset)
        .extent(extent)
        .build();

    let scissors = [scissor];
    
    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .viewports(&viewports)
        .scissor_count(1)
        .scissors(&scissors);
    
    let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);
    
    let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
        .blend_enable(false)
        .build();

    let color_blend_attachments = [color_blend_attachment];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder();
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None).expect("Unable to create pipeline layout") };

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_create_info)
        .viewport_state(&viewport_state_create_info)
        .rasterization_state(&rasterizer_create_info)
        .multisample_state(&multisampling_create_info)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build();

    let create_infos = [pipeline_create_info];
    
    let mut graphics_pipelines = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None).unwrap() };

    // Cleanup
    unsafe { device.destroy_shader_module(vertex_shader_module, None) } ;
    unsafe { device.destroy_shader_module(frag_shader_module, None) } ;

    return (pipeline_layout, graphics_pipelines.remove(0));
}

fn create_shader_module(device: &Device, bytes: &[u8]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::builder()
        .code(bytes.as_slice_of::<u32>().unwrap());

    unsafe { device.create_shader_module(&create_info, None).expect("Unable to create shader module") }
}

fn create_render_pass(format: vk::Format, device: &Device) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .samples(vk::SampleCountFlags::TYPE_1)
        .build();

    let color_attachments = [color_attachment];

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let color_attachment_refs = [color_attachment_ref];

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .build();
    let subpasses = [subpass];

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build();
    let dependencies = [dependency];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&color_attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&render_pass_create_info, None).unwrap() }
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
fn create_swap_chain (
    context: &VulkanContext,
    window: &Window) -> (khr::Swapchain, vk::SwapchainKHR, vk::Format, vk::Extent2D) {
    let instance = &context.instance;
    let entry = &context.entry;
    let device = &context.device;
    let physical_device = context.physical_device;
    let surface = context.surface;

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

    let swapchain_ext = khr::Swapchain::new(instance, device);
    let swapchain = unsafe { swapchain_ext.create_swapchain(&create_info, None) }.expect("Unable to create Swapchain");
    (swapchain_ext, swapchain, surface_format.format, extent)
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

fn get_swap_chain_images(context: &VulkanContext, swap_chain: vk::SwapchainKHR) -> Vec<vk::Image> {
    let swap_chain_ext = khr::Swapchain::new(&context.instance, &context.device);
    unsafe { swap_chain_ext.get_swapchain_images(swap_chain).expect("Unable to get swapchain images") }
}

fn create_image_views(swap_chain_images: &mut Vec<vk::Image>, format: vk::Format, context: &VulkanContext) -> Vec<vk::ImageView> {
    swap_chain_images.drain(..).map(|image| {
        let components = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY)
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

        unsafe { context.device.create_image_view(&create_info, None).expect("Unable to get image view") }
    }).collect::<Vec<_>>()
}

fn create_framebuffers(image_views: &Vec<vk::ImageView>, device: &Device, render_pass: vk::RenderPass, extent: vk::Extent2D) -> Vec<vk::Framebuffer> {
    image_views.iter().map(|v| {
        let attachments = [*v]; //.. really?
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        
        unsafe { device.create_framebuffer(&create_info, None).unwrap() }
    })
    .collect::<Vec<_>>()
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