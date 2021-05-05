#![allow(dead_code)]
#![allow(unused_variables)]

mod swap_chain;
mod vulkan_context;
use crate::vulkan_context::VulkanContext;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use byte_slice_cast::AsSliceOf;
use cgmath::{vec2, vec3, Vector2, Vector3};
use memoffset::offset_of;
use std::{ffi::CString, mem::size_of};
use swap_chain::SwapChain;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

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

        for (i, queue) in unsafe { instance.get_physical_device_queue_family_properties(device) }
            .iter()
            .enumerate()
        {
            let i = i as u32;
            if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }
            if unsafe { surface.get_physical_device_surface_support(device, i, surface_khr) }
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

struct Vertex {
    pos: Vector2<f32>,
    colour: Vector3<f32>,
}

impl Vertex {
    fn new(pos: Vector2<f32>, colour: Vector3<f32>) -> Self {
        Self { pos, colour }
    }
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

struct HelloTriangleApplication {
    context: VulkanContext,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<Option<vk::Fence>>,
    current_frame: usize,
    framebuffer_resized: bool,
    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    indices: Vec<u16>,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    swap_chain: SwapChain,
    frame_buffers: Vec<vk::Framebuffer>,
}

impl HelloTriangleApplication {
    pub fn new(window: &Window) -> HelloTriangleApplication {
        // Build the Vulkan Context
        let context = VulkanContext::new(&window);

        // Build the swapchain
        let mut swap_chain = SwapChain::new(&context, window);

        // Create render pass, pipeline
        let render_pass = create_render_pass(swap_chain.format, &context.device);
        let (pipeline_layout, pipeline) =
            create_graphics_pipeline(&context.device, swap_chain.extent, render_pass);

        // Create swapchain framebuffers
        let frame_buffers = swap_chain.create_framebuffers(&context, render_pass);

        // Create vertex buffer
        let vertices = vec![
            Vertex::new(vec2(-0.5, -0.5), vec3(1.0, 0.0, 1.0)),
            Vertex::new(vec2(0.5, -0.5), vec3(0.0, 1.0, 1.0)),
            Vertex::new(vec2(0.5, 0.5), vec3(0.0, 0.0, 1.0)),
            Vertex::new(vec2(-0.5, 0.5), vec3(1.0, 0.0, 1.0)),
        ];
        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(&context, &vertices);

        // Index buffer
        let indices = vec![0, 1, 2, 2, 3, 0];
        let (index_buffer, index_buffer_memory) = create_index_buffer(&context, &indices);

        // Command buffers
        let command_buffers = create_command_buffers(
            &context,
            &frame_buffers,
            render_pass,
            swap_chain.extent,
            pipeline,
            vertex_buffer,
            vertices.len(),
            index_buffer,
            indices.len(),
        );

        // Sync objects
        let (image_available, render_finished, in_flight_fences, images_in_flight) =
            create_sync_objects(&context.device, swap_chain.image_views.len());

        HelloTriangleApplication {
            context,
            swap_chain,
            render_pass,
            pipeline_layout,
            pipeline,
            command_buffers,
            image_available_semaphores: image_available,
            render_finished_semaphores: render_finished,
            in_flight_fences,
            images_in_flight,
            current_frame: 0,
            framebuffer_resized: false,
            vertices,
            vertex_buffer,
            vertex_buffer_memory,
            indices,
            index_buffer,
            index_buffer_memory,
            frame_buffers,
        }
    }

    pub fn draw_frame(&mut self, window: &Window) {
        let device = &self.context.device;
        let instance = &self.context.instance;

        let fence = self
            .in_flight_fences
            .get(self.current_frame)
            .expect("Unable to get fence!");
        let fences = [*fence];

        unsafe { device.wait_for_fences(&fences, true, u64::MAX) }
            .expect("Unable to wait for fence");

        let image_available_semaphore = self
            .image_available_semaphores
            .get(self.current_frame)
            .expect("Unable to get image_available semaphore for frame!");
        let render_finished_semaphore = self
            .render_finished_semaphores
            .get(self.current_frame)
            .expect("Unable to get render_finished semaphore");

        let image_index = unsafe {
            match self.swap_chain.loader.acquire_next_image(
                self.swap_chain.handle,
                u64::MAX,
                *image_available_semaphore,
                vk::Fence::null(),
            ) {
                Ok((index, _)) => index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swap_chain(&window),
                _ => panic!("Failed to acquire swap chain image!"),
            }
        };

        if let Some(image_in_flight_fence) =
            unsafe { self.images_in_flight.get_unchecked(image_index as usize) }
        {
            let fences = [*image_in_flight_fence];
            unsafe { device.wait_for_fences(&fences, true, u64::MAX) }
                .expect("Unable to wait for image_in_flight_fence");
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
        unsafe {
            device
                .queue_submit(self.context.graphics_queue, &submits, *fence)
                .expect("Unable to submit to queue")
        };

        let swap_chains = [self.swap_chain.handle];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swap_chains)
            .wait_semaphores(&signal_semaphores)
            .image_indices(&image_indices);

        unsafe {
            match self
                .swap_chain
                .loader
                .queue_present(self.context.present_queue, &present_info)
            {
                Ok(false) => {
                    if self.framebuffer_resized {
                        return self.recreate_swap_chain(&window);
                    }
                }
                Ok(true) => return self.recreate_swap_chain(&window),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swap_chain(&window),
                _ => panic!("Unable to present"),
            }
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn recreate_swap_chain(&mut self, window: &Window) {
        unsafe {
            self.context
                .device
                .device_wait_idle()
                .expect("Could not wait idle")
        };

        // Remove the old swap chain
        unsafe { self.cleanup_swap_chain() };

        // Create a new one
        self.swap_chain = SwapChain::new(&self.context, window);

        // Build a render pass and pipeline
        self.render_pass = create_render_pass(self.swap_chain.format, &self.context.device);
        let (pipeline_layout, pipeline) = create_graphics_pipeline(
            &self.context.device,
            self.swap_chain.extent,
            self.render_pass,
        );
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;

        // Create framebuffers for the swapchain
        self.swap_chain
            .create_framebuffers(&self.context, self.render_pass);

        self.command_buffers = create_command_buffers(
            &self.context,
            &self.frame_buffers,
            self.render_pass,
            self.swap_chain.extent,
            pipeline,
            self.vertex_buffer,
            self.vertices.len(),
            self.index_buffer,
            self.indices.len(),
        );
        self.framebuffer_resized = false;
    }

    pub fn resized(&mut self, new_size: PhysicalSize<u32>) {
        self.framebuffer_resized = true;
    }

    pub unsafe fn cleanup_swap_chain(&mut self) {
        for framebuffer in self.frame_buffers.drain(..) {
            self.context.device.destroy_framebuffer(framebuffer, None);
        }

        self.context
            .device
            .free_command_buffers(self.context.command_pool, &self.command_buffers);

        self.context.device.destroy_pipeline(self.pipeline, None);

        self.context
            .device
            .destroy_pipeline_layout(self.pipeline_layout, None);

        self.context
            .device
            .destroy_render_pass(self.render_pass, None);

        for view in self.swap_chain.image_views.drain(..) {
            self.context.device.destroy_image_view(view, None);
        }

        self.swap_chain
            .loader
            .destroy_swapchain(self.swap_chain.handle, None);
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
            self.context
                .device
                .free_memory(self.vertex_buffer_memory, None);

            self.context.device.destroy_buffer(self.index_buffer, None);
            self.context
                .device
                .free_memory(self.index_buffer_memory, None);

            for semaphore in self.render_finished_semaphores.drain(..) {
                self.context.device.destroy_semaphore(semaphore, None);
            }

            for semaphore in self.image_available_semaphores.drain(..) {
                self.context.device.destroy_semaphore(semaphore, None);
            }

            for fence in self.in_flight_fences.drain(..) {
                self.context.device.destroy_fence(fence, None);
            }
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
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(new_size) => app.resized(new_size),
                _ => {}
            },
            Event::LoopDestroyed => {
                println!("Exiting!");
                unsafe {
                    app.context
                        .device
                        .device_wait_idle()
                        .expect("Failed to wait for device idle")
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => {
                app.draw_frame(&window);
            }
            _ => (),
        }
    });
}

fn find_memory_type(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };
    for i in 0..device_memory_properties.memory_type_count {
        let has_type = type_filter & (1 << i) != 0;
        let has_properties = device_memory_properties.memory_types[i as usize]
            .property_flags
            .contains(properties);
        if has_type && has_properties {
            return i;
        }
    }

    panic!("Unable to find suitable memory type")
}

// Buffers
fn create_buffer(
    context: &VulkanContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        context
            .device
            .create_buffer(&create_info, None)
            .expect("Unable to create buffer")
    };

    let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };

    let memory_type = find_memory_type(
        &context.instance,
        &context.physical_device,
        requirements.memory_type_bits,
        properties,
    );
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type)
        .build();

    let device_memory = unsafe {
        context
            .device
            .allocate_memory(&alloc_info, None)
            .expect("Unable to allocate memory")
    };
    unsafe {
        context
            .device
            .bind_buffer_memory(buffer, device_memory, 0)
            .expect("Unable to bind memory");
    }

    (buffer, device_memory)
}

fn create_vertex_buffer(
    context: &VulkanContext,
    vertices: &Vec<Vertex>,
) -> (vk::Buffer, vk::DeviceMemory) {
    create_buffer_from_data(context, vk::BufferUsageFlags::VERTEX_BUFFER, vertices)
}

fn create_index_buffer(
    context: &VulkanContext,
    indices: &Vec<u16>,
) -> (vk::Buffer, vk::DeviceMemory) {
    create_buffer_from_data(context, vk::BufferUsageFlags::INDEX_BUFFER, indices)
}

fn create_buffer_from_data<T>(
    context: &VulkanContext,
    final_usage: vk::BufferUsageFlags,
    data: &Vec<T>,
) -> (vk::Buffer, vk::DeviceMemory) {
    let size = (size_of::<T>() * data.len()) as u64;
    let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
    let staging_properties =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let (staging_buffer, staging_memory) =
        create_buffer(context, size, staging_usage, staging_properties);
    unsafe {
        let dst = context
            .device
            .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())
            .expect("Unable to map memory");
        let dst = dst as *mut T;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        context.device.unmap_memory(staging_memory)
    }

    let final_usage = final_usage | vk::BufferUsageFlags::TRANSFER_DST;
    let final_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let (final_buffer, final_buffer_memory) =
        create_buffer(context, size, final_usage, final_properties);
    copy_buffer(context, staging_buffer, final_buffer, size);

    unsafe {
        context.device.destroy_buffer(staging_buffer, None);
        context.device.free_memory(staging_memory, None);
    }

    (final_buffer, final_buffer_memory)
}

fn copy_buffer(
    context: &VulkanContext,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: vk::DeviceSize,
) {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(context.command_pool);

    let command_buffer = unsafe {
        context
            .device
            .allocate_command_buffers(&alloc_info)
            .map(|mut b| b.pop().unwrap())
            .expect("Unable to allocate command buffer")
    };

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        context
            .device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Unable to begin command buffer")
    }

    let copy_region = vk::BufferCopy::builder()
        .src_offset(0)
        .dst_offset(0)
        .size(size)
        .build();

    let regions = [copy_region];

    unsafe {
        context
            .device
            .cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &regions);
        context
            .device
            .end_command_buffer(command_buffer)
            .expect("Unable to end command buffer");
    }

    let command_buffers = &[command_buffer];

    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers)
        .build();

    let submit_info = &[submit_info];

    unsafe {
        context
            .device
            .queue_submit(context.graphics_queue, submit_info, vk::Fence::null())
            .expect("Unable to submit to queue");
        context
            .device
            .queue_wait_idle(context.graphics_queue)
            .expect("Unable to wait idle");
        context
            .device
            .free_command_buffers(context.command_pool, command_buffers)
    }
}

// Semaphores
fn create_sync_objects(
    device: &Device,
    swapchain_images_size: usize,
) -> (
    Vec<vk::Semaphore>,
    Vec<vk::Semaphore>,
    Vec<vk::Fence>,
    Vec<Option<vk::Fence>>,
) {
    let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut inflight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut images_in_flight = Vec::with_capacity(swapchain_images_size);

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let image_available = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .expect("Unable to create semaphore")
        };
        image_available_semaphores.push(image_available);

        let render_finished = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .expect("Unable to create semaphore")
        };
        render_finished_semaphores.push(render_finished);

        let in_flight_fence =
            unsafe { device.create_fence(&fence_info, None) }.expect("Unable to create fence!");
        inflight_fences.push(in_flight_fence);
    }

    for _ in 0..swapchain_images_size {
        images_in_flight.push(None);
    }

    (
        image_available_semaphores,
        render_finished_semaphores,
        inflight_fences,
        images_in_flight,
    )
}

fn create_command_buffers(
    context: &VulkanContext,
    swap_chain_framebuffers: &Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    graphics_pipeline: vk::Pipeline,
    vertex_buffer: vk::Buffer,
    vertex_count: usize,
    index_buffer: vk::Buffer,
    index_count: usize,
) -> Vec<vk::CommandBuffer> {
    let device = &context.device;
    let command_pool = &context.command_pool;

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swap_chain_framebuffers.len() as u32);

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .expect("Unable to allocate frame_buffers")
    };

    for (command_buffer, framebuffer) in command_buffers.iter().zip(swap_chain_framebuffers) {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            device
                .begin_command_buffer(*command_buffer, &begin_info)
                .expect("Unable to begin command buffer");
        }
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
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
            device.cmd_begin_render_pass(
                *command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &vertex_buffers, &offsets);
            device.cmd_bind_index_buffer(*command_buffer, index_buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(*command_buffer, index_count as u32, 1, 0, 0, 0);
            device.cmd_end_render_pass(*command_buffer);
            device
                .end_command_buffer(*command_buffer)
                .expect("Unable to record command buffer!");
        }
    }

    command_buffers
}

// Graphics Pipeline
fn create_graphics_pipeline(
    device: &Device,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
) -> (vk::PipelineLayout, vk::Pipeline) {
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

    let offset = vk::Offset2D { x: 0, y: 0 };
    let scissor = vk::Rect2D::builder().offset(offset).extent(extent).build();

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
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)
        .build();

    let color_blend_attachments = [color_blend_attachment];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder();
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Unable to create pipeline layout")
    };

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

    let mut graphics_pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
            .unwrap()
    };

    // Cleanup
    unsafe { device.destroy_shader_module(vertex_shader_module, None) };
    unsafe { device.destroy_shader_module(frag_shader_module, None) };

    return (pipeline_layout, graphics_pipelines.remove(0));
}

fn create_shader_module(device: &Device, bytes: &[u8]) -> vk::ShaderModule {
    let create_info =
        vk::ShaderModuleCreateInfo::builder().code(bytes.as_slice_of::<u32>().unwrap());

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .expect("Unable to create shader module")
    }
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

    unsafe {
        device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap()
    }
}
