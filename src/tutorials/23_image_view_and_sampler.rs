mod swap_chain;
mod vulkan_context;
use crate::vulkan_context::VulkanContext;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use byte_slice_cast::AsSliceOf;
use cgmath::{perspective, vec2, vec3, Deg, Matrix4, Point3, Vector2, Vector3};
use itertools::izip;
use memoffset::offset_of;
use std::{ffi::CString, mem::size_of, time::Instant};
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

#[repr(C)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

#[repr(C)]
struct Vertex {
    pos: Vector3<f32>,
    colour: Vector3<f32>,
    texture_coordinate: Vector2<f32>,
}

impl Vertex {
    fn new(pos: Vector3<f32>, colour: Vector3<f32>, texture_coordinate: Vector2<f32>) -> Self {
        Self {
            pos,
            colour,
            texture_coordinate,
        }
    }
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }
    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let position_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32)
            .build();

        let colour_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, colour) as u32)
            .build();

        let texture_coordinate_attribute = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, texture_coordinate) as u32)
            .build();

        [
            position_attribute,
            colour_attribute,
            texture_coordinate_attribute,
        ]
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
    _vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    indices: Vec<u16>,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    swap_chain: SwapChain,
    frame_buffers: Vec<vk::Framebuffer>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    start_time: Instant,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
}

impl HelloTriangleApplication {
    pub fn new(window: &Window) -> HelloTriangleApplication {
        let start_time = Instant::now();
        // Build the Vulkan Context
        let context = VulkanContext::new(&window);

        // Build the swapchain
        let mut swap_chain = SwapChain::new(&context, window);

        // Create render pass, pipeline
        let render_pass = create_render_pass(swap_chain.format, &context.device);
        let descriptor_set_layout = create_descriptor_set_layout(&context);
        let (pipeline_layout, pipeline) = create_graphics_pipeline(
            &context.device,
            swap_chain.extent,
            render_pass,
            &[descriptor_set_layout],
        );

        // Create swapchain framebuffers
        let frame_buffers = swap_chain.create_framebuffers(&context, render_pass);

        // Create texture image
        let (texture_image, texture_image_memory) = create_texture_image(&context);
        let texture_image_view = create_texture_image_view(&context, texture_image);
        let texture_sampler = create_texture_sampler(&context);

        // Create vertex buffer
        let vertices = vec![
            Vertex::new(vec3(-0.5, -0.5, 0.0), vec3(1.0, 0.0, 1.0), vec2(0.0, 0.0)),
            Vertex::new(vec3(0.5, -0.5, 0.0), vec3(0.0, 1.0, 1.0), vec2(1.0, 0.0)),
            Vertex::new(vec3(0.5, 0.5, 0.0), vec3(0.0, 0.0, 1.0), vec2(1.0, 1.0)),
            Vertex::new(vec3(-0.5, 0.5, 0.0), vec3(1.0, 0.0, 1.0), vec2(0.0, 1.0)),
            Vertex::new(vec3(-0.5, -0.5, -0.5), vec3(1.0, 0.0, 1.0), vec2(0.0, 0.0)),
            Vertex::new(vec3(0.5, -0.5, -0.5), vec3(0.0, 1.0, 1.0), vec2(1.0, 0.0)),
            Vertex::new(vec3(0.5, 0.5, -0.5), vec3(0.0, 0.0, 1.0), vec2(1.0, 1.0)),
            Vertex::new(vec3(-0.5, 0.5, -0.5), vec3(1.0, 0.0, 1.0), vec2(0.0, 1.0)),
        ];
        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(&context, &vertices);

        // Index buffer
        let indices = vec![0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];
        let (index_buffer, index_buffer_memory) = create_index_buffer(&context, &indices);

        let (uniform_buffers, uniform_buffers_memory) =
            create_uniform_buffers(&context, &swap_chain);

        // Descriptor pool
        let descriptor_pool = create_descriptor_pool(&context, &swap_chain);
        let descriptor_sets = create_descriptor_sets(
            &context,
            &uniform_buffers,
            &descriptor_pool,
            descriptor_set_layout,
            texture_image_view,
            texture_sampler,
        );

        // Command buffers
        let command_buffers = create_command_buffers(
            &context,
            &frame_buffers,
            render_pass,
            swap_chain.extent,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            index_buffer,
            indices.len(),
            &descriptor_sets,
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
            _vertices: vertices,
            vertex_buffer,
            vertex_buffer_memory,
            indices,
            index_buffer,
            index_buffer_memory,
            frame_buffers,
            descriptor_set_layout,
            uniform_buffers,
            uniform_buffers_memory,
            descriptor_pool,
            start_time,
            texture_image_view,
            texture_image,
            texture_image_memory,
            texture_sampler,
        }
    }

    pub fn draw_frame(&mut self, window: &Window) {
        let fence = self
            .in_flight_fences
            .get(self.current_frame)
            .expect("Unable to get fence!");
        let fences = [*fence];

        unsafe { self.context.device.wait_for_fences(&fences, true, u64::MAX) }
            .expect("Unable to wait for fence");

        let image_index = unsafe {
            let image_available_semaphore = self
                .image_available_semaphores
                .get(self.current_frame)
                .expect("Unable to get image_available semaphore for frame!");
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
            unsafe { self.context.device.wait_for_fences(&fences, true, u64::MAX) }
                .expect("Unable to wait for image_in_flight_fence");
        }

        self.update_uniform_buffer(image_index, self.start_time);

        let render_finished_semaphore = self
            .render_finished_semaphores
            .get(self.current_frame)
            .expect("Unable to get render_finished semaphore");
        let image_available_semaphore = self
            .image_available_semaphores
            .get(self.current_frame)
            .expect("Unable to get image_available semaphore for frame!");

        let fence = self
            .in_flight_fences
            .get(self.current_frame)
            .expect("Unable to get fence!");
        let fences = [*fence];

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
        unsafe { self.context.device.reset_fences(&fences) }.expect("Unable to reset fences");
        unsafe {
            self.context
                .device
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
            &[self.descriptor_set_layout],
        );
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;

        // Create framebuffers for the swapchain
        self.frame_buffers = self
            .swap_chain
            .create_framebuffers(&self.context, self.render_pass);

        // UBOs
        let (uniform_buffers, uniform_buffers_memory) =
            create_uniform_buffers(&self.context, &self.swap_chain);
        self.uniform_buffers = uniform_buffers;
        self.uniform_buffers_memory = uniform_buffers_memory;

        // Descriptor Pool
        self.descriptor_pool = create_descriptor_pool(&self.context, &self.swap_chain);
        let descriptor_sets = create_descriptor_sets(
            &self.context,
            &self.uniform_buffers,
            &self.descriptor_pool,
            self.descriptor_set_layout,
            self.texture_image_view,
            self.texture_sampler,
        );

        self.command_buffers = create_command_buffers(
            &self.context,
            &self.frame_buffers,
            self.render_pass,
            self.swap_chain.extent,
            self.pipeline,
            self.pipeline_layout,
            self.vertex_buffer,
            self.index_buffer,
            self.indices.len(),
            &descriptor_sets,
        );

        self.framebuffer_resized = false;
    }

    pub fn update_uniform_buffer(&mut self, image_index: u32, start_time: Instant) {
        let current_time = Instant::now();
        let delta = current_time.duration_since(start_time).as_secs_f32();
        let angle = Deg(10.0 * delta);
        let model = Matrix4::from_angle_z(angle);
        // let model = Matrix4::from_scale(1.0);

        let eye = Point3::new(2.0, 2.0, 2.0);
        let center = Point3::new(0.0, 0.0, 0.0);
        let up = vec3(0.0, 1.0, 0.0);
        let view = Matrix4::look_at_rh(eye, center, up);

        let fovy = Deg(45.0);
        let aspect = self.swap_chain.extent.width / self.swap_chain.extent.height;
        let near = 0.1;
        let far = 10.0;
        let projection = perspective(fovy, aspect as f32, near, far);
        // projection[1][1] *= -1.0;

        let ubo = UniformBufferObject {
            model,
            view,
            projection,
        };

        let memory = self.uniform_buffers_memory[image_index as usize];
        unsafe { self.context.copy_pointer_to_device_memory(&ubo, memory, 1) }
    }

    pub fn resized(&mut self, _new_size: PhysicalSize<u32>) {
        // println!("Resized! New size: {:?}", _new_size);
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

        for buffer in self.uniform_buffers.drain(..) {
            self.context.device.destroy_buffer(buffer, None);
        }

        for memory in self.uniform_buffers_memory.drain(..) {
            self.context.device.free_memory(memory, None)
        }

        self.context
            .device
            .destroy_descriptor_pool(self.descriptor_pool, None)
    }
}

fn create_texture_sampler(context: &VulkanContext) -> vk::Sampler {
    let create_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0)
        .build();

    unsafe {
        context
            .device
            .create_sampler(&create_info, None)
            .expect("Unable to create sampler")
    }
}

fn create_texture_image(context: &VulkanContext) -> (vk::Image, vk::DeviceMemory) {
    let img = image::io::Reader::open("./src/tutorials/images/malaysia.jpg")
        .expect("Unable to read image")
        .decode()
        .expect("Unable to read image")
        .to_rgba8();

    let width = img.width();
    let height = img.height();
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let buf = img.into_raw();
    let size = buf.len() * 8;

    let usage = vk::BufferUsageFlags::TRANSFER_SRC;
    let staging_properties =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let (staging_buffer, staging_memory) =
        context.create_buffer(size as u64, usage, staging_properties);
    unsafe { context.copy_pointer_to_device_memory(buf.as_ptr(), staging_memory, buf.len()) };
    let format = vk::Format::R8G8B8A8_SRGB;
    let tiling = vk::ImageTiling::OPTIMAL;

    let image_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;

    let (texture_image, texture_image_memory) =
        context.create_image(extent, image_properties, usage, format, tiling);

    let old_layout = vk::ImageLayout::UNDEFINED;
    let copy_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    let final_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

    context.transition_image_layout(texture_image, format, old_layout, copy_layout);
    context.copy_buffer_to_image(staging_buffer, texture_image, extent);
    context.transition_image_layout(texture_image, format, copy_layout, final_layout);

    unsafe {
        context.device.free_memory(staging_memory, None);
        context.device.destroy_buffer(staging_buffer, None);
    };

    (texture_image, texture_image_memory)
}

fn create_texture_image_view(context: &VulkanContext, texture_image: vk::Image) -> vk::ImageView {
    context.create_image_view(texture_image, vk::Format::R8G8B8A8_SRGB)
}

fn create_descriptor_sets(
    context: &VulkanContext,
    uniform_buffers: &Vec<vk::Buffer>,
    descriptor_pool: &vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..uniform_buffers.len())
        .map(|_| descriptor_set_layout.clone())
        .collect::<Vec<_>>();

    let buffer_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(*descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = unsafe {
        context
            .device
            .allocate_descriptor_sets(&buffer_info)
            .expect("Unable to allocate descriptor sets")
    };

    let range = size_of::<UniformBufferObject>() as u64;
    for (descriptor_set, buffer) in descriptor_sets.iter().zip(uniform_buffers) {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(*buffer)
            .range(range)
            .offset(0)
            .build();

        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture_image_view)
            .sampler(texture_sampler)
            .build();

        let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(*descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&[buffer_info])
            .build();

        let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(*descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&[image_info])
            .build();

        unsafe {
            context
                .device
                .update_descriptor_sets(&[ubo_descriptor_write, sampler_descriptor_write], &[])
        }
    }

    descriptor_sets
}

fn create_descriptor_pool(context: &VulkanContext, swap_chain: &SwapChain) -> vk::DescriptorPool {
    let size = swap_chain.images.len() as u32;
    assert!(size > 0);
    let ubo_pool_size = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(size)
        .build();
    let sampler_pool_size = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(size)
        .build();

    let pool_sizes = [ubo_pool_size, sampler_pool_size];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(size);

    unsafe {
        context
            .device
            .create_descriptor_pool(&create_info, None)
            .expect("Unable to create descriptor pool")
    }
}

fn create_descriptor_set_layout(context: &VulkanContext) -> vk::DescriptorSetLayout {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();

    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();

    let bindings = [ubo_binding, sampler_binding];

    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        context
            .device
            .create_descriptor_set_layout(&create_info, None)
            .expect("Unable to create descriptor set layout")
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
            self.context
                .device
                .destroy_sampler(self.texture_sampler, None);
            self.context
                .device
                .destroy_image_view(self.texture_image_view, None);
            self.context.device.destroy_image(self.texture_image, None);
            self.context
                .device
                .free_memory(self.texture_image_memory, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

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
        *control_flow = ControlFlow::Poll;

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

fn create_vertex_buffer(
    context: &VulkanContext,
    vertices: &Vec<Vertex>,
) -> (vk::Buffer, vk::DeviceMemory) {
    context.create_buffer_from_data(vk::BufferUsageFlags::VERTEX_BUFFER, vertices)
}

fn create_index_buffer(
    context: &VulkanContext,
    indices: &Vec<u16>,
) -> (vk::Buffer, vk::DeviceMemory) {
    context.create_buffer_from_data(vk::BufferUsageFlags::INDEX_BUFFER, indices)
}

fn create_uniform_buffers(
    context: &VulkanContext,
    swap_chain: &SwapChain,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let size = size_of::<UniformBufferObject>() as u64;
    let usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
    let properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buffer_count = swap_chain.images.len();

    (0..buffer_count)
        .map(|_| context.create_buffer(size, usage, properties))
        .unzip()
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
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    index_count: usize,
    descriptor_sets: &Vec<vk::DescriptorSet>,
) -> Vec<vk::CommandBuffer> {
    let device = &context.device;
    let command_pool = &context.command_pool;
    let command_buffer_count = swap_chain_framebuffers.len() as u32;
    assert!(command_buffer_count > 0);

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(command_buffer_count);

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .expect("Unable to allocate frame_buffers")
    };

    for (command_buffer, framebuffer, descriptor_set) in
        izip!(&command_buffers, swap_chain_framebuffers, descriptor_sets)
    {
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
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[*descriptor_set],
                &[],
            );
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
    descriptor_set_layout: &[vk::DescriptorSetLayout],
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

    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layout);
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
