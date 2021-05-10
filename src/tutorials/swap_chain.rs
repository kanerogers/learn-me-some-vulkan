use std::cmp;

use crate::QueueFamilyIndices;
use ash::{extensions::khr, version::DeviceV1_0, vk, Entry, Instance};
use winit::window::Window;

use crate::vulkan_context::VulkanContext;

pub struct SwapChain {
    pub loader: khr::Swapchain,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
}

impl SwapChain {
    pub fn new(context: &VulkanContext, window: &Window) -> Self {
        let (loader, swap_chain, format, extent) = create_swap_chain(&context, window);
        let mut images = get_swap_chain_images(&context, swap_chain);
        let image_views = create_image_views(&mut images, format, &context);

        Self {
            loader,
            handle: swap_chain,
            images,
            format,
            extent,
            image_views,
        }
    }
    // Frame Buffers
    pub fn create_framebuffers(
        &mut self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
    ) -> Vec<vk::Framebuffer> {
        self.image_views
            .iter()
            .map(|v| {
                let attachments = [*v]; //.. really?
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(self.extent.width)
                    .height(self.extent.height)
                    .layers(1);

                unsafe {
                    context
                        .device
                        .create_framebuffer(&create_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>()
    }
}

// Swap Chain
fn create_swap_chain(
    context: &VulkanContext,
    window: &Window,
) -> (khr::Swapchain, vk::SwapchainKHR, vk::Format, vk::Extent2D) {
    let instance = &context.instance;
    let entry = &context.entry;
    let device = &context.device;
    let physical_device = context.physical_device;
    let surface = context.surface;

    let swap_chain_support = SwapChainSupportDetails::query_swap_chain_support(
        entry,
        instance,
        physical_device,
        surface,
    );

    let surface_format = choose_swap_surface_format(swap_chain_support.surface_formats);
    let present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
    let extent = choose_swap_extent(swap_chain_support.capabilities, window);

    let image_count = swap_chain_support.capabilities.min_image_count + 1;
    let image_count = if swap_chain_support.capabilities.max_image_count > 0
        && image_count > swap_chain_support.capabilities.max_image_count
    {
        swap_chain_support.capabilities.max_image_count
    } else {
        image_count
    };

    let indices =
        QueueFamilyIndices::find_queue_families(instance, physical_device, entry, surface);
    let indices_are_same = indices.are_same();
    let image_sharing_mode = if indices_are_same {
        vk::SharingMode::EXCLUSIVE
    } else {
        vk::SharingMode::CONCURRENT
    };
    let queue_family_indices = if indices_are_same {
        Vec::new()
    } else {
        vec![
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ]
    };

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
    let swapchain = unsafe { swapchain_ext.create_swapchain(&create_info, None) }
        .expect("Unable to create Swapchain");
    (swapchain_ext, swapchain, surface_format.format, extent)
}

fn choose_swap_surface_format(formats: Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    for available_format in &formats {
        if available_format.format == vk::Format::B8G8R8A8_SRGB
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return *available_format;
        }
    }

    return *formats.first().unwrap();
}

fn choose_swap_present_mode(
    available_present_modes: Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
    for available_present_mode in &available_present_modes {
        if available_present_mode == &vk::PresentModeKHR::MAILBOX {
            return *available_present_mode;
        }
    }
    return *available_present_modes.first().unwrap();
}

fn choose_swap_extent(capabilities: vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }
    let physical_size = window.inner_size();
    let width = cmp::max(
        capabilities.min_image_extent.width,
        cmp::min(capabilities.max_image_extent.width, physical_size.width),
    );
    let height = cmp::max(
        capabilities.max_image_extent.height,
        cmp::min(capabilities.min_image_extent.height, physical_size.height),
    );

    vk::Extent2D { width, height }
}

fn get_swap_chain_images(context: &VulkanContext, swap_chain: vk::SwapchainKHR) -> Vec<vk::Image> {
    let swap_chain_ext = khr::Swapchain::new(&context.instance, &context.device);
    unsafe {
        swap_chain_ext
            .get_swapchain_images(swap_chain)
            .expect("Unable to get swapchain images")
    }
}

fn create_image_views(
    swap_chain_images: &mut Vec<vk::Image>,
    format: vk::Format,
    context: &VulkanContext,
) -> Vec<vk::ImageView> {
    swap_chain_images
        .iter()
        .map(|image| context.create_image_view(*image, format, vk::ImageAspectFlags::COLOR))
        .collect::<Vec<_>>()
}

pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    pub fn query_swap_chain_support(
        entry: &Entry,
        instance: &Instance,
        device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> SwapChainSupportDetails {
        let surface_ext = khr::Surface::new(entry, instance);
        let capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(device, surface)
                .expect("unable to get capabilities")
        };
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(device, surface)
                .expect("unable to get surface formats")
        };
        let present_modes = unsafe {
            surface_ext
                .get_physical_device_surface_present_modes(device, surface)
                .expect("unable to get present modes")
        };

        SwapChainSupportDetails {
            capabilities,
            surface_formats,
            present_modes,
        }
    }
}
