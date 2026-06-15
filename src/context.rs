use std::sync::Arc;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::window::Window;

pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub swapchain_images: Vec<Arc<Image>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub recreate_swapchain: bool,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl VulkanContext {
    pub fn new(window: Arc<Window>) -> Self {
        let library = VulkanLibrary::new().expect("failed to load Vulkan library");

        let required_extensions = Surface::required_extensions(&window).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create Vulkan instance");

        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("failed to create surface");

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter(|p| {
                let features = p.supported_features();
                features.shader_float64 && features.shader_storage_image_extended_formats
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable GPU found (need shader_float64 + shader_storage_image_extended_formats)");

        eprintln!(
            "Using device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    shader_float64: true,
                    shader_storage_image_extended_formats: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        let (swapchain, swapchain_images) =
            Self::create_swapchain(&device, &surface, &window, None);

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let framebuffers = Self::create_framebuffers(&swapchain_images, &render_pass);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        VulkanContext {
            device,
            queue,
            surface,
            swapchain,
            swapchain_images,
            render_pass,
            framebuffers,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            recreate_swapchain: false,
            previous_frame_end,
        }
    }

    fn create_swapchain(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        window: &Window,
        old_swapchain: Option<&Arc<Swapchain>>,
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .unwrap();

        let image_format = device
            .physical_device()
            .surface_formats(surface, Default::default())
            .unwrap()
            .into_iter()
            .find(|(f, _)| {
                matches!(
                    f,
                    Format::B8G8R8A8_SRGB | Format::R8G8B8A8_SRGB | Format::B8G8R8A8_UNORM
                )
            })
            .unwrap_or_else(|| {
                device
                    .physical_device()
                    .surface_formats(surface, Default::default())
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap()
            })
            .0;

        if let Some(old) = old_swapchain {
            let create_info = SwapchainCreateInfo {
                image_extent: window.inner_size().into(),
                ..old.create_info().clone()
            };
            old.recreate(create_info)
                .expect("failed to recreate swapchain")
        } else {
            let create_info = SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            };
            Swapchain::new(device.clone(), surface.clone(), create_info)
                .expect("failed to create swapchain")
        }
    }

    fn create_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    pub fn recreate_swapchain_if_needed(&mut self, window: &Window) {
        if !self.recreate_swapchain {
            return;
        }
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        let (new_swapchain, new_images) = Self::create_swapchain(
            &self.device,
            &self.surface,
            window,
            Some(&self.swapchain),
        );
        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;
        self.framebuffers = Self::create_framebuffers(&self.swapchain_images, &self.render_pass);
        self.recreate_swapchain = false;
    }

    pub fn window_size(&self) -> [u32; 2] {
        self.swapchain.image_extent()
    }

    pub fn acquire_next_image(
        &mut self,
    ) -> Result<(u32, bool, swapchain::SwapchainAcquireFuture), ()> {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        match swapchain::acquire_next_image(self.swapchain.clone(), None)
            .map_err(Validated::unwrap)
        {
            Ok(r) => Ok(r),
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                Err(())
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        }
    }

    pub fn present(
        &mut self,
        future: Box<dyn GpuFuture>,
        image_index: u32,
    ) {
        let future = future
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                future.wait(None).unwrap();
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                eprintln!("Out of date. Recreating swapchain.");
                self.recreate_swapchain = true;
                self.previous_frame_end =
                    Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                eprintln!("failed to flush future: {e}");
                self.previous_frame_end =
                    Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

    pub fn subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }
}
