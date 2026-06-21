use std::{cell::Cell, cmp::min, sync::Arc};
use vulkano::{
    Validated, VulkanError, VulkanLibrary, command_buffer::allocator::StandardCommandBufferAllocator, descriptor_set::allocator::StandardDescriptorSetAllocator, device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFamilyProperties, QueueFlags, physical::PhysicalDeviceType
    }, format::Format, image::{Image, ImageCreateInfo, ImageFormatInfo, ImageLayout, ImageTiling, ImageType, ImageUsage, SampleCount, view::ImageView}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass, SubpassDescription}, swapchain::{
        self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    }, sync::{self, GpuFuture}
};
use winit::window::Window;

pub struct VulkanData {
    pub device: Arc<Device>,
    pub graphics_queue: Arc<Queue>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}
impl VulkanData {
    pub fn new(device: Arc<Device>, graphics_queue: Arc<Queue>) -> Self {
        VulkanData {
            device: device.clone(),
            graphics_queue,
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default())),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(device, Default::default())),
        }
    }
}
pub struct VulkanContext {
    pub data: Arc<VulkanData>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub swapchain_images: Vec<Arc<Image>>,
    pub compute_queues: Vec<Arc<Queue>>,
    pub sample_count: SampleCount,
    pub msaa_images: Vec<Arc<Image>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
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
                enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_string()],
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

        const TARGET_COMPUTE_CNT : usize = 2;
        const COMPUTE_PRIO: f32 = 0.5 / TARGET_COMPUTE_CNT as f32;
        let (physical_device, queue_create_infos) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter(|p| {
                let features = p.supported_features();
                features.shader_float64 && features.shader_storage_image_extended_formats
            })
            .filter_map(|p| {
                let graphics_only_family = p.queue_family_properties().iter().enumerate()
                    .find(|(_, q)|
                        q.queue_flags.contains(QueueFlags::GRAPHICS) &&
                        !q.queue_flags.contains(QueueFlags::COMPUTE) && q.queue_count > 0)
                    .map(|(i, _)| i as u32);
                let compute_families = p.queue_family_properties()
                .iter().enumerate().map(|(k, v)| (k as u32, v))
                    .filter(|(_, q)| q.queue_flags.contains(QueueFlags::COMPUTE) && q.queue_count > 0)
                    .collect::<Vec<_>>();
                if compute_families.is_empty() {
                    return None;
                }
                let mut queues = Vec::<QueueCreateInfo>::with_capacity(TARGET_COMPUTE_CNT + 1);
                let compute_cnt = Cell::new(0);
                let mut add_compute_family = |(i, q): (u32, &QueueFamilyProperties)| {
                    let cnt = min(q.queue_count as usize, TARGET_COMPUTE_CNT - compute_cnt.get());
                    queues.push(QueueCreateInfo {
                        queue_family_index: i,
                        queues: std::iter::repeat_n(COMPUTE_PRIO, cnt).collect(),
                        ..Default::default()
                    });
                    compute_cnt.update(|c| c + cnt);
                };
                if let Some(graphics_only_family) = graphics_only_family {
                    let mut family_iter = compute_families.into_iter();
                    while compute_cnt.get() < TARGET_COMPUTE_CNT && let Some(cf) = family_iter.next() {
                        add_compute_family(cf);
                    }
                    queues.push(QueueCreateInfo {
                        queue_family_index: graphics_only_family,
                        queues: vec![1.0],
                        ..Default::default()
                    });
                } else {
                    let graphics_family = *compute_families.iter()
                        .find(|(_, q)| q.queue_flags.contains(QueueFlags::GRAPHICS))?;
                    let mut separate_compute_families =
                        compute_families.into_iter().filter(|(i, _)| *i != graphics_family.0);
                    while compute_cnt.get() < TARGET_COMPUTE_CNT &&
                            let Some(cf) = separate_compute_families.next() {
                        add_compute_family(cf);
                    }
                    let mut gqueues = Vec::<f32>::with_capacity(TARGET_COMPUTE_CNT + 1);
                    let cnt = min((TARGET_COMPUTE_CNT-compute_cnt.get()) as i64, graphics_family.1.queue_count as i64-1);
                    if cnt > 0 {
                        gqueues.extend(std::iter::repeat_n(COMPUTE_PRIO, cnt as usize));
                    }
                    gqueues.push(1.0);
                    queues.push(QueueCreateInfo {
                        queue_family_index: graphics_family.0,
                        queues: gqueues,
                        ..Default::default()
                    });
                }
                Some((p, queues))
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

        let (device, queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    shader_float64: true,
                    shader_storage_image_extended_formats: true,
                    sample_rate_shading: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let (swapchain, swapchain_images) =
            Self::create_swapchain(&device, &surface, &window, None);

        let sample_count = Self::max_sample_count(&device, swapchain.image_format());
        eprintln!("Using MSAA sample count: {}", u32::from(sample_count));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let msaa_images = if u32::from(sample_count) > 1 {
            Self::create_msaa_images(
                &memory_allocator,
                swapchain.image_extent(),
                swapchain.image_format(),
                sample_count,
                swapchain_images.len(),
            )
        } else {
            Vec::new()
        };

        let render_pass = Self::create_render_pass(&device, &swapchain, sample_count);

        let framebuffers = Self::create_framebuffers(
            &msaa_images,
            &swapchain_images,
            &render_pass,
            sample_count,
        );
        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        let mut queues = queues.collect::<Vec<_>>();
        let graphics_queue = queues.pop().expect("Vulkan queue was not created as expected");
        if queues.is_empty() {
            queues.push(graphics_queue.clone());
        }
        VulkanContext {
            data:Arc::new(VulkanData::new(device.clone(), graphics_queue)),
            surface,
            swapchain,
            swapchain_images,
            compute_queues: queues,
            sample_count,
            msaa_images,
            render_pass,
            framebuffers,
            memory_allocator,
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
                    /*Format::B8G8R8A8_SRGB | Format::R8G8B8A8_SRGB |*/ Format::B8G8R8A8_UNORM
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

    fn max_sample_count(device: &Arc<Device>, format: Format) -> SampleCount {
        device
            .physical_device()
            .image_format_properties(ImageFormatInfo {
                format,
                image_type: ImageType::Dim2d,
                tiling: ImageTiling::Optimal,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            })
            .unwrap()
            .map(|properties| properties.sample_counts.max_count())
            .unwrap_or(SampleCount::Sample1)
    }

    fn create_msaa_images(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
        format: Format,
        sample_count: SampleCount,
        count: usize,
    ) -> Vec<Arc<Image>> {
        let extent = [extent[0], extent[1], 1];
        (0..count)
            .map(|_| {
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent,
                        samples: sample_count,
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    fn create_render_pass(
        device: &Arc<Device>,
        swapchain: &Arc<Swapchain>,
        sample_count: SampleCount,
    ) -> Arc<RenderPass> {
        let format = swapchain.image_format();
        if u32::from(sample_count) > 1 {
            RenderPass::new(
                device.clone(),
                RenderPassCreateInfo {
                    attachments: vec![
                        AttachmentDescription {
                            format,
                            samples: sample_count,
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::DontCare,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        },
                        AttachmentDescription {
                            format,
                            samples: SampleCount::Sample1,
                            load_op: AttachmentLoadOp::DontCare,
                            store_op: AttachmentStoreOp::Store,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::PresentSrc,
                            ..Default::default()
                        },
                    ],
                    subpasses: vec![SubpassDescription {
                        color_attachments: vec![Some(AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        })],
                        color_resolve_attachments: vec![Some(AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        })],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )
            .unwrap()
        } else {
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        format: format,
                        samples: 1,
                        load_op: Clear,
                        store_op: Store,
                        final_layout: ImageLayout::PresentSrc,
                    },
                },
                pass: {
                    color: [color],
                    depth_stencil: {},
                },
            )
            .unwrap()
        }
    }

    fn create_framebuffers(
        msaa_images: &[Arc<Image>],
        swapchain_images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
        sample_count: SampleCount,
    ) -> Vec<Arc<Framebuffer>> {
        if u32::from(sample_count) > 1 {
            msaa_images
                .iter()
                .zip(swapchain_images.iter())
                .map(|(msaa_image, swapchain_image)| {
                    let msaa_view = ImageView::new_default(msaa_image.clone()).unwrap();
                    let swapchain_view = ImageView::new_default(swapchain_image.clone()).unwrap();
                    Framebuffer::new(
                        render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![msaa_view, swapchain_view],
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect()
        } else {
            swapchain_images
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
            &self.data.device,
            &self.surface,
            window,
            Some(&self.swapchain),
        );
        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;
        if u32::from(self.sample_count) > 1 {
            self.msaa_images = Self::create_msaa_images(
                &self.memory_allocator,
                self.swapchain.image_extent(),
                self.swapchain.image_format(),
                self.sample_count,
                self.swapchain_images.len(),
            );
        }
        self.framebuffers = Self::create_framebuffers(
            &self.msaa_images,
            &self.swapchain_images,
            &self.render_pass,
            self.sample_count,
        );
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
                self.data.graphics_queue.clone(),
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
                    Some(sync::now(self.data.device.clone()).boxed());
            }
            Err(e) => {
                eprintln!("failed to flush future: {e}");
                self.previous_frame_end =
                    Some(sync::now(self.data.device.clone()).boxed());
            }
        }
    }

    pub fn subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }
}
