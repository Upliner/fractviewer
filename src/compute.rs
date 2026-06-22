use std::{array, sync::Arc};
use smallvec::smallvec;
use ash::vk::{self, PipelineStageFlags};
use vulkano::{
    VulkanObject, command_buffer::{CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, RecordingCommandBuffer}, descriptor_set::{
        WriteDescriptorSet, layout::DescriptorSetLayout, sys::RawDescriptorSet
    }, device::{Device, Queue}, format::Format, image::{ImageAspect, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage, sys::RawImage, view::ImageView}, memory::{DeviceMemory, MemoryAllocateInfo, MemoryMapInfo, MemoryPropertyFlags, ResourceMemory}, pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo, layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo}
    }, shader::ShaderModule, sync::{AccessFlags, DependencyInfo, ImageMemoryBarrier, PipelineStages, QueueFamilyOwnershipTransfer, fence::{Fence, FenceCreateInfo}, semaphore::Semaphore}
};

use crate::context::VulkanData;
use crate::tile::{TileCoord, TILE_SIZE};

mod mandelbrot_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 460

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform writeonly image2D tile;

layout(push_constant) uniform PushConstants {
    double origin_x;
    double origin_y;
    double pixel_scale;
    uint max_iterations;
    uint padding;
};

void main() {
    double cr = origin_x + double(gl_GlobalInvocationID.x) * pixel_scale;
    double ci = origin_y + double(gl_GlobalInvocationID.y) * pixel_scale;

    double zr = 0.0;
    double zi = 0.0;
    uint iter = 0;

    for (; iter < max_iterations; iter++) {
        double zr2 = zr * zr;
        double zi2 = zi * zi;
        if (zr2 + zi2 > 4.0) break;
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    imageStore(tile, ivec2(gl_GlobalInvocationID.xy), vec4(float(iter) / float(max_iterations), 0.0, 0.0, 0.0));
}
"
    }
}

mod tilemap_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 460

layout(local_size_x = 2, local_size_y = 2) in;

layout(set = 0, binding = 0, r16) uniform readonly image2D in_tile;
layout(set = 1, binding = 0, r32ui) uniform writeonly uimage2D map;

layout(push_constant) uniform PushConstants {
    int size;
};

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 base = pos*(size-1);
    for (int y = 0; y <= size; y++) {
        for (int x = 0; x <= size; x++) {
            if (imageLoad(in_tile, base+ivec2(x, y)).r < 1.0) {
                imageStore(map, pos, uvec4(0));
                return;
            }
        }
    }
    imageStore(map, pos, uvec4(1));
}
"
    }
}

struct MyPipeline {
    pipeline: Arc<ComputePipeline>,
    dset: RawDescriptorSet,
}
pub struct ComputeEngine {
    pub max_iterations: u32,
    vk: Arc<VulkanData>,
    q: Arc<Queue>,
    _iv: Arc<ImageView>,
    map_mem: *const u32,
    row_pitch: usize,
    tile_compute: MyPipeline,
    map_compute: MyPipeline,
    bar_map_in: ImageMemoryBarrier,
    semaphores: Option<[Semaphore; 2]>,
    fence: Fence,
}

fn find_memory_type(dev: &Device, memory_type_bits : u32) -> Option<u32> {
    let types = &dev.physical_device().memory_properties().memory_types;
    for (i, memory_type) in types.iter().enumerate() {
        if memory_type_bits & (1 << i) != 0 && memory_type.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_CACHED) {
            return Some(i as u32);
        }
    }
    for (i, memory_type) in types.iter().enumerate() {
        if memory_type_bits & (1 << i) != 0 && memory_type.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT) {
            return Some(i as u32);
        }
    }
    None
}

impl ComputeEngine {
    pub fn new(vk: Arc<VulkanData>, q: Arc<Queue>, max_iterations: u32) -> Self {
        let dev = vk.device.clone();
        let make_pipeline = |shader: Arc<ShaderModule>, mut set_layouts: Vec<Arc<DescriptorSetLayout>>| {
            let entry = shader.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(entry);
            let pdslci = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage]);
            let dset_layout = DescriptorSetLayout::new(
                dev.clone(), pdslci.set_layouts.last().unwrap().clone()).unwrap();
            set_layouts.push(dset_layout.clone());
            let layout = PipelineLayout::new(dev.clone(), PipelineLayoutCreateInfo {
                flags: pdslci.flags,
                set_layouts,
                push_constant_ranges: pdslci.push_constant_ranges,
                ..Default::default()
            }).unwrap();
            (MyPipeline {
                pipeline: ComputePipeline::new(
                    dev.clone(),
                    None,
                    ComputePipelineCreateInfo::stage_layout(stage, layout),
                )
                .expect("failed to create compute pipeline"),
                dset: RawDescriptorSet::new(
                    vk.descriptor_set_allocator.clone(),
                    &dset_layout,
                    1
                ).unwrap(),
            }, dset_layout)
        };

        let (tile_compute, dset_layout) = make_pipeline(mandelbrot_shader::load(dev.clone())
            .expect("failed to load mandelbrot compute shader"), vec![]);
        let (map_compute, _) = make_pipeline(tilemap_shader::load(dev.clone())
            .expect("failed to load tilemap compute shader"), vec![dset_layout]);
        let raw_image = RawImage::new(dev.clone(), ImageCreateInfo{
            image_type: ImageType::Dim2d,
            format: Format::R32_UINT,
            extent: [2, 2, 1],
            tiling: ImageTiling::Linear,
            usage: ImageUsage::STORAGE,
            ..Default::default()
        }).expect("failed to create map image");
        let req = raw_image.memory_requirements().first().unwrap();
        let sz = req.layout.size();
        let mut mem = DeviceMemory::allocate(dev.clone(), MemoryAllocateInfo{
            allocation_size: sz,
            memory_type_index: find_memory_type(dev.as_ref(), req.memory_type_bits).expect("no host visible memory type found"),
            ..Default::default()
        }).expect("failed to allocate map memory");
        let row_pitch = raw_image.subresource_layout(ImageAspect::Color, 0, 0)
            .expect("failed to get subresource layout").row_pitch;
        mem.map(MemoryMapInfo{size: row_pitch+8,..Default::default()}).expect("failed to map map memory");
        let map_mem = mem.mapping_state().expect("map memory is not mapped").ptr().as_ptr() as *const u32;
        let map_img = ImageView::new_default(Arc::new(raw_image.bind_memory(
            std::iter::once(unsafe { ResourceMemory::from_device_memory_unchecked(Arc::new(mem), 0, sz) })
        ).map_err(|e| e.0).expect("failed to bind map memory"))).expect("failed to create map image view");
        unsafe {
            map_compute.dset.update(&[WriteDescriptorSet::image_view(0, map_img.clone())], &[])
        }.expect("failed to update map descriptor set");

        ComputeEngine{
            fence: Fence::new(dev.clone(), FenceCreateInfo::default()).unwrap(),
            bar_map_in: ImageMemoryBarrier {
                src_stages: PipelineStages::TOP_OF_PIPE,
                dst_stages: PipelineStages::COMPUTE_SHADER,
                src_access: AccessFlags::empty(),
                dst_access: AccessFlags::SHADER_WRITE,
                old_layout: ImageLayout::Undefined,
                new_layout: ImageLayout::General,
                subresource_range: ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1},
                ..ImageMemoryBarrier::image(map_img.image().clone())
            },
            semaphores: if q.queue_family_index() == vk.graphics_queue.queue_family_index() {
                None
            } else {
                Some(array::from_fn(|_| Semaphore::from_pool(dev.clone()).unwrap()))
            },
            max_iterations, q, vk, map_mem, tile_compute, map_compute, _iv: map_img, row_pitch: (row_pitch/4) as usize,
        }
    }

    /// Record compute dispatches for the given tile coords into a command buffer.
    pub fn compute_tile(
        &mut self,
        iv: Arc<ImageView>,
        coord: TileCoord,
    ) -> [bool; 4] {
        let (ox, oy) = coord.origin();
        let push_tile = mandelbrot_shader::PushConstants {
            origin_x: ox,
            origin_y: oy,
            pixel_scale: coord.pixel_scale(),
            max_iterations: self.max_iterations,
            padding: 0,
        };
        let push_map = tilemap_shader::PushConstants {
            size: (TILE_SIZE / 2) as i32,
        };

        let qfi = self.q.queue_family_index();
        let gqfi: u32 = self.vk.graphics_queue.queue_family_index();

        let srr = ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1};
        let bar1 = ImageMemoryBarrier {
            src_stages: PipelineStages::TOP_OF_PIPE,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::empty(),
            dst_access: AccessFlags::SHADER_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::General,
            subresource_range: srr.clone(),
            ..ImageMemoryBarrier::image(iv.image().clone())
        };
        let mut submit_info = vk::SubmitInfo::default();
        let bar_tile_make_map = ImageMemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_access: AccessFlags::SHADER_READ,
            old_layout: ImageLayout::General,
            new_layout: ImageLayout::General,
            subresource_range: srr.clone(),
            ..ImageMemoryBarrier::image(iv.image().clone())
        };
        let bar_map_read = ImageMemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            dst_stages: PipelineStages::HOST,
            src_access: AccessFlags::SHADER_WRITE,
            dst_access: AccessFlags::HOST_READ,
            old_layout: ImageLayout::General,
            new_layout: ImageLayout::General,
            subresource_range: srr.clone(),
            ..ImageMemoryBarrier::image(iv.image().clone())
        };
        let bar_tile_out = if qfi != gqfi {
            ImageMemoryBarrier {
                src_stages: PipelineStages::COMPUTE_SHADER,
                dst_stages: PipelineStages::BOTTOM_OF_PIPE,
                src_access: AccessFlags::SHADER_READ,
                dst_access: AccessFlags::empty(),
                old_layout: ImageLayout::General,
                new_layout: ImageLayout::General,
                queue_family_ownership_transfer: Some(QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index: qfi, dst_index: gqfi}),
                subresource_range: srr.clone(),
                ..ImageMemoryBarrier::image(iv.image().clone())
            }
        } else {
            ImageMemoryBarrier {
                src_stages: PipelineStages::COMPUTE_SHADER,
                dst_stages: PipelineStages::FRAGMENT_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_access: AccessFlags::SHADER_READ,
                old_layout: ImageLayout::General,
                new_layout: ImageLayout::General,
                subresource_range: srr.clone(),
                ..ImageMemoryBarrier::image(iv.image().clone())
            }
        };

        let mut builder = RecordingCommandBuffer::new(
            self.vk.command_buffer_allocator.clone(), qfi,
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        ).unwrap();

        let tile_layout = self.tile_compute.pipeline.layout().as_ref();
        let map_layout = self.map_compute.pipeline.layout().as_ref();
        let cb = unsafe {
            self.tile_compute.dset.update(&[WriteDescriptorSet::image_view(0, iv.clone())], &[])
                .expect("failed to update descriptor set");
            builder
            .bind_pipeline_compute(&self.tile_compute.pipeline)
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, tile_layout, 0,
                &[&self.tile_compute.dset],&[])
            .unwrap()
            .push_constants(tile_layout, 0, &push_tile)
            .unwrap()
            .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar1],..Default::default()})
            .unwrap()
            .dispatch([TILE_SIZE / 16, TILE_SIZE / 16, 1])
            .unwrap()
            .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar_tile_make_map, self.bar_map_in.clone()],..Default::default()})
            .unwrap()
            .bind_pipeline_compute(&self.map_compute.pipeline)
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, map_layout, 0,
                &[&self.tile_compute.dset, &self.map_compute.dset],&[] )
            .unwrap()
            .push_constants(map_layout, 0, &push_map)
            .unwrap()
            .dispatch([1, 1, 1])
            .unwrap()
            .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar_map_read],..Default::default()})
            .unwrap()
            .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar_tile_out],..Default::default()})
            .unwrap();
            builder.end()
        }.expect("failed to build compute command buffer");
        let cb_arr = [cb.handle()];
        submit_info = submit_info.command_buffers(&cb_arr);
        let mut fence = self.fence.handle();
        let mut aq_semaphore = Option::<[ash::vk::Semaphore; 1]>::None;
        if let Some(semaphores) = &self.semaphores {
            submit_info = submit_info.signal_semaphores(aq_semaphore.insert([semaphores[1].handle()]));
            fence = ash::vk::Fence::null();
        }
        self.q.with(|_| unsafe {
            let result = (self.vk.device.fns().v1_0.queue_submit)(self.q.handle(), 1, &submit_info, fence);
            if result != vk::Result::SUCCESS {
                panic!("failed to submit compute command buffer");
            }
        });
        let mut aq_cb = Option::<CommandBuffer>::None;
        if let Some(semaphores) = &self.semaphores {
            let mut builder = RecordingCommandBuffer::new(
                self.vk.command_buffer_allocator.clone(),
                gqfi,
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            ).unwrap();
            let bar = ImageMemoryBarrier {
                src_stages: PipelineStages::TOP_OF_PIPE,
                dst_stages: PipelineStages::FRAGMENT_SHADER,
                src_access: AccessFlags::empty(),
                dst_access: AccessFlags::SHADER_READ,
                old_layout: ImageLayout::General,
                new_layout: ImageLayout::General,
                queue_family_ownership_transfer: Some(QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index: qfi, dst_index: gqfi}),
                subresource_range: ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1},
                ..ImageMemoryBarrier::image(iv.image().clone())
            };
            let cb = aq_cb.insert(unsafe {
                builder
                .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar],..Default::default()})
                .unwrap();
                builder.end()
            }.expect("failed to build queue transfer command buffer"));
            let cb_arr = [cb.handle()];
            let semaphores = [semaphores[1].handle()];
            let dst_stage_mask = [PipelineStageFlags::FRAGMENT_SHADER];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_arr)
                .wait_semaphores(&semaphores).wait_dst_stage_mask(&dst_stage_mask);
            self.vk.graphics_queue.with(|_| unsafe {
                let result = (self.vk.device.fns().v1_0.queue_submit)(self.vk.graphics_queue.handle(), 1, &submit_info, self.fence.handle());
                if result != vk::Result::SUCCESS {
                    panic!("failed to submit queue transfer command buffer");
                }
            });
        }
        self.fence.wait(None).expect("Vulkan fence wait failed");
        unsafe {
            self.fence.reset().expect("Vulkan fence reset failed");
        }
        self.bar_map_in.src_stages = PipelineStages::HOST;
        self.bar_map_in.src_access = AccessFlags::HOST_READ;
        let row1 = unsafe { std::ptr::read_volatile(self.map_mem as *const [u32;2]) };
        let row2= unsafe { std::ptr::read_volatile(self.map_mem.add(self.row_pitch) as *const [u32;2]) };
        [row1[0] > 0, row1[1] > 0, row2[0] > 0, row2[1] > 0]
    }
}
