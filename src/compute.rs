use std::sync::Arc;
use smallvec::smallvec;
use ash::vk;
use vulkano::{
    VulkanObject, command_buffer::{CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, RecordingCommandBuffer}, descriptor_set::{
        WriteDescriptorSet, sys::RawDescriptorSet
    }, image::{ImageAspects, ImageLayout, ImageSubresourceRange, view::ImageView}, pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo
    }, sync::{AccessFlags, DependencyInfo, ImageMemoryBarrier, PipelineStages, QueueFamilyOwnershipTransfer, fence::{Fence, FenceCreateInfo}}
};

use crate::context::VulkanData;
use crate::tile::{TileCoord, TILE_SIZE};

mod mandelbrot_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 460

layout(local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0) uniform writeonly image2D atlas;

layout(push_constant) uniform PushConstants {
    double origin_x;
    double origin_y;
    double pixel_scale;
    uint max_iterations;
    uint padding;
};

void main() {
    ivec2 local_pos = ivec2(gl_GlobalInvocationID.xy);

    ivec2 atlas_pos = local_pos;

    double cr = origin_x + double(local_pos.x) * pixel_scale;
    double ci = origin_y + double(local_pos.y) * pixel_scale;

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

    imageStore(atlas, atlas_pos, vec4(float(iter) / float(max_iterations), 0.0, 0.0, 0.0));
}
"
    }
}

pub struct ComputeEngine {
    vk: Arc<VulkanData>,
    pipeline: Arc<ComputePipeline>,
    pub max_iterations: u32,
    fence: Fence,
    //semaphore: Semaphore,
}

impl ComputeEngine {
    pub fn new(vk: Arc<VulkanData>, max_iterations: u32) -> Self {
        let dev = vk.device.clone();
        let shader = mandelbrot_shader::load(dev.clone())
            .expect("failed to load mandelbrot compute shader");

        let entry = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry);
        let layout = PipelineLayout::new(
            dev.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(dev.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = ComputePipeline::new(
            dev.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        ComputeEngine{vk, pipeline, max_iterations,
            fence: Fence::new(dev.clone(), FenceCreateInfo::default()).unwrap(),
            //semaphore: Semaphore::new(dev, SemaphoreCreateInfo{semaphore_type: SemaphoreType::Timeline,..Default::default() }).unwrap(),
        }
    }

    /// Record compute dispatches for the given tile coords into a command buffer.
    pub fn compute_tile(
        &self,
        iv: Arc<ImageView>,
        coord: TileCoord,
    ) {
        let set = RawDescriptorSet::new(
            self.vk.descriptor_set_allocator.clone(),
            &self.pipeline.layout().set_layouts()[0],
            1
        ).unwrap();

        let (ox, oy) = coord.origin();
        let push = mandelbrot_shader::PushConstants {
            origin_x: ox,
            origin_y: oy,
            pixel_scale: coord.pixel_scale(),
            max_iterations: self.max_iterations,
            padding: 0,
        };

        let mut builder = RecordingCommandBuffer::new(
            self.vk.command_buffer_allocator.clone(),
            self.vk.compute_queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        ).unwrap();
        let bar1 = ImageMemoryBarrier {
            src_stages: PipelineStages::TOP_OF_PIPE,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::empty(),
            dst_access: AccessFlags::SHADER_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::General,
            subresource_range: ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1},
            ..ImageMemoryBarrier::image(iv.image().clone())
        };
        let queue_transfer = self.vk.compute_queue.queue_family_index() != self.vk.graphics_queue.queue_family_index();
        let bar2 = if queue_transfer {
            ImageMemoryBarrier {
                src_stages: PipelineStages::COMPUTE_SHADER,
                dst_stages: PipelineStages::BOTTOM_OF_PIPE,
                src_access: AccessFlags::SHADER_WRITE,
                dst_access: AccessFlags::empty(),
                old_layout: ImageLayout::General,
                new_layout: ImageLayout::ShaderReadOnlyOptimal,
                queue_family_ownership_transfer: Some(QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index: self.vk.compute_queue.queue_family_index(), dst_index: self.vk.graphics_queue.queue_family_index()}),
                subresource_range: ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1},
                ..ImageMemoryBarrier::image(iv.image().clone())
            }
        } else {
            ImageMemoryBarrier {
                src_stages: PipelineStages::COMPUTE_SHADER,
                dst_stages: PipelineStages::FRAGMENT_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_access: AccessFlags::SHADER_READ,
                old_layout: ImageLayout::General,
                new_layout: ImageLayout::ShaderReadOnlyOptimal,
                subresource_range: ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1},
                ..ImageMemoryBarrier::image(iv.image().clone())
            }
        };
        let cb = unsafe {
            set.update(&[WriteDescriptorSet::image_view(0, iv.clone())], &[])
                .expect("failed to update descriptor set");
            builder
            .bind_pipeline_compute(&self.pipeline)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                &mut self.pipeline.layout(),
                0,&[&set],&[]
            )
            .unwrap()
            .push_constants(&self.pipeline.layout(), 0, &push)
            .unwrap()
            .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar1],..Default::default()})
            .unwrap()
            .dispatch([TILE_SIZE / 32, TILE_SIZE / 32, 1])
            .unwrap()
            .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar2],..Default::default()})
            .unwrap();
            builder.end()
        }.expect("failed to build compute command buffer");
        let cb_arr = [cb.handle()];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cb_arr);
        self.vk.compute_queue.with(|_| unsafe {
            let result = (self.vk.device.fns().v1_0.queue_submit)(self.vk.compute_queue.handle(), 1, &submit_info, self.fence.handle());
            if result != vk::Result::SUCCESS {
                panic!("failed to submit compute command buffer");
            }
        });
        self.fence.wait(None).expect("Vulkan fence wait failed");
        unsafe {
            self.fence.reset().expect("Vulkan fence reset failed");
        }
        if queue_transfer {
            let mut builder = RecordingCommandBuffer::new(
                self.vk.command_buffer_allocator.clone(),
                self.vk.graphics_queue.queue_family_index(),
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
                new_layout: ImageLayout::ShaderReadOnlyOptimal,
                queue_family_ownership_transfer: Some(QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index: self.vk.compute_queue.queue_family_index(), dst_index: self.vk.graphics_queue.queue_family_index()}),
                subresource_range: ImageSubresourceRange{aspects: ImageAspects::COLOR, mip_levels: 0..1, array_layers: 0..1},
                ..ImageMemoryBarrier::image(iv.image().clone())
            };
            let cb = unsafe {
                builder
                .pipeline_barrier(&DependencyInfo{image_memory_barriers: smallvec![bar],..Default::default()})
                .unwrap();
                builder.end()
            }.expect("failed to build queue transfer command buffer");
            let cb_arr = [cb.handle()];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_arr);
            self.vk.graphics_queue.with(|_| unsafe {
                let result = (self.vk.device.fns().v1_0.queue_submit)(self.vk.graphics_queue.handle(), 1, &submit_info, self.fence.handle());
                if result != vk::Result::SUCCESS {
                    panic!("failed to submit queue transfer command buffer");
                }
            });
            self.fence.wait(None).expect("Vulkan fence wait failed");
            unsafe {
                self.fence.reset().expect("Vulkan fence reset failed");
            }
        }
    }
}
