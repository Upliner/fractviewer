use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
};

use crate::tile::{QuadTree, TileCoord, TILE_SIZE};

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
    int atlas_offset_x;
    int atlas_offset_y;
    uint max_iterations;
    uint padding;
};

void main() {
    ivec2 local_pos = ivec2(gl_GlobalInvocationID.xy);

    ivec2 atlas_pos = local_pos + ivec2(atlas_offset_x, atlas_offset_y);

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
    pipeline: Arc<ComputePipeline>,
    pub max_iterations: u32,
}

impl ComputeEngine {
    pub fn new(device: &Arc<Device>, max_iterations: u32) -> Self {
        let shader = mandelbrot_shader::load(device.clone())
            .expect("failed to load mandelbrot compute shader");

        let entry = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        ComputeEngine {
            pipeline,
            max_iterations,
        }
    }

    /// Record compute dispatches for the given tile coords into a command buffer.
    pub fn dispatch_tiles(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        quadtree: &mut QuadTree,
        coords: &[TileCoord],
    ) {
        for &coord in coords {
            let slot = quadtree.insert(coord);
            let atlas = &quadtree.pool.atlases[slot.atlas_index];

            let set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                self.pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .unwrap()
                    .clone(),
                [WriteDescriptorSet::image_view(0, atlas.image_view.clone())],
                [],
            )
            .unwrap();

            let (ox, oy) = coord.origin();
            let (atlas_ox, atlas_oy) = slot.pixel_offset();

            let push = mandelbrot_shader::PushConstants {
                origin_x: ox,
                origin_y: oy,
                pixel_scale: coord.pixel_scale(),
                atlas_offset_x: atlas_ox as i32,
                atlas_offset_y: atlas_oy as i32,
                max_iterations: self.max_iterations,
                padding: 0,
            };

            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .push_constants(self.pipeline.layout().clone(), 0, push)
                .unwrap();
            unsafe {
                builder
                    .dispatch([TILE_SIZE / 32, TILE_SIZE / 32, 1])
                    .unwrap();
            }
        }
    }

}
