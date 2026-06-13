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

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r16ui) uniform writeonly uimage2D atlas;

layout(push_constant) uniform PushConstants {
    // Each f64 encoded as two uint32 (lo, hi) to avoid alignment issues
    uint origin_x_lo;
    uint origin_x_hi;
    uint origin_y_lo;
    uint origin_y_hi;
    uint scale_lo;
    uint scale_hi;
    int atlas_offset_x;
    int atlas_offset_y;
    uint max_iterations;
};

double decode_double(uint lo, uint hi) {
    return packDouble2x32(uvec2(lo, hi));
}

void main() {
    ivec2 local_pos = ivec2(gl_GlobalInvocationID.xy);
    if (local_pos.x >= 256 || local_pos.y >= 256) return;

    ivec2 atlas_pos = local_pos + ivec2(atlas_offset_x, atlas_offset_y);

    double tile_origin_x = decode_double(origin_x_lo, origin_x_hi);
    double tile_origin_y = decode_double(origin_y_lo, origin_y_hi);
    double pixel_scale = decode_double(scale_lo, scale_hi);

    double cr = tile_origin_x + double(local_pos.x) * pixel_scale;
    double ci = tile_origin_y + double(local_pos.y) * pixel_scale;

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

    imageStore(atlas, atlas_pos, uvec4(iter, 0, 0, 0));
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
            let ps = coord.pixel_scale();
            let (atlas_ox, atlas_oy) = slot.pixel_offset();

            let ox_bits = ox.to_bits();
            let oy_bits = oy.to_bits();
            let ps_bits = ps.to_bits();

            let push = mandelbrot_shader::PushConstants {
                origin_x_lo: ox_bits as u32,
                origin_x_hi: (ox_bits >> 32) as u32,
                origin_y_lo: oy_bits as u32,
                origin_y_hi: (oy_bits >> 32) as u32,
                scale_lo: ps_bits as u32,
                scale_hi: (ps_bits >> 32) as u32,
                atlas_offset_x: atlas_ox as i32,
                atlas_offset_y: atlas_oy as i32,
                max_iterations: self.max_iterations,
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
                    .dispatch([TILE_SIZE / 16, TILE_SIZE / 16, 1])
                    .unwrap();
            }
        }
    }

}
