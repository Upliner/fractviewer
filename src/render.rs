use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    image::SampleCount,
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, Subpass},
};

use crate::tile::{TILE_SIZE, TileCoord};

#[derive(vulkano::buffer::BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct TileVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
}

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 460

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 f_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    f_uv = uv;
}
"
    }
}

mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 460

#define M_PI 3.1415926535897932384626433832795

layout(location = 0) in vec2 f_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tile;

void main() {
    float t = texture(tile, f_uv).r;
    vec3 col = vec3(
        0.5 + 0.5 * sin(8 * t * M_PI * 3.0),
        0.5 + 0.5 * sin(8 * t * M_PI * 5.0 + 4.188),
        0.5 + 0.5 * sin(8 * t * M_PI * 7.0 + 2.094));

    if (t > 63.0/64.0) {
        col *= t*-64+64;
    }
    out_color = vec4(col, 1.0);
}
"
    }
}

pub struct Renderer {
    pipeline: Arc<GraphicsPipeline>,
    tile_sampler: Arc<Sampler>,
    sample_count: SampleCount,
}

impl Renderer {
    pub fn new(
        device: &Arc<Device>,
        subpass: Subpass,
        sample_count: SampleCount,
    ) -> Self {
        let vs = vertex_shader::load(device.clone()).unwrap();
        let fs = fragment_shader::load(device.clone()).unwrap();

        let vs_entry = vs.entry_point("main").unwrap();
        let fs_entry = fs.entry_point("main").unwrap();

        let vertex_input_state = TileVertex::per_vertex().definition(&vs_entry).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        Renderer {
            pipeline: GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState {
                        rasterization_samples: sample_count,
                        sample_shading: Some(1.0),
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                }).expect("failed to create graphics pipeline"),
            tile_sampler: Sampler::new(
                device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    ..Default::default()
                }).expect("failed to create tile sampler"),
            sample_count,
        }
    }

    /// Record render commands for visible tiles.
    pub fn render(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
        viewport: Viewport,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        visible_tiles: &[(TileCoord, Arc<ImageView>)],
        vp_left: f64,
        vp_right: f64,
        vp_bottom: f64,
        vp_top: f64,
    ) {
        let clear_values = if u32::from(self.sample_count) > 1 {
            vec![Some([0.0, 0.0, 0.0, 1.0].into()), None]
        } else {
            vec![Some([0.0, 0.0, 0.0, 1.0].into())]
        };
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();

        let vp_w = vp_right - vp_left;
        let vp_h = vp_top - vp_bottom;

        let uv1 = 0.5 / TILE_SIZE as f32;
        let uv2 = 1.0 - uv1;

        for (coord, tile) in visible_tiles {

            let (tile_ox, tile_oy) = coord.origin();
            let tile_ext = coord.tile_extent();

            // Screen-space NDC for this tile
            let sx0 = ((tile_ox - vp_left) / vp_w * 2.0 - 1.0) as f32;
            let sy0 = ((tile_oy - vp_bottom) / vp_h * 2.0 - 1.0) as f32;
            let sx1 = ((tile_ox + tile_ext - vp_left) / vp_w * 2.0 - 1.0) as f32;
            let sy1 = ((tile_oy + tile_ext - vp_bottom) / vp_h * 2.0 - 1.0) as f32;

            let vertices = [
                TileVertex { position: [sx0, sy0], uv: [uv1, uv1] },
                TileVertex { position: [sx1, sy0], uv: [uv2, uv1] },
                TileVertex { position: [sx0, sy1], uv: [uv1, uv2] },
                TileVertex { position: [sx1, sy0], uv: [uv2, uv1] },
                TileVertex { position: [sx1, sy1], uv: [uv2, uv2] },
                TileVertex { position: [sx0, sy1], uv: [uv1, uv2] },
            ];

            let vertex_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vertices,
            )
            .unwrap();

            let set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                self.pipeline.layout().set_layouts().get(0).unwrap().clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        tile.clone(),
                        self.tile_sampler.clone(),
                    )
                ],
                [],
            )
            .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer)
                .unwrap();
            unsafe {
                builder.draw(6, 1, 0, 0).unwrap();
            }
        }

        builder.end_render_pass(SubpassEndInfo::default()).unwrap();
    }
}
