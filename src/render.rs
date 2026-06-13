use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
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

use crate::tile::{QuadTree, TileCoord, TileSlot};

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

layout(location = 0) in vec2 f_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform usampler2D tile_atlas;
layout(set = 0, binding = 1) uniform sampler1D palette;

layout(push_constant) uniform PushConstants {
    uint max_iterations;
};

void main() {
    uint iter = texture(tile_atlas, f_uv).r;
    if (iter >= max_iterations) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float t = float(iter) / float(max_iterations);
        out_color = texture(palette, t);
    }
}
"
    }
}

pub struct Renderer {
    pipeline: Arc<GraphicsPipeline>,
    tile_sampler: Arc<Sampler>,
    palette_sampler: Arc<Sampler>,
    pub palette_image_view: Arc<ImageView>,
    pub max_iterations: u32,
}

impl Renderer {
    pub fn new(
        device: &Arc<Device>,
        subpass: Subpass,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        max_iterations: u32,
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

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .expect("failed to create graphics pipeline");

        let tile_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let palette_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let palette_image_view = Self::create_palette(memory_allocator);

        Renderer {
            pipeline,
            tile_sampler,
            palette_sampler,
            palette_image_view,
            max_iterations,
        }
    }

    fn create_palette(memory_allocator: &Arc<StandardMemoryAllocator>) -> Arc<ImageView> {
        let palette_size = 1024u32;
        let mut palette_data: Vec<u8> = Vec::with_capacity(palette_size as usize * 4);

        for i in 0..palette_size {
            let t = i as f32 / palette_size as f32;
            let r = (0.5 + 0.5 * (3.0 + t * std::f32::consts::TAU * 3.0).cos()) * 255.0;
            let g = (0.5 + 0.5 * (3.0 + t * std::f32::consts::TAU * 5.0 + 2.094).cos()) * 255.0;
            let b = (0.5 + 0.5 * (3.0 + t * std::f32::consts::TAU * 7.0 + 4.188).cos()) * 255.0;
            palette_data.push(r as u8);
            palette_data.push(g as u8);
            palette_data.push(b as u8);
            palette_data.push(255u8);
        }

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim1d,
                format: Format::R8G8B8A8_UNORM,
                extent: [palette_size, 1, 1],
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create palette image");

        ImageView::new_default(image).unwrap()
    }

    /// Build a command buffer that uploads palette data to the palette image.
    /// Must be submitted before first render.
    pub fn upload_palette(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) {
        let palette_size = 1024u32;
        let mut palette_data: Vec<u8> = Vec::with_capacity(palette_size as usize * 4);

        for i in 0..palette_size {
            let t = i as f32 / palette_size as f32;
            let r = (0.5 + 0.5 * (3.0 + t * std::f32::consts::TAU * 3.0).cos()) * 255.0;
            let g = (0.5 + 0.5 * (3.0 + t * std::f32::consts::TAU * 5.0 + 2.094).cos()) * 255.0;
            let b = (0.5 + 0.5 * (3.0 + t * std::f32::consts::TAU * 7.0 + 4.188).cos()) * 255.0;
            palette_data.push(r as u8);
            palette_data.push(g as u8);
            palette_data.push(b as u8);
            palette_data.push(255u8);
        }

        let staging_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            palette_data,
        )
        .unwrap();

        let image = self.palette_image_view.image().clone();

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer,
                image,
            ))
            .unwrap();
    }

    /// Record render commands for visible tiles.
    pub fn render(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
        viewport: Viewport,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        quadtree: &QuadTree,
        visible_tiles: &[(TileCoord, TileSlot)],
        vp_left: f64,
        vp_right: f64,
        vp_bottom: f64,
        vp_top: f64,
    ) {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
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

        let push = fragment_shader::PushConstants {
            max_iterations: self.max_iterations,
        };

        builder
            .push_constants(self.pipeline.layout().clone(), 0, push)
            .unwrap();

        let vp_w = vp_right - vp_left;
        let vp_h = vp_top - vp_bottom;

        for &(coord, slot) in visible_tiles {
            let atlas = &quadtree.pool.atlases[slot.atlas_index];
            let uv = slot.uv_rect();
            let (tile_ox, tile_oy) = coord.origin();
            let tile_ext = coord.tile_extent();

            // Screen-space NDC for this tile
            let sx0 = ((tile_ox - vp_left) / vp_w * 2.0 - 1.0) as f32;
            let sy0 = ((tile_oy - vp_bottom) / vp_h * 2.0 - 1.0) as f32;
            let sx1 = ((tile_ox + tile_ext - vp_left) / vp_w * 2.0 - 1.0) as f32;
            let sy1 = ((tile_oy + tile_ext - vp_bottom) / vp_h * 2.0 - 1.0) as f32;

            let vertices = [
                TileVertex { position: [sx0, sy0], uv: [uv[0], uv[1]] },
                TileVertex { position: [sx1, sy0], uv: [uv[2], uv[1]] },
                TileVertex { position: [sx0, sy1], uv: [uv[0], uv[3]] },
                TileVertex { position: [sx1, sy0], uv: [uv[2], uv[1]] },
                TileVertex { position: [sx1, sy1], uv: [uv[2], uv[3]] },
                TileVertex { position: [sx0, sy1], uv: [uv[0], uv[3]] },
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
                        atlas.image_view.clone(),
                        self.tile_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        self.palette_image_view.clone(),
                        self.palette_sampler.clone(),
                    ),
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
