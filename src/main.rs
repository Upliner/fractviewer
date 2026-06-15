mod camera;
mod compute;
mod context;
mod render;
mod tile;

use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    pipeline::graphics::viewport::Viewport,
    sync::{GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use camera::Camera;
use compute::ComputeEngine;
use context::VulkanContext;
use render::Renderer;
use tile::QuadTree;

const MAX_ITERATIONS: u32 = 8192;
const MAX_COMPUTE_PER_FRAME: usize = 1;

struct App {
    window: Option<Arc<Window>>,
    ctx: Option<VulkanContext>,
    camera: Camera,
    compute: Option<ComputeEngine>,
    renderer: Option<Renderer>,
    quadtree: Option<QuadTree>,
    left_mouse_down: bool,
    right_mouse_down: bool,
    cursor_pos: (f64, f64),
}

impl App {
    fn new() -> Self {
        App {
            window: None,
            ctx: None,
            camera: Camera::new(),
            compute: None,
            renderer: None,
            quadtree: None,
            left_mouse_down: false,
            right_mouse_down: false,
            cursor_pos: (0.0, 0.0),
        }
    }

    fn redraw(&mut self) {
        let window = self.window.as_ref().unwrap();
        let ctx = self.ctx.as_mut().unwrap();
        let compute = self.compute.as_ref().unwrap();
        let renderer = self.renderer.as_ref().unwrap();
        let quadtree = self.quadtree.as_mut().unwrap();

        ctx.recreate_swapchain_if_needed(window);

        let [w, h] = ctx.window_size();
        if w == 0 || h == 0 {
            return;
        }

        let (image_index, suboptimal, acquire_future) = match ctx.acquire_next_image() {
            Ok(r) => r,
            Err(()) => return,
        };

        if suboptimal {
            ctx.recreate_swapchain = true;
        }

        // Update camera zoom if mouse is held
        if self.left_mouse_down || self.right_mouse_down {
            let direction = if self.left_mouse_down { 1.0 } else { -1.0 };
            self.camera
                .zoom_toward_screen_point(self.cursor_pos.0, self.cursor_pos.1, w, h, direction);
        }

        let (vp_left, vp_right, vp_bottom, vp_top) = self.camera.viewport(w, h);
        let target_ps = self.camera.pixel_scale(w, h);

        // Find tiles that need computing
        let needed = quadtree.find_needed_tiles(
            vp_left,
            vp_right,
            vp_bottom,
            vp_top,
            target_ps,
            MAX_COMPUTE_PER_FRAME,
        );

        // Single command buffer for both compute and render
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Compute needed tiles first
        compute.dispatch_tiles(
            &mut builder,
            &ctx.descriptor_set_allocator,
            quadtree,
            &needed,
        );

        // Collect visible tiles for rendering (includes newly computed ones)
        let visible = quadtree.get_visible_tiles(vp_left, vp_right, vp_bottom, vp_top);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [w as f32, h as f32],
            depth_range: 0.0..=1.0,
        };

        // Render tiles
        renderer.render(
            &mut builder,
            ctx.framebuffers[image_index as usize].clone(),
            viewport,
            &ctx.descriptor_set_allocator,
            &ctx.memory_allocator,
            &visible,
            vp_left,
            vp_right,
            vp_bottom,
            vp_top,
        );

        let cb = builder.build().unwrap();

        let previous_future = ctx.previous_frame_end.take().unwrap();
        let after_exec = previous_future
            .join(acquire_future)
            .then_execute(ctx.queue.clone(), cb)
            .unwrap()
            .boxed();

        ctx.present(after_exec, image_index);

        // Request continuous redraw if zooming or tiles still needed
        if self.left_mouse_down || self.right_mouse_down || !needed.is_empty() {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default().with_title("Fractal Viewer - Mandelbrot");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let ctx = VulkanContext::new(window.clone());
        let compute = ComputeEngine::new(&ctx.device, MAX_ITERATIONS);
        let renderer = Renderer::new(&ctx.device, ctx.subpass());
        let quadtree = QuadTree::new(ctx.device.clone());

        self.ctx = Some(ctx);
        self.compute = Some(compute);
        self.renderer = Some(renderer);
        self.quadtree = Some(quadtree);

        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.recreate_swapchain = true;
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.left_mouse_down = pressed;
                        if pressed {
                            self.camera.reset_timer();
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                    }
                    MouseButton::Right => {
                        self.right_mouse_down = pressed;
                        if pressed {
                            self.camera.reset_timer();
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x, position.y);
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Error creating event loop");
    event_loop.set_control_flow(ControlFlow::Wait); // Use WaitUntil 5ms ?
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Error running app");
}
