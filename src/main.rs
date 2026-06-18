mod camera;
mod compute;
mod context;
mod render;
mod tile;

use std::{ops::{Deref, DerefMut}, sync::{Arc, Condvar, Mutex}, thread, time::Instant};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    pipeline::graphics::viewport::Viewport,
    sync::GpuFuture,
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use camera::Camera;
use compute::ComputeEngine;
use context::{VulkanContext, VulkanData};
use render::Renderer;
use tile::{QuadTree, TilePool};

const MAX_ITERATIONS: u32 = 8192;
struct Worker {
    data: Arc<SharedData>,
    join_handle: thread::JoinHandle<()>,
}
#[derive(PartialEq)]
enum WorkerState {
    Idle,
    Work,
    Finish
}
struct SharedData {
    window: Arc<Window>,
    quadtree: Mutex<QuadTree>,
    camera: Mutex<Camera>,
    state: Mutex<WorkerState>,
    cond_var: Condvar,
}
impl SharedData {
    fn new(window: Arc<Window>) -> Self {
        SharedData {
            window,
            quadtree: Mutex::new(QuadTree::new()),
            camera: Mutex::new(Camera::new()),
            state: Mutex::new(WorkerState::Work),
            cond_var: Condvar::new(),
        }
    }
    fn signal_check(&self) {
        let mut state = self.state.lock().unwrap();
        if *state == WorkerState::Idle {
            *state = WorkerState::Work;
        }
        self.cond_var.notify_all();
    }
    fn signal_exit(&self) {
        let mut state = self.state.lock().unwrap();
        *state = WorkerState::Finish;
        self.cond_var.notify_all();
    }
    fn go_idle(&self) {
        let mut state = self.state.lock().unwrap();
        if *state == WorkerState::Work {
            *state = WorkerState::Idle;
        }
        self.cond_var.notify_all();
    }
    fn check_state(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        while *state == WorkerState::Idle {
            state = self.cond_var.wait(state).unwrap();
        }
        return *state != WorkerState::Finish;
    }
}
impl Worker {
    fn new(vk: Arc<VulkanData>, window: Arc<Window>) -> Self {
        let data = Arc::new(SharedData::new(window));
        Worker {
            data: data.clone(),
            join_handle: thread::spawn(move || Worker::run(vk, data)),
        }
    }
    fn run(vk: Arc<VulkanData>, data: Arc<SharedData>) {
        let mut pool = TilePool::new(vk.device.clone());
        let compute = ComputeEngine::new(vk.clone(), MAX_ITERATIONS);
        let mut series_start: bool = false;
        let mut series_start_time: Instant = Instant::now();
        let mut tile_count: u32 = 0;
        while data.check_state() {
            if series_start {
                series_start = false;
                series_start_time = Instant::now();
            }
            let mut go_idle = || {
                series_start = true;
                if tile_count > 0 {
                    let elapsed = series_start_time.elapsed();
                    eprintln!("Computed {} tiles in {:?} ({} tiles/s)", tile_count, elapsed, tile_count as f64 / elapsed.as_secs_f64());
                    tile_count = 0;
                }
                data.go_idle();
            };
            let (w, h) = data.window.inner_size().into();
            if w == 0 || h == 0 {
                go_idle();
                continue;
            }

            // Find tiles that need computing
            let camera = data.camera.lock().unwrap();
            let (vp_left, vp_right, vp_bottom, vp_top, pixel_scale) = camera.viewport(w, h);
            drop(camera);
            let tile_coord = data.quadtree.lock().unwrap().next_tile(vp_left, vp_right, vp_bottom, vp_top, pixel_scale);
            if let Some(tile_coord) = tile_coord {
                let tile = pool.allocate();
                compute.compute_tile(
                    tile.clone(),
                    tile_coord,
                );
                if !data.quadtree.lock().unwrap().insert(tile_coord, Some(tile)) {
                    eprintln!("Failed to insert tile to location {:?}", tile_coord);
                }
                tile_count += 1;
                data.window.request_redraw();
            } else {
                go_idle();
            }
        }
    }
    fn finish(self) {
        self.data.signal_exit();
        self.join_handle.join().unwrap();
    }
}
struct AppInternal {
    ctx: VulkanContext,
    renderer: Renderer,
    worker: Worker,
    left_mouse_down: bool,
    right_mouse_down: bool,
    cursor_pos: (f64, f64),
}

struct App(Option<AppInternal>);

impl App {
    fn new() -> Self {
        App(None)
    }
}
impl AppInternal {
    fn quadtree(&self) -> impl Deref<Target = QuadTree> {
        self.worker.data.quadtree.lock().unwrap()
    }
    fn camera(&self) -> impl DerefMut<Target = Camera> {
        self.worker.data.camera.lock().unwrap()
    }
    fn window(&self) -> &Window {
        self.worker.data.window.as_ref()
    }
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let attrs = WindowAttributes::default().with_title("Fractal Viewer - Mandelbrot");
        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));
        let ctx = VulkanContext::new(window.clone());
        let renderer = Renderer::new(&ctx.data.device, ctx.subpass());
        let worker = Worker::new(ctx.data.clone(), window);
        AppInternal {
            ctx,
            renderer,
            worker,
            left_mouse_down: false,
            right_mouse_down: false,
            cursor_pos: (0.0, 0.0),
        }
    }
    fn redraw(&mut self) {
        self.ctx.recreate_swapchain_if_needed(self.worker.data.window.as_ref());

        let [w, h] = self.ctx.window_size();
        if w == 0 || h == 0 {
            return;
        }

        let (image_index, suboptimal, acquire_future) = match self.ctx.acquire_next_image() {
            Ok(r) => r,
            Err(()) => return,
        };

        if suboptimal {
            self.ctx.recreate_swapchain = true;
        }

        let mut camera = self.camera();
        let (vp_left, vp_right, vp_bottom, vp_top, pixel_scale) = camera.viewport(w, h);
        // Update camera zoom if mouse is held
        if self.left_mouse_down || self.right_mouse_down {
            let direction = if self.left_mouse_down { 1.0 } else { -1.0 };
            camera.zoom_toward_screen_point(self.cursor_pos.0, self.cursor_pos.1, w, h, direction);
        }
        drop(camera);
        let start = Instant::now();
        let vkdata = &self.ctx.data.as_ref();
        let mut builder = AutoCommandBufferBuilder::primary(
            vkdata.command_buffer_allocator.clone(),
            vkdata.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();


        // Collect visible tiles for rendering (includes newly computed ones)
        let visible = self.quadtree().get_visible_tiles(vp_left, vp_right, vp_bottom, vp_top, pixel_scale);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [w as f32, h as f32],
            depth_range: 0.0..=1.0,
        };

        // Render tiles
        self.renderer.render(
            &mut builder,
            self.ctx.framebuffers[image_index as usize].clone(),
            viewport,
            &vkdata.descriptor_set_allocator,
            &self.ctx.memory_allocator,
            &visible,
            vp_left,
            vp_right,
            vp_bottom,
            vp_top,
        );

        let cb = builder.build().unwrap();

        let previous_future = self.ctx.previous_frame_end.take().unwrap();
        let after_exec = previous_future
            .join(acquire_future)
            .then_execute(vkdata.graphics_queue.clone(), cb)
            .unwrap()
            .boxed();

        //eprintln!("presenting...");
        self.ctx.present(after_exec, image_index);
        eprintln!("Render time: {:?}", start.elapsed());
        // Request continuous redraw if zooming or tiles still needed
        if self.left_mouse_down || self.right_mouse_down {
            self.window().request_redraw();
            self.worker.data.signal_check();
        }
    }
    fn window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Resized(_) => {
                self.ctx.recreate_swapchain = true;
                self.worker.data.signal_check();
                self.window().request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.left_mouse_down = pressed;
                        if pressed {
                            self.camera().reset_timer();
                            self.window().request_redraw();
                        }
                    }
                    MouseButton::Right => {
                        self.right_mouse_down = pressed;
                        if pressed {
                            self.camera().reset_timer();
                            self.window().request_redraw();
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.0.is_some() {
            return;
        }
        self.0 = Some(AppInternal::new(event_loop));
        //let compute = ComputeEngine::new(&ctx.device, MAX_ITERATIONS);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::CloseRequested {
            if let Some(app) = self.0.take() {
                app.worker.finish();
            }
            event_loop.exit();
        } else if let Some(app) = &mut self.0 {
            app.window_event(event);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Error creating event loop");
    event_loop.set_control_flow(ControlFlow::Wait); // Use WaitUntil 5ms ?
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Error running app");
}
