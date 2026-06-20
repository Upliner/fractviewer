mod camera;
mod compute;
mod context;
mod render;
mod tile;

use std::{ops::DerefMut, sync::{Arc, Condvar, Mutex}, thread::{self, JoinHandle}, time::Instant};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage}, device::Queue, pipeline::graphics::viewport::Viewport, sync::GpuFuture
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
#[derive(PartialEq, Debug)]
enum WorkerState {
    Idle,
    Work,
    Finish
}
struct SharedData {
    vk: Arc<VulkanData>,
    pool: Mutex<TilePool>,
    window: Arc<Window>,
    quadtree: Mutex<QuadTree>,
    camera: Mutex<Camera>,
    state: Mutex<WorkerState>,
    cond_var: Condvar,
}
struct SeriesCounter {
    worker_id: usize,
    start_time: Instant,
    tile_count: u32,
    cull_count: u32,
}
impl SeriesCounter {
    fn new(worker_id: usize) -> Self {
        SeriesCounter {
            worker_id,
            start_time: Instant::now(),
            tile_count: 0,
            cull_count: 0,
       }
    }
    fn report(&mut self) {
        if self.tile_count == 0 {
            return;
        }
        let elapsed = self.start_time.elapsed();
        eprintln!("Worker {} Computed {} tiles in {:?} ({:.2} tiles/s), {} tiles culled",
            self.worker_id, self.tile_count, elapsed, self.tile_count as f64 / elapsed.as_secs_f64(), self.cull_count);
        self.tile_count = 0;
        self.cull_count = 0;
    }
}
impl SharedData {
    fn new(vk: Arc<VulkanData>, window: Arc<Window>) -> Self {
        SharedData {
            pool: Mutex::new(TilePool::new(vk.device.clone())),
            vk,
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
    }
    fn check_state(&self, sc: &mut SeriesCounter) -> bool {
        let mut state = self.state.lock().unwrap();
        while *state == WorkerState::Idle {
            sc.report();
            state = self.cond_var.wait(state).unwrap();
            sc.start_time = Instant::now();
        }
        return *state != WorkerState::Finish;
    }
}
fn run_worker(data: Arc<SharedData>, q: Arc<Queue>, id: usize) -> JoinHandle<()> {
    thread::spawn(move || worker_func(data, q, id))
}
fn worker_func(data: Arc<SharedData>, q: Arc<Queue>, id: usize) {
    let vk = &data.vk;
    let mut compute = ComputeEngine::new(vk.clone(), MAX_ITERATIONS);
    let mut sc = SeriesCounter::new(id);
    while data.check_state(&mut sc) {
        let (w, h) = data.window.inner_size().into();
        if w == 0 || h == 0 {
            data.go_idle();
            continue;
        }
        // Find tiles that need computing
        let tile_coord = data.quadtree.lock().unwrap().next_tile(data.camera.lock().unwrap().viewport(w, h));
        if let Some(tile_coord) = tile_coord {
            let tile = data.pool.lock().unwrap().allocate();
            let blocks = compute.compute_tile(
                &q,
                tile.clone(),
                tile_coord,
            );
            blocks.iter().for_each(|block| if *block { sc.cull_count += 1; });
            if !data.quadtree.lock().unwrap().insert(tile_coord, Some(tile), blocks) {
                eprintln!("Failed to insert tile to location {:?}", tile_coord);
            }
            data.signal_check();
            sc.tile_count += 1;
            data.window.request_redraw();
        } else {
            data.go_idle();
        }
    }
}
struct AppInternal {
    ctx: VulkanContext,
    renderer: Renderer,
    sd: Arc<SharedData>,
    workers: Vec<JoinHandle<()>>,
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
    fn camera(&self) -> impl DerefMut<Target = Camera> {
        self.sd.camera.lock().unwrap()
    }
    fn window(&self) -> &Window {
        self.sd.window.as_ref()
    }
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let attrs = WindowAttributes::default().with_title("Fractal Viewer - Mandelbrot");
        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));
        let ctx = VulkanContext::new(window.clone());
        let sd = Arc::new(SharedData::new(ctx.data.clone(), window));
        AppInternal {
            renderer: Renderer::new(&ctx.data.device, ctx.subpass(), ctx.sample_count),
            workers: ctx.compute_queues.iter().enumerate().map(|(i, q)| run_worker(sd.clone(), q.clone(), i)).collect(),
            ctx, sd,
            left_mouse_down: false,
            right_mouse_down: false,
            cursor_pos: (0.0, 0.0),
        }
    }
    fn redraw(&mut self) {
        self.ctx.recreate_swapchain_if_needed(self.sd.window.as_ref());

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
        let vp = camera.viewport(w, h).scale_pix(0.5);
        // Update camera zoom if mouse is held
        if self.left_mouse_down || self.right_mouse_down {
            let direction = if self.left_mouse_down { 1.0 } else { -1.0 };
            camera.zoom_toward_screen_point(self.cursor_pos.0, self.cursor_pos.1, w, h, direction);
        }
        drop(camera);
        //let start = Instant::now();
        let vkdata = &self.ctx.data.as_ref();
        let mut builder = AutoCommandBufferBuilder::primary(
            vkdata.command_buffer_allocator.clone(),
            vkdata.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();


        // Collect visible tiles for rendering (includes newly computed ones)
        let visible = self.sd.quadtree.lock().unwrap().get_visible_tiles(&vp);

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
            &self.ctx.memory_allocator.clone(),
            &visible,
            &vp,
        );

        let cb = builder.build().unwrap();

        let previous_future = self.ctx.previous_frame_end.take().unwrap();
        let after_exec = previous_future
            .join(acquire_future)
            .then_execute(vkdata.graphics_queue.clone(), cb)
            .unwrap()
            .boxed();

        self.ctx.present(after_exec, image_index);
        //eprintln!("Render time: {:?}", start.elapsed());
        // Request continuous redraw if zooming or tiles still needed
        if self.left_mouse_down || self.right_mouse_down {
            self.window().request_redraw();
            self.sd.signal_check();
        }
    }
    fn window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Resized(_) => {
                self.ctx.recreate_swapchain = true;
                self.sd.signal_check();
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
    fn finish(self) {
        self.sd.signal_exit();
        self.workers.into_iter().for_each(|w| w.join().unwrap());
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
                app.finish();
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
