use std::time::Instant;

const ZOOM_SPEED: f64 = 2.0; // 2x zoom per second

pub struct Camera {
    pub center_x: f64,
    pub center_y: f64,
    /// Half-extent in complex plane units: viewport spans center +/- scale
    pub scale: f64,
    last_update: Instant,
}

impl Camera {
    pub fn new() -> Self {
        Camera {
            center_x: -0.5,
            center_y: 0.0,
            scale: 2.0,
            last_update: Instant::now(),
        }
    }

    /// Zoom toward (or away from) a point in screen coordinates.
    /// `direction`: positive = zoom in, negative = zoom out
    pub fn zoom_toward_screen_point(
        &mut self,
        screen_x: f64,
        screen_y: f64,
        window_width: u32,
        window_height: u32,
        direction: f64,
    ) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        let dt = dt.min(0.1);

        let zoom_factor = ZOOM_SPEED.powf(direction * dt);

        let (world_x, world_y) =
            self.screen_to_world(screen_x, screen_y, window_width, window_height);

        // Move center toward the target point proportionally to the zoom change
        self.center_x += (world_x - self.center_x) * (1.0 - 1.0 / zoom_factor);
        self.center_y += (world_y - self.center_y) * (1.0 - 1.0 / zoom_factor);

        self.scale /= zoom_factor;
    }

    pub fn reset_timer(&mut self) {
        self.last_update = Instant::now();
    }

    pub fn screen_to_world(
        &self,
        screen_x: f64,
        screen_y: f64,
        window_width: u32,
        window_height: u32,
    ) -> (f64, f64) {
        let aspect = window_width as f64 / window_height as f64;
        let nx = (screen_x / window_width as f64) * 2.0 - 1.0;
        let ny = (screen_y / window_height as f64) * 2.0 - 1.0;
        let world_x = self.center_x + nx * self.scale * aspect;
        let world_y = self.center_y + ny * self.scale;
        (world_x, world_y)
    }

    /// Returns (left, right, bottom, top) in complex plane coordinates.
    pub fn viewport(&self, window_width: u32, window_height: u32) -> (f64, f64, f64, f64) {
        let aspect = window_width as f64 / window_height as f64;
        let half_w = self.scale * aspect;
        let half_h = self.scale;
        (
            self.center_x - half_w,
            self.center_x + half_w,
            self.center_y - half_h,
            self.center_y + half_h,
        )
    }

    /// Complex-plane units per pixel at current zoom/window size.
    pub fn pixel_scale(&self, _window_width: u32, window_height: u32) -> f64 {
        (self.scale * 2.0) / window_height as f64
    }
}
