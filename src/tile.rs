#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use vulkano::{
    device::Device,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

pub const TILE_SIZE: u32 = 256;
pub const ATLAS_TILES_PER_SIDE: u32 = 16;
pub const ATLAS_SIZE: u32 = TILE_SIZE * ATLAS_TILES_PER_SIDE; // 4096
pub const TILES_PER_ATLAS: u32 = ATLAS_TILES_PER_SIDE * ATLAS_TILES_PER_SIDE; // 256

/// Root tile covers [-2.5, 1.5] x [-2.0, 2.0] in complex plane.
pub const ROOT_LEFT: f64 = -2.5;
pub const ROOT_BOTTOM: f64 = -2.0;
pub const ROOT_EXTENT: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub level: u32,
    pub x: u64,
    pub y: u64,
}

impl TileCoord {
    pub fn root() -> Self {
        TileCoord { level: 0, x: 0, y: 0 }
    }

    pub fn children(&self) -> [TileCoord; 4] {
        let l = self.level + 1;
        let bx = self.x * 2;
        let by = self.y * 2;
        [
            TileCoord { level: l, x: bx, y: by },
            TileCoord { level: l, x: bx + 1, y: by },
            TileCoord { level: l, x: bx, y: by + 1 },
            TileCoord { level: l, x: bx + 1, y: by + 1 },
        ]
    }

    pub fn parent(&self) -> Option<TileCoord> {
        if self.level == 0 {
            None
        } else {
            Some(TileCoord {
                level: self.level - 1,
                x: self.x / 2,
                y: self.y / 2,
            })
        }
    }

    /// Size of one tile at this level in complex-plane units.
    pub fn tile_extent(&self) -> f64 {
        ROOT_EXTENT / (1u64 << self.level) as f64
    }

    /// Bottom-left corner of this tile in complex-plane coordinates.
    pub fn origin(&self) -> (f64, f64) {
        let ext = self.tile_extent();
        (
            ROOT_LEFT + self.x as f64 * ext,
            ROOT_BOTTOM + self.y as f64 * ext,
        )
    }

    /// Complex-plane units per pixel for this tile.
    pub fn pixel_scale(&self) -> f64 {
        self.tile_extent() / (TILE_SIZE - 1) as f64
    }

    /// Does this tile's region overlap the given viewport?
    pub fn overlaps(&self, vp_left: f64, vp_right: f64, vp_bottom: f64, vp_top: f64) -> bool {
        let (ox, oy) = self.origin();
        let ext = self.tile_extent();
        ox < vp_right && ox + ext > vp_left && oy < vp_top && oy + ext > vp_bottom
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TileSlot {
    pub atlas_index: usize,
    pub slot_x: u32,
    pub slot_y: u32,
}

impl TileSlot {
    /// Pixel offset within the atlas image.
    pub fn pixel_offset(&self) -> (u32, u32) {
        (self.slot_x * TILE_SIZE, self.slot_y * TILE_SIZE)
    }

    /// UV coordinates (0..1) within the atlas for this tile.
    pub fn uv_rect(&self) -> [f32; 4] {
        let inv = 1.0 / ATLAS_TILES_PER_SIDE as f32;
        let u0 = self.slot_x as f32 * inv;
        let v0 = self.slot_y as f32 * inv;
        let inv_2 = inv / (TILE_SIZE * 2) as f32;
        [u0 + inv_2, v0 + inv_2, u0 + inv - inv_2, v0 + inv - inv_2]
    }
}

pub struct TileAtlas {
    pub image: Arc<Image>,
    pub image_view: Arc<ImageView>,
    free_slots: Vec<(u32, u32)>,
}

impl TileAtlas {
    pub fn new(memory_allocator: &Arc<StandardMemoryAllocator>) -> Self {
        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16_UNORM,
                extent: [ATLAS_SIZE, ATLAS_SIZE, 1],
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create atlas image");

        let image_view = ImageView::new_default(image.clone()).unwrap();

        let mut free_slots = Vec::with_capacity(TILES_PER_ATLAS as usize);
        for sy in (0..ATLAS_TILES_PER_SIDE).rev() {
            for sx in (0..ATLAS_TILES_PER_SIDE).rev() {
                free_slots.push((sx, sy));
            }
        }

        TileAtlas {
            image,
            image_view,
            free_slots,
        }
    }

    pub fn allocate_slot(&mut self) -> Option<(u32, u32)> {
        self.free_slots.pop()
    }

    pub fn free_slot(&mut self, sx: u32, sy: u32) {
        self.free_slots.push((sx, sy));
    }

    pub fn is_empty(&self) -> bool {
        self.free_slots.len() == TILES_PER_ATLAS as usize
    }

    pub fn has_free_slots(&self) -> bool {
        !self.free_slots.is_empty()
    }
}

pub struct TilePool {
    pub atlases: Vec<TileAtlas>,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

impl TilePool {
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        TilePool {
            atlases: Vec::new(),
            memory_allocator,
        }
    }

    pub fn allocate(&mut self) -> TileSlot {
        for (i, atlas) in self.atlases.iter_mut().enumerate() {
            if let Some((sx, sy)) = atlas.allocate_slot() {
                return TileSlot {
                    atlas_index: i,
                    slot_x: sx,
                    slot_y: sy,
                };
            }
        }
        let mut atlas = TileAtlas::new(&self.memory_allocator);
        let (sx, sy) = atlas.allocate_slot().unwrap();
        let idx = self.atlases.len();
        self.atlases.push(atlas);
        TileSlot {
            atlas_index: idx,
            slot_x: sx,
            slot_y: sy,
        }
    }

    pub fn free(&mut self, slot: TileSlot) {
        self.atlases[slot.atlas_index].free_slot(slot.slot_x, slot.slot_y);
    }
}

pub struct QuadTree {
    pub tiles: HashMap<TileCoord, TileSlot>,
    pub pool: TilePool,
}

impl QuadTree {
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        QuadTree {
            tiles: HashMap::new(),
            pool: TilePool::new(memory_allocator),
        }
    }

    pub fn get(&self, coord: &TileCoord) -> Option<&TileSlot> {
        self.tiles.get(coord)
    }

    /// Insert a tile, returning the slot it was assigned.
    pub fn insert(&mut self, coord: TileCoord) -> TileSlot {
        if let Some(&slot) = self.tiles.get(&coord) {
            return slot;
        }
        let slot = self.pool.allocate();
        self.tiles.insert(coord, slot);
        slot
    }

    /// Remove a tile and free its slot.
    pub fn remove(&mut self, coord: &TileCoord) {
        if let Some(slot) = self.tiles.remove(coord) {
            self.pool.free(slot);
        }
    }

    /// Find visible tiles at the best available resolution for the given viewport.
    /// Returns list of (TileCoord, TileSlot, screen_rect) for rendering.
    pub fn get_visible_tiles(
        &self,
        vp_left: f64,
        vp_right: f64,
        vp_bottom: f64,
        vp_top: f64,
    ) -> Vec<(TileCoord, TileSlot)> {
        let mut result = Vec::new();
        self.collect_visible(
            TileCoord::root(),
            vp_left,
            vp_right,
            vp_bottom,
            vp_top,
            &mut result,
        );
        result
    }

    fn collect_visible(
        &self,
        coord: TileCoord,
        vp_left: f64,
        vp_right: f64,
        vp_bottom: f64,
        vp_top: f64,
        result: &mut Vec<(TileCoord, TileSlot)>,
    ) {
        if !coord.overlaps(vp_left, vp_right, vp_bottom, vp_top) {
            return;
        }
        // Try to go deeper if children exist
        let children = coord.children();
        let all_children_exist = children.iter().all(|c| {
            self.tiles.contains_key(c)
                && c.overlaps(vp_left, vp_right, vp_bottom, vp_top)
                || !c.overlaps(vp_left, vp_right, vp_bottom, vp_top)
        });

        if all_children_exist && coord.level < 50 {
            // Check if any child actually needs to be visible
            let mut any_child_visible = false;
            for child in &children {
                if child.overlaps(vp_left, vp_right, vp_bottom, vp_top) {
                    any_child_visible = true;
                    self.collect_visible(*child, vp_left, vp_right, vp_bottom, vp_top, result);
                }
            }
            if any_child_visible {
                return;
            }
        }

        // Use this tile if it exists
        if let Some(&slot) = self.tiles.get(&coord) {
            result.push((coord, slot));
        }
    }

    /// Find tiles that need to be computed to improve resolution for the viewport.
    /// Returns coords not yet in the tree that would refine visible areas.
    pub fn find_needed_tiles(
        &self,
        vp_left: f64,
        vp_right: f64,
        vp_bottom: f64,
        vp_top: f64,
        target_pixel_scale: f64,
        max_per_frame: usize,
    ) -> Vec<TileCoord> {
        let mut needed = Vec::new();
        self.collect_needed(
            TileCoord::root(),
            vp_left,
            vp_right,
            vp_bottom,
            vp_top,
            target_pixel_scale,
            &mut needed,
            max_per_frame,
        );
        needed
    }

    fn collect_needed(
        &self,
        coord: TileCoord,
        vp_left: f64,
        vp_right: f64,
        vp_bottom: f64,
        vp_top: f64,
        target_pixel_scale: f64,
        needed: &mut Vec<TileCoord>,
        max: usize,
    ) {
        if needed.len() >= max {
            return;
        }
        if !coord.overlaps(vp_left, vp_right, vp_bottom, vp_top) {
            return;
        }

        if !self.tiles.contains_key(&coord) {
            needed.push(coord);
            return;
        }

        if coord.pixel_scale() > target_pixel_scale && coord.level < 50 {
            for child in coord.children() {
                self.collect_needed(
                    child,
                    vp_left,
                    vp_right,
                    vp_bottom,
                    vp_top,
                    target_pixel_scale,
                    needed,
                    max,
                );
            }
        }
    }

    /// Create a nearest-neighbor sampler suitable for integer tile data.
    pub fn create_tile_sampler(device: &Arc<Device>) -> Arc<Sampler> {
        Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap()
    }
}
