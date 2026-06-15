#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use vulkano::{
    device::Device,
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage, sys::RawImage, view::ImageView},
    memory::{DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory, allocator::align_up},
};

pub const TILE_SIZE: u32 = 256;
pub const ARENA_SIZE: u32 = 1024;

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

pub struct TileArena {
    pub items: Vec<Arc<ImageView>>,
    free_slots: Vec<u32>,
}
fn find_memory_type(dev: &Device, memory_type_bits : u32) -> Option<u32> {
    let types = &dev.physical_device().memory_properties().memory_types;
    for (i, memory_type) in types.iter().enumerate() {
        if memory_type_bits & (1 << i) != 0 && memory_type.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
            return Some(i as u32);
        }
    }
    for i in 0..types.len() {
        if memory_type_bits & (1 << i) != 0 {
            return Some(i as u32);
        }
    }
    None
}
impl TileArena {
    pub fn new_default(dev: &Arc<Device>) -> Self {
        Self::new(dev, TILE_SIZE, ARENA_SIZE)
    }
    pub fn new(dev: &Arc<Device>, size: u32, count: u32) -> Self {
        let ici = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16_UNORM,
            extent: [size, size, 1],
            usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
            ..Default::default()
        };
        let make_raw = || { RawImage::new(dev.clone(),ici.clone()).expect("failed to create tile image") };
        let raw_image = make_raw();
        let req = raw_image.memory_requirements().first().expect("failed to get memory requirements");
        let l = req.layout;
        let sz = align_up(l.size(), l.alignment());

        let dm = Arc::new(DeviceMemory::allocate(dev.clone(), MemoryAllocateInfo{
            allocation_size: sz * count as u64,
            memory_type_index: find_memory_type(dev, req.memory_type_bits).expect("failed to find memory type"),
            ..Default::default()
        }).expect("failed to allocate tile memory"));

        let make_iv = |raw_image: RawImage, offset: u64| {
            ImageView::new_default(Arc::new(raw_image
                .bind_memory(std::iter::once(unsafe { ResourceMemory::from_device_memory_unchecked(dm.clone(), offset, l.size()) }))
                .map_err(|e| e.0)
                .expect("failed to bind memory"))).expect("failed to create image view")
        };

        let mut items: Vec<Arc<ImageView>> = Vec::with_capacity(count as usize);
        items.push(make_iv(raw_image, 0));

        for i in 1..count {
            items.push(make_iv(make_raw(), i as u64 * sz as u64));
        }

        TileArena {items, free_slots: (0..count).collect()}
    }

    pub fn allocate_slot(&mut self) -> Option<Arc<ImageView>> {
        match self.free_slots.pop() {
            Some(index) => Some(self.items[index as usize].clone()),
            None => None,
        }
    }

    pub fn free_slot(&mut self, index: u32) {
        self.free_slots.push(index);
    }

    pub fn is_empty(&self) -> bool {
        self.free_slots.len() >= self.items.len()
    }

    pub fn has_free_slots(&self) -> bool {
        !self.free_slots.is_empty()
    }
}

pub struct TilePool {
    arenas: Vec<TileArena>,
    dev: Arc<Device>,
}

impl TilePool {
    pub fn new(dev: Arc<Device>) -> Self {
        TilePool {
            arenas: Vec::new(),
            dev,
        }
    }

    pub fn allocate(&mut self) -> Arc<ImageView> {
        for arena in self.arenas.iter_mut() {
            if let Some(result) = arena.allocate_slot() {
                return result;
            }
        }
        eprintln!("Allocating arena {}", self.arenas.len() + 1);
        self.arenas.push_mut(TileArena::new_default(&self.dev)).allocate_slot().unwrap()
    }

    pub fn free(&mut self, slot: &Arc<ImageView>) {
        for arena in self.arenas.iter_mut() {
            let index = arena.items.iter().position(|item| item == slot);
            match index {
                Some(index) => {
                    arena.free_slot(index as u32);
                    return;
                }
                None => continue,
            }
        }
    }
}

pub struct QuadTree {
    pub tiles: HashMap<TileCoord, Arc<ImageView>>,
    pub pool: TilePool,
}

impl QuadTree {
    pub fn new(dev: Arc<Device>) -> Self {
        QuadTree {
            tiles: HashMap::new(),
            pool: TilePool::new(dev),
        }
    }

    pub fn get(&self, coord: &TileCoord) -> Option<&Arc<ImageView>> {
        self.tiles.get(coord)
    }

    /// Insert a tile, returning the slot it was assigned.
    pub fn insert(&mut self, coord: TileCoord) -> Arc<ImageView> {
        if let Some(slot) = self.tiles.get(&coord) {
            return slot.clone();
        }
        let slot = self.pool.allocate();
        self.tiles.insert(coord, slot.clone());
        slot
    }

    /// Remove a tile and free its slot.
    pub fn remove(&mut self, coord: &TileCoord) {
        if let Some(slot) = self.tiles.remove(coord) {
            self.pool.free(&slot);
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
    ) -> Vec<(TileCoord, Arc<ImageView>)> {
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
        result: &mut Vec<(TileCoord, Arc<ImageView>)>,
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
        if let Some(slot) = self.tiles.get(&coord) {
            result.push((coord, slot.clone()));
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
}
