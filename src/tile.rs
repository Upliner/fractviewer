#![allow(dead_code)]

use std::sync::Arc;
use vulkano::{
    device::Device,
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage, sys::RawImage, view::ImageView},
    memory::{DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory, allocator::align_up},
};

use crate::camera::Viewport;

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
    pub fn child(&self) -> (usize, TileCoord) {
        let lev = self.level - 1;
        let mask = !((!0u64) << lev);
        (((self.y >> lev) * 2 + (self.x >> lev)) as usize,
            TileCoord {
                level: lev,
                x: self.x & mask,
                y: self.y & mask,
            }
        )
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
    pub fn overlaps(&self, vp: &Viewport) -> bool {
        let (ox, oy) = self.origin();
        let ext = self.tile_extent();
        ox < vp.right && ox + ext > vp.left && oy < vp.top && oy + ext > vp.bottom
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

const MAX_DEPTH: u32 = 46;

struct QuadTreeNode {
    item: Arc<ImageView>,
    children: [Option<Box<QuadTreeNode>>; 4],
}

impl QuadTreeNode {
    fn new(item: Arc<ImageView>) -> Self {
        QuadTreeNode {item: item, children: Default::default()}
    }

    fn get(&self, coord: TileCoord) -> Option<&Arc<ImageView>> {
        match coord.level {
            0 => Some(&self.item),
            _ => {
                let (quadrant, child_coord) = coord.child();
                self.children[quadrant].as_ref()?.get(child_coord)
            }
        }
    }
    fn insert(&mut self, coord: TileCoord, tile: Option<Arc<ImageView>>) -> bool {
        match coord.level {
            1 => {self.children[(coord.y * 2 + coord.x) as usize] = tile.map(|t| Box::new(QuadTreeNode::new(t))); true}
            _ => {
                let (quadrant, child_coord) = coord.child();
                match self.children[quadrant].as_mut() {
                    Some(child) => child.insert(child_coord, tile),
                    None => false,
                }
            }
        }
    }

    fn children_iter(&self, coord: TileCoord, vp: &Viewport) -> impl Iterator<Item = (TileCoord, &Option<Box<QuadTreeNode>>)> {
        coord.children().into_iter().zip(self.children.iter()).filter(|(child_coord, _)| child_coord.overlaps(vp))
    }

    fn collect_visible(&self, coord: TileCoord, vp: &Viewport, result: &mut Vec<(TileCoord, Arc<ImageView>)>) {
        let target_depth = coord.pixel_scale() <= vp.pixel_scale || coord.level >= MAX_DEPTH;
        if target_depth || self.children.iter().any(|c| c.is_none()) {
            result.push((coord.clone(), self.item.clone()));
        }
        if target_depth {
            return;
        }
        for (child_coord, child) in self.children_iter(coord, vp) {
            if let Some(child) = child {
                child.collect_visible(child_coord, vp, result);
            }
        }
    }

    fn next_tile(&self, coord: TileCoord, vp: &Viewport) -> Option<TileCoord> {
        let mut result: Option<TileCoord> = None;
        let go_deeper = coord.level < MAX_DEPTH && coord.pixel_scale() > vp.pixel_scale;
        for (child_coord, child) in self.children_iter(coord, vp) {
            match child {
                None => return Some(child_coord),
                Some(child) if go_deeper => {
                    if let Some(deeper_tile) = child.next_tile(child_coord, vp) {
                        result = match result {
                            None => Some(deeper_tile),
                            Some(tc) if deeper_tile.level < tc.level => Some(deeper_tile),
                            otc => otc,
                        }
                    }
                }
                _ => (),
            }
        }
        result
    }
}
pub struct QuadTree {
    root: Option<QuadTreeNode>,
}
impl QuadTree {
    pub fn new() -> Self {
        QuadTree { root: None }
    }
    pub fn insert(&mut self, coord: TileCoord, tile: Option<Arc<ImageView>>) -> bool {
        match coord.level {
            0 => { self.root = tile.map(|t| QuadTreeNode::new(t)); true}
            _ => {
                match self.root.as_mut() {
                    Some(root) => root.insert(coord, tile),
                    None => false,
                }
            }
        }
    }
    /// Find visible tiles at the best available resolution for the given viewport.
    /// Returns list of (TileCoord, TileSlot, screen_rect) for rendering.
    pub fn get_visible_tiles(&self, vp: &Viewport) -> Vec<(TileCoord, Arc<ImageView>)> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            root.collect_visible(TileCoord::root(), vp, &mut result);
        }
        result
    }
    /// Find next tiles that need to be computed to improve resolution for the viewport.
    /// Returns coords not yet in the tree that would refine visible areas.
    pub fn next_tile(&self, vp: Viewport) -> Option<TileCoord> {
        let root_coord = TileCoord::root();
        match &self.root {
            None => Some(root_coord),
            Some(root) => root.next_tile(root_coord, &vp.scale_pix(2.0)),
        }
    }
}
