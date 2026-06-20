#![allow(dead_code)]

use std::{num::NonZero, sync::Arc};
use vulkano::{
    device::{Device, DeviceOwned},
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage, sys::RawImage, view::ImageView},
    memory::{DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory, allocator::align_up},
};

use crate::{camera::Viewport, tile::ChildNode::Present};

pub const TILE_SIZE: u32 = 256;
pub const ARENA_SIZE: NonZero<u32> = NonZero::new(1024).unwrap();

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

struct TileArena {
    dm: Arc<DeviceMemory>,
    mem_sz: u64,
    tile_sz: u32,
    ptr: u32, cnt: u32,
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
fn make_raw(dev: &Arc<Device>, size: u32) -> RawImage {
    let ici = ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::R16_UNORM,
        extent: [size, size, 1],
        usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
        ..Default::default()
    };
    RawImage::new(dev.clone(),ici.clone()).expect("failed to create tile image")
}
impl TileArena {
    pub fn new_default(dev: &Arc<Device>) -> (Self, Arc<ImageView>) {
        Self::new(dev, TILE_SIZE, ARENA_SIZE)
    }
    fn make_iv(&mut self, ri: RawImage) -> Arc<ImageView> {
        let iv = ImageView::new_default(Arc::new(ri
            .bind_memory(std::iter::once(unsafe { ResourceMemory::from_device_memory_unchecked(
                self.dm.clone(), self.ptr as u64 * self.mem_sz, self.mem_sz) }))
            .map_err(|e| e.0)
            .expect("failed to bind memory"))).expect("failed to create image view");
        self.ptr += 1;
        iv
    }
    pub fn new(dev: &Arc<Device>, tile_sz: u32, count: NonZero<u32>) -> (Self, Arc<ImageView>) {
        let count = count.get();
        let raw_image = make_raw(dev, tile_sz);
        let req = raw_image.memory_requirements().first().expect("failed to get memory requirements");
        let l = req.layout;
        let mem_sz = align_up(l.size(), l.alignment());

        let mut result = TileArena {
            dm: Arc::new(DeviceMemory::allocate(dev.clone(), MemoryAllocateInfo{
                allocation_size: mem_sz * count as u64,
                memory_type_index: find_memory_type(dev, req.memory_type_bits).expect("failed to find memory type"),
                ..Default::default()
            }).expect("failed to allocate tile memory")),
            mem_sz, tile_sz, ptr: 0, cnt: count,
        };
        let iv = result.make_iv(raw_image);
        (result, iv)
    }

    pub fn make(&mut self) -> Option<Arc<ImageView>> {
        if self.exhausted() {
            None
        } else {
            Some(self.make_iv(make_raw(self.dm.device(), self.tile_sz)))
        }
    }

    pub fn exhausted(&self) -> bool {
        self.ptr >= self.cnt
    }
}

pub struct TilePool {
    cur_arena: Option<TileArena>,
    dev: Arc<Device>,
}

impl TilePool {
    pub fn new(dev: Arc<Device>) -> Self {
        TilePool { cur_arena: None, dev}
    }

    pub fn allocate(&mut self) -> Arc<ImageView> {
        if let Some(arena) = &mut self.cur_arena {
            if let Some(iv) = arena.make() {
                return iv;
            }
        }
        let (new_arena, iv) = TileArena::new_default(&self.dev);
        self.cur_arena = Some(new_arena);
        iv
    }
}

const MAX_DEPTH: u32 = 46;

enum ChildNode {
    Pending,
    Blocked,
    Present(Box<QuadTreeNode>)
}

impl ChildNode {
    fn as_ref(&self) -> Option<&QuadTreeNode> {
        match self {
            ChildNode::Present(node) => Some(node),
            _ => None,
        }
    }
}

struct QuadTreeNode {
    item: Arc<ImageView>,
    children: [ChildNode; 4],
}

impl QuadTreeNode {
    fn new(item: Arc<ImageView>, blocks: [bool; 4]) -> Self {
        QuadTreeNode {item: item, children: blocks.map(|b| if b { ChildNode::Blocked } else { ChildNode::Pending })}
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
    fn insert(&mut self, coord: TileCoord, tile: Option<Arc<ImageView>>, blocks: [bool; 4]) -> bool {
        match coord.level {
            1 => {
                self.children[(coord.y * 2 + coord.x) as usize] = match tile {
                    None => ChildNode::Pending,
                    Some(t) => ChildNode::Present(Box::new(QuadTreeNode::new(t, blocks))),
                };
                true
            }
            _ => {
                let (quadrant, child_coord) = coord.child();
                match &mut self.children[quadrant] {
                    Present(child) => child.insert(child_coord, tile, blocks),
                    _ => false,
                }
            }
        }
    }

    fn children_iter(&mut self, coord: TileCoord, vp: &Viewport) -> impl Iterator<Item = (TileCoord, &mut ChildNode)> {
        coord.children().into_iter().zip(self.children.iter_mut()).filter(|(child_coord, _)| child_coord.overlaps(vp))
    }

    fn collect_visible(&mut self, coord: TileCoord, vp: &Viewport, result: &mut Vec<(TileCoord, Arc<ImageView>)>) {
        let target_depth = coord.pixel_scale() <= vp.pixel_scale || coord.level >= MAX_DEPTH;
        if target_depth || self.children.iter().any(|c| !matches!(c, ChildNode::Present(_))) {
            result.push((coord.clone(), self.item.clone()));
        }
        if target_depth {
            return;
        }
        for (child_coord, child) in self.children_iter(coord, vp) {
            if let Present(child) = child {
                child.collect_visible(child_coord, vp, result);
            }
        }
    }

    fn next_tile(&mut self, coord: TileCoord, vp: &Viewport) -> Option<(TileCoord, &mut ChildNode)> {
        let mut result: Option<(TileCoord, &mut ChildNode)> = None;
        let mut go_deeper = coord.level < MAX_DEPTH && coord.pixel_scale() > vp.pixel_scale;
        for (child_coord, child) in self.children_iter(coord, vp) {
            match child {
                ChildNode::Pending => return Some((child_coord, child)),
                ChildNode::Present(child) if go_deeper => {
                    if let Some(deeper_tile) = child.next_tile(child_coord, vp) {
                        if deeper_tile.0.level <= child_coord.level+1 {
                            go_deeper = false;
                        }
                        result = match result {
                            None => Some(deeper_tile),
                            Some(tc) if deeper_tile.0.level < tc.0.level => Some(deeper_tile),
                            otc => otc,
                        };
                    }
                }
                _ => (),
            }
        }
        result
    }
}
pub struct QuadTree {
    root: ChildNode,
}
impl QuadTree {
    pub fn new() -> Self {
        QuadTree { root: ChildNode::Pending }
    }
    pub fn insert(&mut self, coord: TileCoord, tile: Option<Arc<ImageView>>, blocks: [bool; 4]) -> bool {
        match coord.level {
            0 => {
                self.root = match tile {
                    None => ChildNode::Pending,
                    Some(t) => ChildNode::Present(Box::new(QuadTreeNode::new(t, blocks))),
                };
                true
            }
            _ => {
                match &mut self.root {
                    Present(root) => root.insert(coord, tile, blocks),
                    _ => false,
                }
            }
        }
    }
    /// Find visible tiles at the best available resolution for the given viewport.
    /// Returns list of (TileCoord, TileSlot, screen_rect) for rendering.
    pub fn get_visible_tiles(&mut self, vp: &Viewport) -> Vec<(TileCoord, Arc<ImageView>)> {
        let mut result = Vec::new();
        if let Present(root) = &mut self.root {
            root.collect_visible(TileCoord::root(), vp, &mut result);
        }
        result
    }
    /// Find next tiles that need to be computed to improve resolution for the viewport.
    /// Returns coords not yet in the tree that would refine visible areas.
    pub fn next_tile(&mut self, vp: Viewport) -> Option<TileCoord> {
        let root_coord = TileCoord::root();
        match &mut self.root {
            ChildNode::Pending => {
                self.root = ChildNode::Blocked;
                Some(root_coord)
            },
            ChildNode::Blocked => None,
            ChildNode::Present(root) => {
                let result = root.next_tile(root_coord, &vp.scale_pix(2.0));
                result.map(|(coord, child)| {
                    *child = ChildNode::Blocked;
                    coord
                })
            }
        }
    }
}
