//! The spatial grid that accelerates the neighbour search (PLAN.md phase 3).
//!
//! Mirrors the CPU oracle's hash grid in `murmur-ref/src/neighbours.rs`, with
//! one difference: where the oracle keeps a linked list per bucket, the GPU
//! sorts the birds by bucket with an atomic histogram, an exclusive prefix scan
//! and a scatter, so a bucket is one contiguous range. The cells, the hash and
//! the exact distance check are the oracle's, so the neighbour sets agree (the
//! phase 3 acceptance test compares them against the brute-force walk).
//!
//! The world is unbounded, so cells hash into a fixed-size table rather than a
//! dense grid over a bounding box: no per-step recentring reduction and no bound
//! on where the flock flies. [`buckets_for`] sizes the table from the bird count
//! so collisions stay rare.
//!
//! The buffers and pipelines live here; [`crate::sim::Sim`] owns one, binds its
//! buffers alongside the bird buffers and records its passes. `shaders/grid.wgsl`
//! documents the passes and which `Grid` method each mirrors.

use std::mem::size_of;

use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, CommandEncoderDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineLayout, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::BIRD_WGSL;
use crate::device::Gpu;
use crate::pass::{Timeline, dispatch};

/// The shader header shared with `sim.wgsl` (`Params`, `Bird`, the cell hash).
pub(crate) const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
/// This module's passes.
const GRID_WGSL: &str = include_str!("../shaders/grid.wgsl");

/// Smallest bucket table (keeps the scan cheap for small flocks).
const MIN_BUCKETS: u32 = 1 << 16;
/// Largest bucket table: the two-level scan scans at most this many blocks.
const MAX_BUCKETS: u32 = 1 << 22;

/// The GPU grid's cell size \[m\].
///
/// The oracle's `neighbours::Grid` uses `params::GRID_CELL = 4.0 m`, but the
/// cell size is only an acceleration parameter: `for_each_within` and the
/// shader's `gather_grid` both keep a candidate only after an exact distance
/// check, so any cell size returns the same neighbour set. Phase 3 tied the two
/// to `GRID_CELL`; phase 4 sizes the GPU cell against the flock instead.
///
/// 4 m is far too coarse for the model's own settled flocks at scale: the
/// adaptive radius `R` shrinks as `n` grows (≈1.4 m at 2,000 birds, under 1 m
/// at 100,000). A query walks every cell the radius overlaps
/// ([`murmur_ref::neighbours::Grid::for_each_within`]), so at `R ≈ 0.9 m` a 4 m
/// cell makes a bird scan cells covering a few hundred m³ to find the half-dozen
/// neighbours inside the radius, while a 1 m cell scans only a few tens m³ —
/// roughly an order of magnitude fewer candidates. `decide` then no longer
/// dominates and the 100,000-bird run is feasible. The benchmark's synthetic
/// flock has only ~7 neighbours per bird, so it cannot show this; the cell
/// matters on the model's own dense flocks. Phase 9 will size it against the
/// benchmark; it is a performance knob, never a model change.
pub const CELL: f32 = 1.0;

/// Threads per workgroup and buckets per scan block, matching `grid.wgsl`.
const WG: u32 = 256;
const ELEMS: u32 = 8;
const BLOCK: u32 = WG * ELEMS;

/// Binding indices of the grid buffers in the simulation's bind group layout.
/// Eight storage buffers in total (the WebGPU limit), so `counts` doubles as
/// the scatter cursor and `block_sums` as the block offsets (see `grid.wgsl`).
pub const BINDING_BUCKET: u32 = 4;
pub const BINDING_COUNTS: u32 = 5;
pub const BINDING_OFFSETS: u32 = 6;
pub const BINDING_SORTED: u32 = 7;
pub const BINDING_BLOCK_SUMS: u32 = 8;

/// The bucket table size for `n` birds: twice the flock, rounded up to a power
/// of two and clamped. Twice the flock keeps the average bucket near a single
/// bird, so a query scans few candidates; the clamps bound the scan's cost.
pub fn buckets_for(n: usize) -> u32 {
    let want = (n.max(1) as u64 * 2).next_power_of_two();
    want.clamp(MIN_BUCKETS as u64, MAX_BUCKETS as u64) as u32
}

/// A built spatial grid: the buffers and the seven passes that fill them.
pub struct Grid {
    /// Bucket table size (a power of two).
    pub(crate) buckets: u32,
    /// Number of scan blocks: `buckets / BLOCK`, rounded up.
    blocks: u32,
    pub(crate) bucket: Buffer,
    /// The bucket histogram, then the scatter cursor (see `grid.wgsl`).
    pub(crate) counts: Buffer,
    pub(crate) offsets: Buffer,
    pub(crate) sorted: Buffer,
    /// The block totals, then the block start offsets (see `grid.wgsl`).
    pub(crate) block_sums: Buffer,
    clear: ComputePipeline,
    histogram: ComputePipeline,
    scan_blocks: ComputePipeline,
    scan_totals: ComputePipeline,
    scan_add: ComputePipeline,
    scatter: ComputePipeline,
    sort_buckets: ComputePipeline,
}

impl Grid {
    /// Creates the grid's buffers and pipelines for `n` birds, reusing the
    /// simulation's pipeline layout so one bind group can carry both.
    pub fn new(gpu: &Gpu, layout: &PipelineLayout, n: usize) -> Self {
        let device = &gpu.device;
        let buckets = buckets_for(n);
        let blocks = buckets.div_ceil(BLOCK);
        let source = format!("{COMMON_WGSL}\n{BIRD_WGSL}\n{GRID_WGSL}");
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("grid"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let storage = |label: &str, bytes: u64| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let words = size_of::<u32>() as u64;
        let bucket = storage("grid: bucket", buckets as u64 * words);
        let counts = storage("grid: counts", buckets as u64 * words);
        let offsets = storage("grid: offsets", (buckets as u64 + 1) * words);
        let sorted = storage("grid: sorted", n.max(1) as u64 * words);
        let block_sums = storage("grid: block sums", blocks as u64 * words);

        // `offsets[buckets]` is the end of the last bucket's range: always the
        // bird count, written once (the scan only writes the first `buckets`).
        gpu.queue
            .write_buffer(&offsets, buckets as u64 * words, &(n as u32).to_le_bytes());

        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let clear = pipeline("grid_clear");
        let histogram = pipeline("grid_histogram");
        let scan_blocks = pipeline("grid_scan_blocks");
        let scan_totals = pipeline("grid_scan_totals");
        let scan_add = pipeline("grid_scan_add");
        let scatter = pipeline("grid_scatter");
        let sort_buckets = pipeline("grid_sort_buckets");

        Self {
            buckets,
            blocks,
            bucket,
            counts,
            offsets,
            sorted,
            block_sums,
            clear,
            histogram,
            scan_blocks,
            scan_totals,
            scan_add,
            scatter,
            sort_buckets,
        }
    }

    /// The bucket table size.
    pub fn buckets(&self) -> u32 {
        self.buckets
    }

    /// Encodes the seven grid passes for `n` birds into `encoder`, using
    /// `bind_group` (which must bind the bird `src` buffer and every grid
    /// buffer, see [`crate::sim::Sim`]).
    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        n: usize,
        timeline: &mut Option<Timeline<'_>>,
    ) {
        let birds = (n as u32).div_ceil(WG);
        dispatch(
            encoder,
            &self.clear,
            bind_group,
            self.buckets.div_ceil(WG),
            timeline.as_mut(),
        );
        dispatch(
            encoder,
            &self.histogram,
            bind_group,
            birds,
            timeline.as_mut(),
        );
        dispatch(
            encoder,
            &self.scan_blocks,
            bind_group,
            self.blocks,
            timeline.as_mut(),
        );
        dispatch(encoder, &self.scan_totals, bind_group, 1, timeline.as_mut());
        dispatch(
            encoder,
            &self.scan_add,
            bind_group,
            self.blocks,
            timeline.as_mut(),
        );
        dispatch(encoder, &self.scatter, bind_group, birds, timeline.as_mut());
        dispatch(
            encoder,
            &self.sort_buckets,
            bind_group,
            self.buckets.div_ceil(WG),
            timeline.as_mut(),
        );
    }

    /// The exclusive prefix sums of the bucket histogram (`buckets + 1` of
    /// them), for the scan unit test.
    pub fn read_offsets(&self, gpu: &Gpu) -> Vec<u32> {
        read_u32(gpu, &self.offsets, self.buckets as usize + 1)
    }

    /// The bird index in each sorted slot, for the sort unit test.
    pub fn read_sorted(&self, gpu: &Gpu, n: usize) -> Vec<u32> {
        read_u32(gpu, &self.sorted, n)
    }

    /// The bucket each bird was placed in, for the sort unit test.
    pub fn read_bucket(&self, gpu: &Gpu, n: usize) -> Vec<u32> {
        read_u32(gpu, &self.bucket, n)
    }
}

/// Reads `len` `u32`s from the front of `buffer`.
fn read_u32(gpu: &Gpu, buffer: &Buffer, len: usize) -> Vec<u32> {
    let len = (len * size_of::<u32>()) as u64;
    let staging = gpu.device.create_buffer(&BufferDescriptor {
        label: Some("grid: readback"),
        size: len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("grid: readback"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len);
    gpu.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("waiting for the grid readback to finish");
    rx.recv()
        .expect("the poll ran the mapping callback before returning")
        .expect("mapping the grid readback buffer");
    let range = staging
        .get_mapped_range(..)
        .expect("reading the mapped range");
    let out = range
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    drop(range);
    staging.unmap();
    out
}
