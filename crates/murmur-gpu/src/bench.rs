//! `--bench`: per-pass GPU timestamps at a set of bird counts (PLAN.md phase
//! 3).
//!
//! Every number comes from GPU timestamp queries ([`crate::pass`]) — AGENTS.md
//! forbids reporting performance any other way — next to the adapter the run
//! landed on, which the caller prints.
//!
//! The benchmark does not start from the model's initial condition: 3,000,000
//! birds packed into the model's 18 m starting ellipsoid is far denser than any
//! settled flock, and the grid's cost is set by the local density (the number of
//! birds a perception radius holds), not by the bird count alone. [`setup`]
//! scatters the flock uniformly at the density the reference settles at instead,
//! and warms it up so the adaptive radius and the grid caches are steady before
//! anything is measured.

use bytemuck::Zeroable as _;
use glam::Vec3;
use murmur_ref::params as p;
use rand_chacha::rand_core::{RngCore as _, SeedableRng as _};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, PollType, QuerySetDescriptor,
    QueryType,
};

use crate::bird::{Bird, initial_rng};
use crate::device::Gpu;
use crate::pass::{STEP_PASSES, TIMESTAMPS_PER_STEP};
use crate::sim::Sim;

/// Birds per cubic metre in the benchmark scatter. Chosen so a bird's perception
/// radius holds about the reference's settled neighbour count (the model's
/// `N_C` is 6.5); see [`setup`].
const DENSITY: f32 = 0.5;

/// The perception radius the benchmark starts birds with. Together with
/// [`DENSITY`] that is about seven neighbours, the reference's settled value.
const RADIUS: f32 = 1.5;

/// Steps run before any timestamp is read, so the radius and caches settle.
const WARMUP: u32 = 20;

/// One pass's mean GPU time over the sampled steps.
pub struct PassTiming {
    /// Pass name, from [`STEP_PASSES`].
    pub pass: &'static str,
    /// Mean GPU time \[ms\].
    pub millis: f64,
}

/// The result of benchmarking one bird count.
pub struct BenchReport {
    /// Birds simulated.
    pub birds: usize,
    /// Steps sampled.
    pub steps: u32,
    /// One entry per pass, in step order.
    pub passes: Vec<PassTiming>,
}

impl BenchReport {
    /// The mean time of one whole step \[ms\] (the sum of its passes).
    pub fn step_millis(&self) -> f64 {
        self.passes.iter().map(|p| p.millis).sum()
    }

    /// Simulated bird-steps per second: `birds / step_seconds`.
    pub fn birds_per_second(&self) -> f64 {
        let step = self.step_millis();
        if step <= 0.0 {
            return f64::INFINITY;
        }
        self.birds as f64 / (step / 1000.0)
    }
}

/// Benchmarks `birds` birds for `steps` timed steps each and returns the
/// per-pass mean times.
///
/// Panics if the device was not opened with `TIMESTAMP_QUERY`; the caller is
/// expected to check `adapter.features()` and skip with a message.
pub fn run(gpu: &Gpu, birds: usize, steps: u32) -> BenchReport {
    assert!(
        gpu.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY),
        "the benchmark needs the device's TIMESTAMP_QUERY feature"
    );
    let device = &gpu.device;

    let mut sim = Sim::new(gpu, crate::params::Params::new(birds, 1));
    sim.write_birds(gpu, &setup(birds, 1));
    sim.advance(gpu, WARMUP);

    let query_set = device.create_query_set(&QuerySetDescriptor {
        label: Some("bench: timestamps"),
        ty: QueryType::Timestamp,
        count: TIMESTAMPS_PER_STEP,
    });
    let bytes = TIMESTAMPS_PER_STEP as u64 * 8;
    let resolve = device.create_buffer(&BufferDescriptor {
        label: Some("bench: resolve"),
        size: bytes,
        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let map = device.create_buffer(&BufferDescriptor {
        label: Some("bench: map"),
        size: bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let period = gpu.queue.get_timestamp_period() as f64;
    let mut totals = vec![0.0f64; STEP_PASSES.len()];
    for _ in 0..steps {
        sim.advance_timeline(gpu, &query_set);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench: resolve timestamps"),
        });
        encoder.resolve_query_set(&query_set, 0..TIMESTAMPS_PER_STEP, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &map, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let ticks = read_ticks(gpu, &map, STEP_PASSES.len());
        for (total, (begin, end)) in totals.iter_mut().zip(ticks) {
            *total += (end - begin) as f64 * period / 1e6;
        }
    }

    let passes = STEP_PASSES
        .iter()
        .zip(totals)
        .map(|(pass, millis)| PassTiming {
            pass,
            millis: millis / steps.max(1) as f64,
        })
        .collect();
    BenchReport {
        birds,
        steps,
        passes,
    }
}

/// Maps `map` and returns the `(begin, end)` timestamp ticks of each pass.
fn read_ticks(gpu: &Gpu, map: &wgpu::Buffer, passes: usize) -> Vec<(u64, u64)> {
    let (tx, rx) = std::sync::mpsc::channel();
    map.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    gpu.device
        .poll(PollType::wait_indefinitely())
        .expect("waiting for the timestamps to arrive");
    rx.recv()
        .expect("the poll ran the mapping callback before returning")
        .expect("mapping the timestamp buffer");
    let range = map.get_mapped_range(..).expect("reading the timestamps");
    let ticks: Vec<u64> = range
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from_le_bytes(*w))
        .collect();
    drop(range);
    map.unmap();
    (0..passes)
        .map(|k| (ticks[2 * k], ticks[2 * k + 1]))
        .collect()
}

/// A settled-like flock: `n` birds scattered uniformly in a cube whose density
/// is [`DENSITY`], flying together with the model's cruise speed and perception
/// radius. See the module docs for why this is not the model's initial
/// condition.
pub fn setup(n: usize, seed: u64) -> Vec<Bird> {
    let side = (n as f32 / DENSITY).cbrt();
    let centre = Vec3::new(0.0, 0.0, p::Z0);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut birds = Vec::with_capacity(n);
    for i in 0..n {
        let offset = Vec3::new(
            (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * side,
            (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * side,
            (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * side,
        );
        let jitter = Vec3::new(
            rng.next_u32() as f32 / u32::MAX as f32 - 0.5,
            rng.next_u32() as f32 / u32::MAX as f32 - 0.5,
            rng.next_u32() as f32 / u32::MAX as f32 - 0.5,
        ) * 0.05;
        let vel = (Vec3::X + jitter).normalize() * p::V0;
        birds.push(Bird {
            pos: (centre + offset).to_array(),
            vel: vel.to_array(),
            radius: RADIUS,
            next_decision: rng.next_u32() as f32 / u32::MAX as f32 * p::DU,
            flap_phase: rng.next_u32() as f32 / u32::MAX as f32 * std::f32::consts::TAU,
            flap_amp: 1.0,
            id: i as u32,
            rng: initial_rng(seed, i as u32),
            ..Bird::zeroed()
        });
    }
    birds
}
