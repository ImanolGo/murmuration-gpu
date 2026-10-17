//! The GPU evening arc (PLAN.md phase 8).
//!
//! Mirrors `murmur_ref::arc::Arc`: groups arrive for the first 15 % of the
//! evening, the flock displays until 70 %, pours down into the reeds until 90 %
//! and then the night clears it, after which the next evening starts.
//!
//! The GPU's bird buffer is a fixed size, so the flock grows and shrinks by
//! *activating* and *retiring* slots rather than by moving records: the arc
//! spawns an arriving group into the next free slots, and a bird that has landed
//! is marked inactive in place. An inactive bird is skipped by the neighbour
//! walk, the flight integration and the renderer, so it is absent from the
//! flock, exactly as a removed record would be.
//!
//! One documented adaptation: the random stream. The oracle draws from ChaCha8,
//! which cannot run in WGSL; the GPU draws from the same counter-based PCG as
//! the birds (see `shaders/arc.wgsl`), so the arrival bearings and the groups'
//! shapes differ from the terminal's while coming from the same distributions.

use bytemuck::{Pod, Zeroable};
use murmur_ref::params as p;

/// The GPU evening state. 48 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ArcState {
    /// Preferred flight height \[m\]: the roost level, raised to the reed tops
    /// as the flock pours down.
    pub z0: f32,
    /// Length of one evening \[s\].
    pub duration: f32,
    /// Bearing of the shared approach corridor \[rad\].
    pub bearing: f32,
    /// Radius of the approach corridor \[m\].
    pub radius: f32,

    /// Reserved arrival accumulator (kept for the shared layout).
    pub spawn_timer: f32,
    /// The next free bird slot (the live count while nothing has landed).
    pub next: u32,
    /// Which evening this is, so a wrap resets the flock once.
    pub epoch: u32,
    /// Counter-based random stream (see the module docs).
    pub rng: u32,

    /// Whether the arc is running (1/0).
    pub present: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

impl ArcState {
    /// The record for a run with no arc: `z0` is the only live field, so the
    /// roost level keeps coming from the params and nothing else happens.
    pub fn absent(z0: f32) -> Self {
        Self {
            z0,
            duration: 0.0,
            bearing: 0.0,
            radius: 0.0,
            spawn_timer: 0.0,
            next: 0,
            epoch: 0,
            rng: 0,
            present: 0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    /// An evening of `minutes` minutes. Mirrors `Arc::new`, with the bearing and
    /// radius drawn from the GPU's own stream.
    pub fn new(minutes: f32, seed: u64, z0: f32) -> Self {
        let mut rng = crate::bird::initial_rng(seed, ARC_STREAM_ID);
        let bearing = next_uniform(&mut rng) * std::f32::consts::TAU;
        let radius = 180.0 + next_uniform(&mut rng) * 20.0;
        Self {
            z0,
            duration: (minutes * 60.0).max(1.0),
            bearing,
            radius,
            spawn_timer: 0.0,
            next: 0,
            epoch: 0,
            rng,
            present: 1,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    /// Whether the evening is running.
    pub fn is_present(&self) -> bool {
        self.present != 0
    }
}

/// The counter-based stream id the arc draws from, kept away from the birds'
/// and the falcon's so the three never share a stream.
const ARC_STREAM_ID: u32 = 0x00A2_C0FE;

/// The next uniform in \[0, 1) from a counter-based stream, matching
/// `next_uniform` in `shaders/arc.wgsl`.
fn next_uniform(rng: &mut u32) -> f32 {
    let r = rng.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut w = ((r >> ((r >> 28) + 4)) ^ r).wrapping_mul(277_803_737);
    w = (w >> 22) ^ w;
    *rng = r;
    w as f32 * (1.0 / 4_294_967_296.0)
}

/// The reference's base roost height, so a test can check the arc's own copy.
pub const BASE_Z0: f32 = p::Z0;
