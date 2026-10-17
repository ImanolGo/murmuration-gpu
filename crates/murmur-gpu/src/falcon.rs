//! The GPU falcon record (PLAN.md phase 7).
//!
//! Mirrors `murmur_ref::falcon::Falcon` field for field and the falcon block of
//! `murmur_ref::world::World::step` function for function; the kernels that
//! advance it are in `crates/murmur-gpu/shaders/falcon.wgsl`, and the struct is
//! checked against `struct Falcon` in `common.wgsl` by `tests/layout.rs`.
//!
//! One documented adaptation: the random stream. The oracle draws its attack
//! intervals from a ChaCha8 stream that cannot run in WGSL, so the GPU draws from
//! the same counter-based PCG as the birds, seeded from the run's seed. Both
//! sides advance it with the same step, so the initial `until` agrees.

use bytemuck::{Pod, Zeroable};

use murmur_ref::params as p;

/// Who is steering the falcon.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Hunts on its own.
    Autonomous,
    /// Chases the point the user points at.
    Mouse,
}

/// The falcon's autonomous behaviour state, mirroring `falcon::State`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Cruise,
    Climb,
    Stoop,
    Retreat,
}

impl Mode {
    /// The shader's `u32` discriminant.
    pub fn code(self) -> u32 {
        match self {
            Mode::Autonomous => 0,
            Mode::Mouse => 1,
        }
    }
}

/// One falcon in the GPU storage buffer. 96 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Falcon {
    /// Position \[m\].
    pub pos: [f32; 3],
    /// Current speed \[m/s\], `|velocity|`, for the flash-expansion test.
    pub speed: f32,

    /// Velocity [m/s].
    pub vel: [f32; 3],
    /// Autonomous (0) or mouse-driven (1).
    pub mode: u32,

    /// Point being chased (mouse mode) or attacked (autonomous mode).
    pub aim: [f32; 3],
    /// Behaviour state (see [`State`]).
    pub state: u32,

    /// Dive direction during a stoop.
    pub dive: [f32; 3],
    /// Time of the next autonomous attack \[s\].
    pub until: f32,

    /// Flock centroid, written by the reduction each step \[m\].
    pub centroid: [f32; 3],
    /// Counter-based random stream (see the module docs).
    pub rng: u32,

    /// Flock extent (max distance from the centroid) \[m\].
    pub extent: f32,
    /// Whether a falcon is hunting (1/0). Zero means every pass is a no-op.
    pub present: u32,
    pub pad0: u32,
    pub pad1: u32,
}

impl Falcon {
    /// The record for a run with no falcon: every field zero, so the shader's
    /// `present` guards make the falcon passes inert.
    pub fn absent() -> Self {
        Self::zeroed()
    }

    /// A falcon cruising near `z0`, attacking every 5–20 s. Mirrors
    /// `falcon::Falcon::new`; only the random stream differs (module docs).
    pub fn spawn(seed: u64, z0: f32) -> Self {
        let mut rng = crate::bird::initial_rng(seed, FALCON_STREAM_ID);
        let until = p::FALCON_ATTACK_MIN
            + next_uniform(&mut rng) * (p::FALCON_ATTACK_MAX - p::FALCON_ATTACK_MIN);
        Self {
            pos: [0.0, -220.0, z0 + 20.0],
            speed: p::FALCON_CRUISE,
            vel: [0.0, p::FALCON_CRUISE, 0.0],
            mode: Mode::Autonomous.code(),
            aim: [0.0, 0.0, z0],
            state: State::Cruise as u32,
            until,
            dive: [0.0, 1.0, 0.0],
            rng,
            centroid: [0.0, 0.0, z0],
            extent: 0.0,
            present: 1,
            pad0: 0,
            pad1: 0,
        }
    }

    /// Whether a falcon is hunting.
    pub fn is_present(&self) -> bool {
        self.present != 0
    }

    /// The position as a vector.
    pub fn position(&self) -> glam::Vec3 {
        glam::Vec3::from_array(self.pos)
    }
}

/// The counter-based stream id the falcon draws from, kept away from the birds'
/// ids so the two never share a stream.
const FALCON_STREAM_ID: u32 = 0x00F0_0DFA;

/// The next uniform in \[0, 1) from a counter-based stream, matching
/// `falcon_next_uniform` in `shaders/falcon.wgsl`.
fn next_uniform(rng: &mut u32) -> f32 {
    let r = rng.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut w = ((r >> ((r >> 28) + 4)) ^ r).wrapping_mul(277_803_737);
    w = (w >> 22) ^ w;
    *rng = r;
    w as f32 * (1.0 / 4_294_967_296.0)
}
