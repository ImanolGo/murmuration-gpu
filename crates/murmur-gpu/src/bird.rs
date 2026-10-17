//! The GPU bird record: the state the shaders read and write.
//!
//! PLAN.md sketches a 64-byte layout for the optimised version. Phase 2 is the
//! brute-force correctness version, so the record is 112 bytes and carries what
//! the mirrored functions actually need: the integrated dynamics (position,
//! velocity, bank, spin, wingbeat), the decision state held between reactions
//! (steering, radius, clock, counts) and the escape state the falcon will use in
//! phase 7. `speed` and `fwd` from the sketch are derived in the shader
//! (`flight::forward`) exactly as the reference derives them, so they are not
//! stored; `id` survives reorders. The layout is checked against
//! `shaders/sim.wgsl` by `tests/layout.rs`.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use murmur_ref::escape::EscapeState;
use murmur_ref::state::Bird as RefBird;
use rand_chacha::rand_core::SeedableRng as _;

/// One bird in the GPU storage buffer. 112 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Bird {
    /// Position \[m\].
    pub pos: [f32; 3],
    /// Bank angle \[rad\].
    pub roll: f32,

    /// Velocity [m/s].
    pub vel: [f32; 3],
    /// Bank rate [rad/s] (used by the inertial turning mode).
    pub spin: f32,

    /// Held steering force \[N\].
    pub steering: [f32; 3],
    /// Adaptive interaction radius \[m\].
    pub radius: f32,

    /// Simulated time of the next decision \[s\].
    pub next_decision: f32,
    /// Centrality `C_i` from the last decision.
    pub centrality: f32,
    /// Wingbeat phase \[rad\].
    pub flap_phase: f32,
    /// Wingbeat amplitude multiplier.
    pub flap_amp: f32,

    /// Neighbour count `|N_i|` from the last decision.
    pub neighbour_count: u32,
    /// Committed to a return turn (1/0).
    pub returning: u32,
    /// The bird's original index, kept across reorders.
    pub id: u32,
    /// Counter-based random stream state (see `shaders/sim.wgsl`).
    pub rng: u32,

    /// Whether the falcon is within `R_THREAT` (1/0).
    pub alarmed: u32,
    /// Whether a zigzag is running (1/0).
    pub zigzag_active: u32,
    /// Zigzag roll direction (+1 / −1).
    pub zigzag_sign: f32,
    /// Start time of the current zigzag \[s\].
    pub zigzag_start: f32,
    /// Time until which no new zigzag may start \[s\].
    pub cooldown_until: f32,
    /// Time until which the flash expansion applies \[s\].
    pub flash_until: f32,
    /// Whether the bird is in the flock (1/0) (`live` in WGSL: `active` is a
    /// reserved word there). The evening arc (PLAN.md phase 8)
    /// spawns arriving groups into spare slots and retires landed birds in
    /// place, so the buffer keeps its size; an inactive bird is skipped by the
    /// neighbour walk, the flight integration and the LOD classifier.
    pub live: u32,
    /// Padding to a 16-byte multiple.
    pub pad1: u32,
}

impl Bird {
    /// Converts one reference bird into its GPU record. The reference's ChaCha
    /// stream cannot be extracted, so the GPU stream is seeded from
    /// `(seed, id)`; see `docs/model.md` for this documented adaptation.
    pub fn from_ref(bird: &RefBird, id: u32, seed: u64) -> Self {
        let e = bird.escape;
        Self {
            pos: bird.pos.to_array(),
            roll: bird.roll,
            vel: bird.vel.to_array(),
            spin: bird.spin,
            steering: bird.steering.to_array(),
            radius: bird.radius,
            next_decision: bird.next_decision,
            centrality: bird.centrality,
            flap_phase: bird.flap_phase,
            flap_amp: bird.flap_amp,
            neighbour_count: bird.neighbour_count,
            returning: u32::from(bird.returning),
            id,
            rng: initial_rng(seed, id),
            alarmed: u32::from(e.alarmed),
            zigzag_active: u32::from(e.zigzag_active),
            zigzag_sign: e.zigzag_sign,
            zigzag_start: e.zigzag_start,
            cooldown_until: e.cooldown_until,
            flash_until: e.flash_until,
            live: 1,
            pad1: 0,
        }
    }

    /// Converts the GPU record back into a reference bird, for analysis.
    ///
    /// The random stream is a placeholder: the analysis reads positions,
    /// velocities, radii and counts, never the generator. A real resume on the
    /// CPU would need the reference's own stream, which the GPU does not keep.
    pub fn to_ref(&self, seed: u64) -> RefBird {
        RefBird {
            pos: Vec3::from_array(self.pos),
            vel: Vec3::from_array(self.vel),
            roll: self.roll,
            spin: self.spin,
            flap_phase: self.flap_phase,
            flap_amp: self.flap_amp,
            radius: self.radius,
            steering: Vec3::from_array(self.steering),
            next_decision: self.next_decision,
            neighbour_count: self.neighbour_count,
            centrality: self.centrality,
            returning: self.returning != 0,
            escape: EscapeState {
                alarmed: self.alarmed != 0,
                zigzag_start: self.zigzag_start,
                zigzag_active: self.zigzag_active != 0,
                zigzag_sign: self.zigzag_sign,
                cooldown_until: self.cooldown_until,
                flash_until: self.flash_until,
            },
            rng: rand_chacha::ChaCha8Rng::seed_from_u64(mix(seed, self.id as u64)),
        }
    }
}

/// SplitMix64, matching `murmur_ref::world`'s per-bird seed derivation.
fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The initial random-stream state for bird `id` of a `seed` run: a 32-bit mix
/// of the two, so birds get independent streams. The shader advances it with a
/// PCG step (`shaders/sim.wgsl`).
pub(crate) fn initial_rng(seed: u64, id: u32) -> u32 {
    let mut h = (seed as u32) ^ ((seed >> 32) as u32).wrapping_mul(0x9E37_79B9);
    h ^= id.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^ (h >> 16)
}
