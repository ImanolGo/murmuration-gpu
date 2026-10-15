//! Export and import of the full simulation state, so the CPU oracle and the
//! GPU simulation can start from the same birds (PLAN.md phase 1).
//!
//! PLAN.md sketches `export(state) -> Vec<Bird>` and `import(&[Bird])`. A bare
//! `Vec<Bird>` cannot carry the world clock, `z0` or the model knobs, and the
//! clock is needed to resume exactly (decisions compare `world.time` against
//! each bird's `next_decision`, and the escape timers are absolute). So the
//! unit is a [`Snapshot`]: the birds plus the scalar world state. `export` and
//! `import` are the thin wrappers over [`crate::World::snapshot`] and
//! [`crate::World::restore`].

use glam::Vec3;
use rand_chacha::ChaCha8Rng;

use crate::escape::EscapeState;
use crate::flight::Turning;

/// One bird's complete dynamic state, including its random stream.
///
/// The per-bird [`ChaCha8Rng`] carries its position in the stream, so a
/// snapshot taken mid-run resumes without replaying draws. The GPU side will
/// translate this into its own step-keyed generator in PLAN.md phase 2.
#[derive(Clone, Debug)]
pub struct Bird {
    /// Position \[m\].
    pub pos: Vec3,
    /// Velocity [m/s].
    pub vel: Vec3,
    /// Bank angle \[rad\].
    pub roll: f32,
    /// Bank rate [rad/s].
    pub spin: f32,
    /// Wingbeat phase \[rad\].
    pub flap_phase: f32,
    /// Wingbeat amplitude multiplier.
    pub flap_amp: f32,
    /// Adaptive interaction radius \[m\].
    pub radius: f32,
    /// Held steering force \[N\].
    pub steering: Vec3,
    /// Simulated time of the next decision \[s\].
    pub next_decision: f32,
    /// Neighbour count from the last decision.
    pub neighbour_count: u32,
    /// Centrality from the last decision.
    pub centrality: f32,
    /// Committed to a return turn.
    pub returning: bool,
    /// Escape state (alarm, zigzag, flash).
    pub escape: EscapeState,
    /// The bird's random stream, restored exactly so the run continues.
    pub rng: ChaCha8Rng,
}

/// A whole-flock snapshot: the birds plus the scalar world state.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Simulated time \[s\].
    pub time: f32,
    /// Preferred flight height \[m\].
    pub z0: f32,
    /// The seed the world was created with.
    pub seed: u64,
    /// Target flock size for the arc.
    pub target_birds: usize,
    /// Zigzag copy threshold.
    pub zigzag_k: u32,
    /// Bank model.
    pub turning: Turning,
    /// The birds.
    pub birds: Vec<Bird>,
}

/// Exports the state as a snapshot.
pub fn export(world: &crate::World) -> Snapshot {
    world.snapshot()
}

/// Imports a snapshot into a fresh world.
pub fn import(snapshot: &Snapshot) -> crate::World {
    crate::World::restore(snapshot)
}
