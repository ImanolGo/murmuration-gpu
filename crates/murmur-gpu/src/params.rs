//! The one uniform block shared by Rust and WGSL.
//!
//! Every model constant lives once in [`murmur_ref::params`]; this struct
//! carries them to the shader so the two sides cannot drift (AGENTS.md: "Params
//! and Bird are defined once in Rust and checked against WGSL by a test"). The
//! values are copied from the reference constants by [`Params::new`]; nothing
//! here may be tuned separately. `tests/layout.rs` checks the field names,
//! order and offsets against `shaders/sim.wgsl`.

use bytemuck::{Pod, Zeroable};

use murmur_ref::flight::Turning;
use murmur_ref::params as p;
use murmur_ref::state::Snapshot;

/// Per-run parameters, mirroring the reference constants plus the clock and the
/// spatial grid's shape.
///
/// 56 `f32`/`u32` scalars = 224 bytes, a multiple of 16, so the WGSL struct
/// matches field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    // Clock and integration.
    pub dt: f32,
    pub du: f32,
    pub time: f32,
    pub turning: u32,

    // Flight (flight.rs, Table 1).
    pub v0: f32,
    pub mass: f32,
    pub g: f32,
    pub l0: f32,
    pub d0: f32,
    pub tau: f32,
    pub tau_roll: f32,
    pub max_roll_rate: f32,
    pub tau_spin: f32,
    pub max_bank: f32,
    pub v_min: f32,
    pub v_max: f32,

    // Interaction radius (params.rs::radius_step).
    pub n_c: f32,
    pub r_max: f32,
    pub radius_lerp: f32,
    pub r_h: f32,

    // Forces (social.rs).
    pub sigma: f32,
    pub w_sep: f32,
    pub w_align: f32,
    pub w_coh: f32,
    pub blind_cos: f32,
    pub w_xi: f32,
    pub r_roost: f32,
    pub w_roost_h: f32,
    pub roost_turn_margin: f32,
    pub w_roost_v: f32,
    pub h_roost_v: f32,
    pub z0: f32,

    // Wingbeat and escape (escape.rs; the escape forces stay inert in phase 2).
    pub flap_hz: f32,
    pub flap_amplitude: f32,
    pub r_threat: f32,
    pub w_flee: f32,
    pub r_flash: f32,
    pub v_flash: f32,
    pub w_flash: f32,
    pub flash_duration: f32,
    pub zigzag_max_roll: f32,
    pub zigzag_duration: f32,
    pub zigzag_cooldown: f32,
    pub reed_top: f32,

    // Runtime.
    pub n: u32,
    pub zigzag_k: u32,
    pub seed: u32,
    /// Number of hash buckets in the spatial grid (a power of two).
    pub buckets: u32,

    /// `1 / grid::CELL`, the GPU grid's cell size the shader inverts. This is a
    /// GPU-only acceleration parameter, not the oracle's `GRID_CELL`; the
    /// neighbour set is fixed by the exact distance check, so the two never
    /// disagree about which birds interact (see [`crate::grid::CELL`]).
    pub inv_cell: f32,
    /// Falcon cruise speed (falcon.rs `FALCON_CRUISE`).
    pub falcon_cruise: f32,
    /// Falcon climb speed (`FALCON_CLIMB`).
    pub falcon_climb: f32,
    /// Falcon stoop speed (`FALCON_STOOP`).
    pub falcon_stoop: f32,
    /// Height above the flock the falcon climbs to (`FALCON_ATTACK_HEIGHT`).
    pub falcon_attack_height: f32,
    /// Shortest autonomous attack interval (`FALCON_ATTACK_MIN`).
    pub falcon_attack_min: f32,
    /// Longest autonomous attack interval (`FALCON_ATTACK_MAX`).
    pub falcon_attack_max: f32,
    pub pad3: u32,
}

/// Map the reference [`Turning`] to the shader's `u32` discriminant.
pub fn turning_code(turning: Turning) -> u32 {
    match turning {
        Turning::Base => 0,
        Turning::Inertial => 1,
    }
}

/// Inverse of [`turning_code`], for reading a snapshot back out of the GPU.
pub fn turning_from_code(code: u32) -> Turning {
    if code == 1 {
        Turning::Inertial
    } else {
        Turning::Base
    }
}

impl Params {
    /// Builds the block for an `n`-bird run seeded with `seed`, from the
    /// reference constants. `w_xi` starts at the model value.
    pub fn new(n: usize, seed: u64) -> Self {
        Self {
            dt: p::DT,
            du: p::DU,
            time: 0.0,
            turning: turning_code(Turning::Base),

            v0: p::V0,
            mass: p::MASS,
            g: p::G,
            l0: p::L0,
            d0: p::D0,
            tau: p::TAU,
            tau_roll: p::TAU_ROLL,
            max_roll_rate: p::MAX_ROLL_RATE,
            tau_spin: p::TAU_SPIN,
            max_bank: p::MAX_BANK,
            v_min: p::V_MIN,
            v_max: p::V_MAX,

            n_c: p::N_C,
            r_max: p::R_MAX,
            radius_lerp: p::RADIUS_LERP,
            r_h: p::R_H,

            sigma: p::SIGMA,
            w_sep: p::W_SEP,
            w_align: p::W_ALIGN,
            w_coh: p::W_COH,
            blind_cos: p::BLIND_COS,
            w_xi: p::W_XI,
            r_roost: p::R_ROOST,
            w_roost_h: p::W_ROOST_H,
            roost_turn_margin: p::ROOST_TURN_MARGIN,
            w_roost_v: p::W_ROOST_V,
            h_roost_v: p::H_ROOST_V,
            z0: p::Z0,

            flap_hz: p::FLAP_HZ,
            flap_amplitude: p::FLAP_AMPLITUDE,
            r_threat: p::R_THREAT,
            w_flee: p::W_FLEE,
            r_flash: p::R_FLASH,
            v_flash: p::V_FLASH,
            w_flash: p::W_FLASH,
            flash_duration: p::FLASH_DURATION,
            zigzag_max_roll: p::ZIGZAG_MAX_ROLL,
            zigzag_duration: p::ZIGZAG_DURATION,
            zigzag_cooldown: p::ZIGZAG_COOLDOWN,
            reed_top: p::REED_TOP,

            n: n as u32,
            zigzag_k: p::ZIGZAG_K,
            seed: seed as u32,
            buckets: crate::grid::buckets_for(n),

            inv_cell: 1.0 / crate::grid::CELL,
            falcon_cruise: p::FALCON_CRUISE,
            falcon_climb: p::FALCON_CLIMB,
            falcon_stoop: p::FALCON_STOOP,
            falcon_attack_height: p::FALCON_ATTACK_HEIGHT,
            falcon_attack_min: p::FALCON_ATTACK_MIN,
            falcon_attack_max: p::FALCON_ATTACK_MAX,
            pad3: 0,
        }
    }

    /// Builds the block from an imported snapshot, so a GPU run starts from
    /// exactly the CPU state (clock, seed, knobs).
    pub fn from_snapshot(snapshot: &Snapshot) -> Self {
        let mut params = Self::new(snapshot.birds.len(), snapshot.seed);
        params.time = snapshot.time;
        params.z0 = snapshot.z0;
        params.zigzag_k = snapshot.zigzag_k;
        params.turning = turning_code(snapshot.turning);
        params.w_xi = snapshot.w_xi;
        params
    }

    /// Overrides the random-force weight. Used by the single-step parity check
    /// to switch the noise off on the GPU; the CPU switches it off through
    /// `World::w_xi`. See `docs/model.md`.
    pub fn with_w_xi(mut self, w_xi: f32) -> Self {
        self.w_xi = w_xi;
        self
    }
}
