//! Every model constant, with its source in a doc comment.
//!
//! This is the phase 1 reimplementation of the published model in the sibling
//! `murmuration` project (MIT). Values and names follow Table 1 of
//! Hildenbrandt, Carere & Hemelrijk (2010), "Self-organized aerial displays of
//! thousands of starlings: a model" (arXiv:0908.2677), unless a doc comment
//! names a documented adaptation. The adaptations are recorded once in
//! `docs/model.md`; the numbers here must not drift from the reference.

/// Integration time step Δt \[s\].
pub const DT: f32 = 0.005;

/// Reaction time Δu \[s\]: steering is held this long between decisions.
pub const DU: f32 = 0.05;

/// Cruise speed v0 [m/s].
pub const V0: f32 = 10.0;

/// Body mass m \[kg\].
pub const MASS: f32 = 0.08;

/// Lift-to-drag ratio C_L/C_D.
pub const CL_CD: f32 = 3.3;

/// Standard gravity [m/s²].
pub const G: f32 = 9.81;

/// Default lift `L0 = m·g` \[N\]; cruise flight balances the weight.
pub const L0: f32 = MASS * G;

/// Default drag, equal to default thrust, `D0 = T0 = L0/(C_L/C_D)` \[N\].
pub const D0: f32 = L0 / CL_CD;

/// Speed relaxation time τ \[s\].
pub const TAU: f32 = 1.0;

/// Time constant of the roll response \[s\]. The paper's Eq. 19–21 integrate
/// the roll in angle directly and pin it at the clamp; murmururation replaces
/// them with a first-order lag toward `atan(a_lat/g)`. See `docs/model.md`.
pub const TAU_ROLL: f32 = 0.15;

/// Maximum roll rate [rad/s], so the roll cannot snap between extremes.
pub const MAX_ROLL_RATE: f32 = 6.0;

/// Time constant of the roll *rate* in the inertial turning mode \[s\].
pub const TAU_SPIN: f32 = 0.3;

/// Bank angle limit \[rad\]. Not in the paper; a numerical safeguard.
pub const MAX_BANK: f32 = 80.0 * std::f32::consts::PI / 180.0;

/// Speed is kept within `[V_MIN, V_MAX]`.
pub const V_MIN: f32 = 0.5 * V0;
/// See [`V_MIN`].
pub const V_MAX: f32 = 2.0 * V0;

/// Topological range n_c: the number of interaction partners a bird strives
/// for.
pub const N_C: f32 = 6.5;

/// Max perception radius R_max \[m\].
pub const R_MAX: f32 = 100.0;

/// Interpolation factor s for the adaptive radius, per reaction ("0.1 Δu").
pub const RADIUS_LERP: f32 = 0.1;

/// Radius of maximal separation, the "hard sphere" r_h \[m\].
pub const R_H: f32 = 0.2;

/// Separation radius r_sep \[m\].
pub const R_SEP: f32 = 4.0;

/// Standard deviation σ of the separation Gaussian \[m\].
pub const SIGMA: f32 = 1.37;

/// Weighting factor for the separation force \[N\].
pub const W_SEP: f32 = 1.0;

/// Weighting factor for the alignment force \[N\].
pub const W_ALIGN: f32 = 0.5;

/// Weighting factor for unit-direction, all-round cohesion \[N\]. Adapted from
/// Table 1's 1 N to balance the corrected separation and banked flight; see
/// `docs/model.md`.
pub const W_COH: f32 = 0.65;

/// Cosine of the rear blind angle for cohesion and alignment. The blind sector
/// is `2 × 45°`, so a neighbour is blind when `f·d̂ < −cos 45°`.
pub const BLIND_COS: f32 = -std::f32::consts::FRAC_1_SQRT_2;

/// Weighting factor for the random force \[N\].
pub const W_XI: f32 = 0.01;

/// Roost boundary radius R_roost \[m\].
pub const R_ROOST: f32 = 150.0;

/// Weighting factor for the horizontal roost force [N/m].
pub const W_ROOST_H: f32 = 0.01;

/// Minimum effective overshoot while a bird completes its return turn \[m\].
pub const ROOST_TURN_MARGIN: f32 = 25.0;

/// Weighting factor for the vertical roost force \[N\].
pub const W_ROOST_V: f32 = 0.2;

/// Distance over which the vertical roost force ramps up to its cap \[m\].
pub const H_ROOST_V: f32 = 40.0;

/// Preferred flight height above the roost, the "zero level" z0 \[m\].
pub const Z0: f32 = 100.0;

/// Reeds are at the bottom of the world; the arc lands birds below this.
pub const REED_TOP: f32 = 3.0;

/// Uniform grid cell size for neighbour search \[m\].
pub const GRID_CELL: f32 = 4.0;

// --- Escape and predation (Hemelrijk et al. 2015) ---

/// A bird is alarmed when the falcon is within this distance \[m\].
pub const R_THREAT: f32 = 30.0;

/// Weighting of the flee force away from the falcon \[N\].
pub const W_FLEE: f32 = 2.0;

/// Flash expansion triggers inside this distance \[m\].
pub const R_FLASH: f32 = 8.0;

/// ...and when the falcon is moving faster than this [m/s].
pub const V_FLASH: f32 = 20.0;

/// Strength of the flash-expansion impulse \[N\].
pub const W_FLASH: f32 = 12.0;

/// How long the flash expansion lasts \[s\].
pub const FLASH_DURATION: f32 = 0.3;

/// Peak zigzag bank angle \[rad\] (target roll ±50°).
pub const ZIGZAG_MAX_ROLL: f32 = 50.0 * std::f32::consts::PI / 180.0;

/// Duration of one zigzag manoeuvre \[s\].
pub const ZIGZAG_DURATION: f32 = 0.6;

/// A bird cannot start another zigzag for this long \[s\].
pub const ZIGZAG_COOLDOWN: f32 = 1.5;

/// Default number of neighbours whose zigzag starts a copy (1–7).
pub const ZIGZAG_K: u32 = 2;

// --- Wingbeat (rendering shimmer) ---

/// Wingbeat frequency \[Hz\].
pub const FLAP_HZ: f32 = 10.0;

/// Fraction by which the apparent wing exposure is modulated by the flap.
pub const FLAP_AMPLITUDE: f32 = 0.15;

// --- Falcon ---

/// Cruising speed of the falcon outside the flock [m/s].
pub const FALCON_CRUISE: f32 = 15.0;
/// Climb speed [m/s].
pub const FALCON_CLIMB: f32 = 15.0;
/// Stooping speed through the flock [m/s].
pub const FALCON_STOOP: f32 = 30.0;
/// Height above the flock the falcon climbs before stooping \[m\].
pub const FALCON_ATTACK_HEIGHT: f32 = 40.0;
/// How often the autonomous falcon attacks [s, min..max].
pub const FALCON_ATTACK_MIN: f32 = 5.0;
/// See [`FALCON_ATTACK_MIN`].
pub const FALCON_ATTACK_MAX: f32 = 20.0;

/// Advances an adaptive interaction radius toward `n_c` partners.
///
/// Relaxes geometrically toward the radius that would give `n_c` partners,
/// assuming `|N| ∝ R³`, so `R` moves by `(n_c/|N|)^(s/3)` per reaction:
/// `|N| = n_c` is a fixed point and the rate stays bounded for an almost
/// isolated bird. This is a stable form of the paper's stated intent, whose
/// Eq. 2 as printed is unstable for the Table 1 values; see `docs/model.md`.
pub fn radius_step(r: f32, n: u32) -> f32 {
    let n = n.max(1) as f32;
    let factor = (N_C / n).powf(RADIUS_LERP / 3.0);
    (r * factor).clamp(R_H, R_MAX)
}
