//! The social and environmental forces a bird sums into its steering force.
//!
//! Separation, centrality and cohesion, alignment and the roost pull follow the
//! StarDisplay equations (Eq. 4–14), with all-round cohesion and committed
//! roost returns (see `docs/model.md`). A random force stands in for perception
//! and control noise. Mirrors `murmuration::sim::social`.

use glam::Vec3;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::escape::{self, EscapeState};
use crate::falcon::Falcon;
use crate::flight::{forward, side_of};
use crate::neighbours::Grid;
use crate::params::*;

/// The result of one bird's decision: its new steering force and interaction
/// radius, plus the neighbourhood statistics used for analysis.
pub struct Decision {
    /// Steering force held until the next decision \[N\].
    pub force: Vec3,
    /// Adaptive interaction radius for the next interval \[m\].
    pub radius: f32,
    /// Number of neighbours within the current radius `|N_i|`.
    pub neighbours: u32,
    /// Centrality `C_i` (Eq. 7): near 0 inside the flock, ~0.5–0.75 at the edge.
    pub centrality: f32,
    /// Sign of a zigzag to start now (+1 / −1), if the copying rule fired.
    pub start_zigzag: Option<f32>,
    /// Complete the return turn rather than balancing on the roost boundary.
    pub returning: bool,
}

/// Computes bird `i`'s steering force and next interaction radius.
#[allow(clippy::too_many_arguments)]
pub fn decide(
    i: usize,
    positions: &[Vec3],
    velocities: &[Vec3],
    radii: &[f32],
    grid: &Grid,
    rng: &mut ChaCha8Rng,
    z0: f32,
    falcon: Option<&Falcon>,
    escape: &[EscapeState],
    time: f32,
    zigzag_k: u32,
    returning: &[bool],
    w_xi: f32,
    arc_active: bool,
) -> Decision {
    let p = positions[i];
    let r = radii[i];
    let f_i = forward(velocities[i]);

    let mut sep = Vec3::ZERO;
    let mut coh = Vec3::ZERO;
    let mut align = Vec3::ZERO;
    let mut n_all = 0u32;
    let mut n_star = 0u32;
    let mut n_align = 0u32;
    let mut zigzag_neighbours = 0u32;
    let mut zigzag_sign_sum = 0.0f32;
    let mut returning_neighbours = 0u32;

    // One pass over the topological neighbourhood (Eq. 3).
    grid.for_each_within(p, r, positions, |j, dist| {
        if j == i {
            return;
        }
        n_all += 1;
        returning_neighbours += u32::from(returning[j]);
        let dhat = (positions[j] - p) / dist.max(1e-6);

        // Separation over all neighbours, no blind angle (Eq. 4).
        let g = if dist <= R_H {
            1.0
        } else {
            (-(dist - R_H).powi(2) / (2.0 * SIGMA * SIGMA)).exp()
        };
        sep -= dhat * g;

        // Cohesion uses spatial awareness on all sides, like separation. Using
        // only the forward hemisphere pulls even interior birds along their
        // heading and steadily stretches straight-flying flocks into filaments.
        // Alignment retains the rear blind sector.
        if dist > R_H {
            coh += dhat;
            n_star += 1;
        }

        // Alignment uses the reduced neighbourhood N*_i (Eq. 6).
        if f_i.dot(dhat) >= BLIND_COS {
            align += forward(velocities[j]);
            n_align += 1;
        }

        // Neighbours whose zigzag is old enough to have been perceived. The
        // lower bound is what stops the wave hopping one neighbour per 5 ms
        // integration step; a manoeuvre must be at least one reaction time old.
        let age = time - escape[j].zigzag_start;
        if escape[j].zigzag_active && age >= DU {
            zigzag_neighbours += 1;
            zigzag_sign_sum += escape[j].zigzag_sign;
        }
    });

    let mut force = Vec3::ZERO;
    if n_all > 0 {
        force += sep * (W_SEP / n_all as f32);
    }

    // Centrality is measured over a wider radius, 2R_i (Eq. 7).
    let mut csum = Vec3::ZERO;
    let mut n_g = 0u32;
    grid.for_each_within(p, 2.0 * r, positions, |j, dist| {
        if j == i {
            return;
        }
        csum += (positions[j] - p) / dist.max(1e-6);
        n_g += 1;
    });
    let centrality = if n_g > 0 {
        csum.length() / n_g as f32
    } else {
        0.0
    };

    // Cohesion, stronger at the border through C_i (Eq. 5). During an evening
    // display (`arc_active`), longitudinal cohesion along the flight axis is
    // relaxed and boundary birds pick up a slow spatial wave + spontaneous edge
    // turns, so the flock folds into sweeping ribbons, sheets and dark turning
    // bands even without a falcon.
    if n_star > 0 {
        let mut coh_dir = coh.normalize_or_zero();
        if arc_active {
            coh_dir -= f_i * (coh_dir.dot(f_i) * 0.68);
        }
        force += coh_dir * (W_COH * centrality);
    }
    if arc_active {
        let side = side_of(f_i);
        let phase_a = p.x * 0.048 + p.y * 0.038 - time * 0.72;
        let phase_b = p.x * 0.028 - p.y * 0.052 + time * 0.48;
        let wave_lat = phase_a.sin() * 0.65 + phase_b.cos() * 0.35;
        let wave_vert = (phase_b * 0.9 + p.z * 0.06).sin();
        let edge_weight = (centrality - 0.14).max(0.0);
        force += side * (wave_lat * 0.36 * (0.40 + edge_weight))
            + Vec3::Z * (wave_vert * 0.12 * (0.25 + edge_weight));
    }

    // Alignment (Eq. 8): proportional to the misalignment from the mean
    // neighbour heading. Normalising this made a nearly-aligned bird receive a
    // full-strength kick in a random direction, which was the "neighbour noise"
    // that saturated the bank.
    let a = if n_align > 0 {
        align / n_align as f32 - f_i
    } else {
        Vec3::ZERO
    };
    force += a * W_ALIGN;

    // A committed return extends Eq. 11 rather than admitting its stable
    // tangential orbit. Nearby birds can anticipate the return through their
    // neighbours, so a wide flock does not turn only one boundary slice at a
    // time.
    let horiz = Vec3::new(p.x, p.y, 0.0);
    let dist_h = horiz.length();
    let near_boundary = dist_h > R_ROOST * 0.8;
    let heading_out = dist_h > 1.0 && f_i.dot(horiz / dist_h) > -0.8;
    let mut returning = returning[i]
        || dist_h > R_ROOST
        || (near_boundary && heading_out && returning_neighbours >= 2);
    if returning && dist_h > 1.0 {
        let n = horiz / dist_h;
        let side = side_of(f_i);
        // Choose a turning direction, not a radial projection. A signed lateral
        // request carries the turn through until the bird heads back over the
        // roost.
        let inward_side = side * if side.dot(-n) >= 0.0 { 1.0 } else { -1.0 };
        if f_i.dot(n) < -0.95 {
            returning = false;
        } else {
            force += inward_side * (W_ROOST_H * (dist_h - R_ROOST).max(ROOST_TURN_MARGIN));
            if arc_active {
                let sweep_vert = (p.x * 0.032 - p.y * 0.028 + time * 0.65).sin();
                force += Vec3::Z * (sweep_vert * 0.11);
            }
        }
    }

    // Vertical roost pull toward the preferred height (Eq. 12), capped at
    // `W_ROOST_V` so it is not an arbitrarily stiff spring.
    force += Vec3::Z * (W_ROOST_V * ((z0 - p.z) / H_ROOST_V).clamp(-1.0, 1.0));

    // Speed control: relax toward the cruise speed (Eq. 1, part of Eq. 14).
    let speed = velocities[i].length();
    force += f_i * ((MASS / TAU) * (V0 - speed));

    // Random force (Eq. 13). `w_xi` is the model's `W_XI`; it is a parameter
    // only so the CPU/GPU single-step parity check can switch the noise off on
    // both sides at once (see `docs/model.md`).
    force += random_unit(rng) * w_xi;

    // Flee and flash expansion.
    force += escape::force(p, falcon, escape[i].flash_until, time);

    // Copy a zigzag: a non-alarmed bird starts one when at least `zigzag_k` of
    // its neighbours are in a copyable zigzag. It takes the roll direction of
    // the majority of those neighbours, so the dark band is coherent rather
    // than noise. During an evening display, border birds also occasionally
    // initiate a spontaneous turn wave that propagates through the flock.
    let start_zigzag =
        if !escape[i].alarmed && !escape[i].zigzag_active && time >= escape[i].cooldown_until {
            if zigzag_neighbours >= zigzag_k {
                Some(if zigzag_sign_sum > 0.0 {
                    1.0
                } else if zigzag_sign_sum < 0.0 {
                    -1.0
                } else if rng.random::<bool>() {
                    1.0
                } else {
                    -1.0
                })
            } else if arc_active && centrality > 0.38 && rng.random::<f32>() < 0.0011 {
                let phase = (p.x * 0.035 - p.y * 0.035 + time * 0.55).sin();
                Some(if phase >= 0.0 { 1.0 } else { -1.0 })
            } else {
                None
            }
        } else {
            None
        };

    Decision {
        force,
        radius: radius_step(r, n_all),
        neighbours: n_all,
        centrality,
        start_zigzag,
        returning,
    }
}

/// A uniformly distributed unit vector.
pub fn random_unit(rng: &mut ChaCha8Rng) -> Vec3 {
    let z: f32 = rng.random_range(-1.0..1.0);
    let theta: f32 = rng.random_range(0.0..std::f32::consts::TAU);
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * theta.cos(), r * theta.sin(), z)
}
