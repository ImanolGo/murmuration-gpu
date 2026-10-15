//! Fixed-wing flight: lift, drag, thrust, gravity and banked turning.
//!
//! Turning comes from banked lift rather than from rotating the heading
//! directly. Mirrors `murmuration::sim::flight`; the GPU vertex/compute code
//! mirrors [`step_bird`] operation by operation (PLAN.md phase 4).

use glam::Vec3;

use crate::params::*;

/// How a bird's bank responds to the steering force.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Turning {
    /// The bank tracks `atan(a_lat/g)` with a first-order lag.
    #[default]
    Base,
    /// The bank *rate* has momentum, after the inertial spin model (Cavagna et
    /// al. 2015): once a bird starts to roll it keeps rolling.
    Inertial,
}

/// Unit forward direction of a velocity, with a stable fallback.
pub fn forward(v: Vec3) -> Vec3 {
    if v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        Vec3::X
    }
}

/// The bird's sideways direction: `normalize(f × ẑ)`, with a fallback when
/// `f` is near vertical.
pub fn side_of(f: Vec3) -> Vec3 {
    let s = f.cross(Vec3::Z);
    if s.length_squared() > 1e-8 {
        s.normalize()
    } else {
        Vec3::X
    }
}

/// The bird's body "up" for a bank angle `beta`, given forward `f`.
///
/// The body frame is `(f, side, up)`; banking rotates `up0 = side × f` about
/// `f` by `beta`, which simplifies to `up0·cos β + side·sin β`.
pub fn body_up(f: Vec3, beta: f32) -> Vec3 {
    let side = side_of(f);
    let up0 = side.cross(f);
    up0 * beta.cos() + side * beta.sin()
}

/// Advances one bird by `dt` under its held steering force. When
/// `forced_roll` is `Some`, that bank angle overrides the steering-driven
/// banking (used by the zigzag manoeuvre).
#[allow(clippy::too_many_arguments)]
pub fn step_bird(
    p: &mut Vec3,
    v: &mut Vec3,
    roll: &mut f32,
    spin: &mut f32,
    steering: Vec3,
    forced_roll: Option<f32>,
    dt: f32,
    turning: Turning,
) {
    let f = forward(*v);
    let side = side_of(f);
    let speed = v.length().max(1e-6);

    // Coordinated turn: bank toward `atan(a_lat / g)`, with a first-order lag
    // and a maximum roll rate. This replaces the paper's Eq. 19–21, which
    // applied per integration step is a high-gain integrator that pins the roll
    // at the clamp (see `docs/model.md`).
    let a_lat = steering.dot(side) / MASS;
    let beta = match forced_roll {
        Some(target) => target.clamp(-MAX_BANK, MAX_BANK),
        None => {
            let target = (a_lat / G).atan().clamp(-MAX_BANK, MAX_BANK);
            let rate = ((target - *roll) / TAU_ROLL).clamp(-MAX_ROLL_RATE, MAX_ROLL_RATE);
            match turning {
                Turning::Base => {
                    *spin = rate;
                    (*roll + rate * dt).clamp(-MAX_BANK, MAX_BANK)
                }
                Turning::Inertial => {
                    // The roll rate has momentum: it eases toward the target
                    // rate with a second time constant.
                    *spin += (rate - *spin) * (dt / TAU_SPIN);
                    (*roll + *spin * dt).clamp(-MAX_BANK, MAX_BANK)
                }
            }
        }
    };
    *roll = beta;

    let up = body_up(f, beta);

    // Lift (Eq. 16) along the body up, divided by `cos β` so a banked bird
    // holds its height.
    let cos_beta = beta.cos().max(0.2);
    let lift = (speed / V0).powi(2) * L0 / cos_beta;
    // Drag (Eq. 16) against the direction of travel.
    let drag = (speed / V0).powi(2) * D0;
    // Default thrust along the forward direction. The speed-control force is
    // part of the steering force.
    let flight_force = up * lift - f * drag + f * D0 - Vec3::Z * MASS * G;

    // Euler integration (Eq. 22–23).
    // Lateral steering requests a bank; the banked lift supplies that force.
    // Adding it directly as well doubled the turn and let birds slide sideways
    // before their wings had rolled. Forward and vertical control remain direct.
    let control = steering - side * steering.dot(side);
    *v += (control + flight_force) / MASS * dt;
    let sp = v.length();
    if sp > V_MAX {
        *v *= V_MAX / sp;
    } else if sp < V_MIN && sp > 1e-6 {
        *v *= V_MIN / sp;
    }
    *p += *v * dt;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lateral_request_does_not_slide_before_banking() {
        let mut p = Vec3::new(0.0, 0.0, 100.0);
        let mut v = Vec3::X * V0;
        let mut roll = 0.0;
        let mut spin = 0.0;
        step_bird(
            &mut p,
            &mut v,
            &mut roll,
            &mut spin,
            Vec3::Y * 0.2,
            None,
            DT,
            Turning::Base,
        );
        // The first step has only the small lift component from a partial roll,
        // not an instantaneous 0.2 N sideways kick as well.
        assert!(v.y > 0.0 && v.y < 0.002, "sideways slip: {v:?}");
    }

    #[test]
    fn level_cruise_holds_height() {
        let mut p = Vec3::new(0.0, 0.0, 100.0);
        let mut v = Vec3::X * V0;
        let mut roll = 0.0;
        let mut spin = 0.0;
        for _ in 0..200 {
            step_bird(
                &mut p,
                &mut v,
                &mut roll,
                &mut spin,
                Vec3::ZERO,
                None,
                DT,
                Turning::Base,
            );
        }
        assert!((p.z - 100.0).abs() < 1.0, "sank or climbed: z = {}", p.z);
        assert!(
            (v.length() - V0).abs() < 0.5,
            "speed drifted: {}",
            v.length()
        );
    }

    #[test]
    fn banked_bird_turns() {
        // A sustained sideways steering force should curl the path.
        let mut p = Vec3::new(0.0, 0.0, 100.0);
        let mut v = Vec3::X * V0;
        let mut roll = 0.0;
        let mut spin = 0.0;
        for _ in 0..400 {
            step_bird(
                &mut p,
                &mut v,
                &mut roll,
                &mut spin,
                Vec3::Y * 0.5,
                None,
                DT,
                Turning::Base,
            );
        }
        assert!(roll.abs() > 0.05, "did not bank: beta = {roll}");
        assert!(v.y.abs() > 1.0, "did not turn: v = {v}");
    }

    #[test]
    fn inertial_banking_also_turns() {
        let mut p = Vec3::new(0.0, 0.0, 100.0);
        let mut v = Vec3::X * V0;
        let mut roll = 0.0;
        let mut spin = 0.0;
        for _ in 0..400 {
            step_bird(
                &mut p,
                &mut v,
                &mut roll,
                &mut spin,
                Vec3::Y * 0.5,
                None,
                DT,
                Turning::Inertial,
            );
        }
        assert!(roll.abs() > 0.05, "did not bank: beta = {roll}");
        assert!(v.y.abs() > 1.0, "did not turn: v = {v}");
    }
}
