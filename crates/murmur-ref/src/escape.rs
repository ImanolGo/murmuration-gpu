//! Threat detection, fleeing, flash expansion and the zigzag manoeuvre.
//!
//! Follows Hemelrijk, van Zuidam & Hildenbrandt (2015): an alarmed bird rolls
//! one way and then the other, and neighbours copy the manoeuvre, so a wave of
//! agitation (a "dark band") travels through the flock. Mirrors
//! `murmuration::sim::escape`. The agitation-wave experiment is deferred to
//! PLAN.md phase 7; only the model state and forces live here.

use glam::Vec3;

use crate::falcon::Falcon;
use crate::params::*;

/// Per-bird escape state.
#[derive(Clone, Copy, Debug)]
pub struct EscapeState {
    /// Whether the falcon is within `R_THREAT`.
    pub alarmed: bool,
    /// Start time of the current zigzag, or `-inf`.
    pub zigzag_start: f32,
    /// Whether a zigzag is running.
    pub zigzag_active: bool,
    /// Which way the zigzag first rolls (+1 or −1).
    pub zigzag_sign: f32,
    /// Time until which no new zigzag may start.
    pub cooldown_until: f32,
    /// Time until which the flash expansion is applied.
    pub flash_until: f32,
}

impl Default for EscapeState {
    fn default() -> Self {
        Self {
            alarmed: false,
            zigzag_start: f32::NEG_INFINITY,
            zigzag_active: false,
            zigzag_sign: 1.0,
            cooldown_until: f32::NEG_INFINITY,
            flash_until: f32::NEG_INFINITY,
        }
    }
}

/// Whether `p` is within the falcon's threat range.
pub fn is_alarmed(p: Vec3, falcon: Option<&Falcon>) -> bool {
    match falcon {
        Some(f) => (p - f.position).length() < R_THREAT,
        None => false,
    }
}

/// The flee (and flash-expansion) force on a bird at `p`.
pub fn force(p: Vec3, falcon: Option<&Falcon>, flash_until: f32, time: f32) -> Vec3 {
    let Some(f) = falcon else {
        return Vec3::ZERO;
    };
    let away = p - f.position;
    let dist = away.length();
    if dist >= R_THREAT {
        return Vec3::ZERO;
    }
    let dir = away / dist.max(1e-6);
    let mut force = dir * (W_FLEE * (1.0 - dist / R_THREAT));
    if time < flash_until {
        force += dir * W_FLASH;
    }
    force
}

/// The forced roll during a zigzag, or `None` when no zigzag is running.
pub fn zigzag_roll(state: &EscapeState, time: f32) -> Option<f32> {
    if !state.zigzag_active {
        return None;
    }
    let t = time - state.zigzag_start;
    if !(0.0..ZIGZAG_DURATION).contains(&t) {
        return None;
    }
    let phase = std::f32::consts::TAU * t / ZIGZAG_DURATION;
    Some(state.zigzag_sign * ZIGZAG_MAX_ROLL * phase.sin())
}
