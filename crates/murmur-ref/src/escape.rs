//! Threat detection, fleeing, flash expansion and the zigzag manoeuvre.
//!
//! Follows Hemelrijk, van Zuidam & Hildenbrandt (2015): an alarmed bird rolls
//! one way and then the other, and neighbours copy the manoeuvre, so a wave of
//! agitation (a "dark band") travels through the flock. Mirrors
//! `murmuration::sim::escape`, including the agitation-wave experiment of
//! PLAN.md phase 7.

use glam::Vec3;

use crate::falcon::Falcon;
use crate::params::*;
use crate::world::World;

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

/// Result of an agitation-wave experiment.
#[derive(Debug, Clone)]
pub struct AgitationWave {
    /// Fitted propagation speed [m/s].
    pub speed: f32,
    /// Coefficient of determination of the fit.
    pub r2: f32,
    /// Birds used in the fit.
    pub samples: usize,
    /// Peak fraction of the flock alarmed during the attack.
    pub peak_alarmed: f32,
    /// Peak fractional increase in flock spread (flash expansion signal).
    pub peak_expansion: f32,
    /// `(distance from the attack, first-roll time)`.
    pub points: Vec<(f32, f32)>,
}

/// Runs an agitation wave and measures how the zigzag copies through the flock.
///
/// Mirrors `murmuration::sim::escape::measure_agitation`. The flock is settled,
/// then a few birds on one side start a zigzag. The wave in nature is seeded by
/// the falcon alarming birds; here it is seeded directly so the measurement
/// isolates the *copying* rule and its threshold `k` from the falcon's own
/// sweep. For every bird we record the first time it starts a zigzag and fit
/// that against its initial distance from the seed.
pub fn measure_agitation(world: &mut World, seconds: f32) -> AgitationWave {
    let n = world.len();
    if n == 0 {
        return empty_wave();
    }
    // Settle without a falcon.
    world.falcon = None;
    let settle = (20.0 / DT).max(1.0) as usize;
    for _ in 0..settle {
        world.step(DT);
    }

    // Seed a compact patch on one side.
    let initial_position: Vec<Vec3> = world.position.clone();
    let centroid = initial_position.iter().copied().sum::<Vec3>() / n as f32;
    let extent = initial_position
        .iter()
        .map(|p| (*p - centroid).length())
        .fold(1.0f32, f32::max);
    let edge = Vec3::new(0.0, -1.0, 0.1).normalize();
    let seed_point = centroid + edge * (extent * 0.8);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        (initial_position[a] - seed_point)
            .length_squared()
            .partial_cmp(&(initial_position[b] - seed_point).length_squared())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let seed_birds = order[..20.min(n)].to_vec();
    let origin = seed_birds
        .iter()
        .map(|&i| initial_position[i])
        .sum::<Vec3>()
        / seed_birds.len() as f32;
    for &i in &seed_birds {
        world.escape[i].zigzag_start = world.time;
        world.escape[i].zigzag_active = true;
        // A real alarm patch is coherent (birds roll away from the falcon), so
        // the seeded manoeuvre is too.
        world.escape[i].zigzag_sign = 1.0;
    }

    let start_time = world.time;
    let steps = (seconds / DT).max(1.0) as usize;
    let mut arrival = vec![f32::INFINITY; n];
    let initial_spread = spread(&world.position);
    let mut peak_spread = initial_spread;

    for _ in 0..steps {
        world.step(DT);
        for (i, a) in arrival.iter_mut().enumerate() {
            let started = world.escape[i].zigzag_start;
            if a.is_infinite() && started >= start_time && started.is_finite() {
                *a = started;
            }
        }
        peak_spread = peak_spread.max(spread(&world.position));
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        if arrival[i].is_finite() {
            xs.push((initial_position[i] - origin).length());
            ys.push(arrival[i] - start_time);
        }
    }
    let (slope, r2) = linfit(&xs, &ys);
    let turned = xs.len() as f32 / n as f32;
    AgitationWave {
        speed: if slope.abs() > 1e-6 {
            1.0 / slope.abs()
        } else {
            f32::NAN
        },
        r2,
        samples: xs.len(),
        peak_alarmed: turned,
        peak_expansion: peak_spread / initial_spread.max(1e-6) - 1.0,
        points: xs.iter().copied().zip(ys.iter().copied()).collect(),
    }
}

/// RMS spread of a flock about its centroid \[m\]. Shared with the GPU
/// experiment, which fits the same line.
pub fn spread(pos: &[Vec3]) -> f32 {
    let n = pos.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let c = pos.iter().copied().sum::<Vec3>() / n;
    let rms = pos.iter().map(|p| (*p - c).length_squared()).sum::<f32>() / n;
    rms.sqrt()
}

/// Least-squares line through `(xs, ys)`: the slope and the coefficient of
/// determination. Shared with the GPU experiment.
pub fn linfit(xs: &[f32], ys: &[f32]) -> (f32, f32) {
    let n = xs.len() as f32;
    if n < 2.0 {
        return (f32::NAN, f32::NAN);
    }
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
    }
    if sxx < 1e-9 {
        return (f32::NAN, f32::NAN);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        let pred = intercept + slope * x;
        ss_res += (y - pred) * (y - pred);
        ss_tot += (y - my) * (y - my);
    }
    (
        slope,
        if ss_tot > 1e-9 {
            1.0 - ss_res / ss_tot
        } else {
            f32::NAN
        },
    )
}

fn empty_wave() -> AgitationWave {
    AgitationWave {
        speed: f32::NAN,
        r2: f32::NAN,
        samples: 0,
        peak_alarmed: 0.0,
        peak_expansion: 0.0,
        points: Vec::new(),
    }
}
