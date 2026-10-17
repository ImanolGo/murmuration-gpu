//! The agitation-wave experiment on the GPU (PLAN.md phase 7).
//!
//! Mirrors `murmur_ref::escape::measure_agitation` step for step, but the flock
//! runs on the GPU: settle without a falcon, seed a compact patch of zigzags on
//! one side, then step and watch which bird rolls first. The seeding and the fit
//! are the oracle's own code path; only the integration differs, so the GPU's
//! speed can be compared with the reference's range for the same bird count.
//!
//! The per-step readback is deliberate: the experiment is a measurement, not a
//! render, and it needs each bird's *first* zigzag start, which the shader does
//! not keep. It costs one round trip per 5 ms step; the settle, which does not
//! need it, is batched.

use glam::Vec3;
use murmur_ref::escape::{AgitationWave, linfit, spread};
use murmur_ref::params::DT;

use crate::device::{Gpu, Stalled};
use crate::sim::Sim;

/// Runs an agitation wave on the GPU and measures how the zigzag copies through
/// the flock, exactly as [`murmur_ref::escape::measure_agitation`] does on the
/// CPU.
pub fn measure(gpu: &Gpu, sim: &mut Sim, seconds: f32) -> Result<AgitationWave, Stalled> {
    let n = sim.len();
    if n == 0 {
        return Ok(AgitationWave {
            speed: f32::NAN,
            r2: f32::NAN,
            samples: 0,
            peak_alarmed: 0.0,
            peak_expansion: 0.0,
            points: Vec::new(),
        });
    }

    // Settle without a falcon.
    sim.advance_seconds(gpu, 20.0)?;

    // Seed a compact patch on one side, from the settled positions.
    let settled = sim.read_birds(gpu)?;
    let initial_position: Vec<Vec3> = settled.iter().map(|b| Vec3::from_array(b.pos)).collect();
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

    let start_time = sim.time();
    let mut seeded = settled;
    for &i in &seed_birds {
        seeded[i].zigzag_start = start_time;
        seeded[i].zigzag_active = 1;
        seeded[i].zigzag_sign = 1.0;
    }
    sim.write_birds(gpu, &seeded);

    let steps = (seconds / DT).max(1.0) as u32;
    let mut arrival = vec![f32::INFINITY; n];
    let initial_spread = spread(&initial_position);
    let mut peak_spread = initial_spread;

    for _ in 0..steps {
        sim.advance(gpu, 1);
        let birds = sim.read_birds(gpu)?;
        for (i, a) in arrival.iter_mut().enumerate() {
            let started = birds[i].zigzag_start;
            if a.is_infinite() && started >= start_time && started.is_finite() {
                *a = started;
            }
        }
        let positions: Vec<Vec3> = birds.iter().map(|b| Vec3::from_array(b.pos)).collect();
        peak_spread = peak_spread.max(spread(&positions));
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
    Ok(AgitationWave {
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
    })
}
