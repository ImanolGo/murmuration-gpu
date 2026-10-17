//! The evening: arrival, display, roosting and night.
//!
//! One evening lasts `minutes` (the terminal default is 15). Groups of
//! starlings arrive, merge into a display flock, pour down into the reeds, and
//! night falls before the next evening starts. Mirrors `murmuration::sim::arc`;
//! used by PLAN.md phase 8 on the GPU side.

use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::params::*;
use crate::world::World;

/// Phase of the evening.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// First ~15 %: groups arrive from outside the view.
    Arrival,
    /// Up to ~70 %: the flock displays over the reeds.
    Display,
    /// Up to ~90 %: the flock pours down and lands.
    Roosting,
    /// The rest: night, then a new evening.
    Night,
}

/// The evening clock.
pub struct Arc {
    /// Length of one evening \[s\].
    pub duration: f32,
    /// Time within the current evening \[s\].
    pub time: f32,
    base_z0: f32,
    rng: ChaCha8Rng,
    arrived: usize,
    arrival_bearing: f32,
    arrival_radius: f32,
}

impl Arc {
    /// A new evening of `minutes` minutes.
    pub fn new(minutes: f32, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x00A2_C0FE);
        let arrival_bearing = rng.random_range(0.0..std::f32::consts::TAU);
        let arrival_radius = rng.random_range(180.0..200.0);
        Self {
            duration: (minutes * 60.0).max(1.0),
            time: 0.0,
            base_z0: Z0,
            rng,
            arrived: 0,
            arrival_bearing,
            arrival_radius,
        }
    }

    /// The current phase.
    pub fn phase(&self) -> Phase {
        let f = self.time / self.duration;
        if f < 0.15 {
            Phase::Arrival
        } else if f < 0.70 {
            Phase::Display
        } else if f < 0.90 {
            Phase::Roosting
        } else {
            Phase::Night
        }
    }

    /// How far into the evening we are, `0` (bright gold) to `1` (night).
    pub fn evening(&self) -> f32 {
        let f = (self.time / self.duration).clamp(0.0, 1.0);
        if f < 0.15 {
            f / 0.15 * 0.35
        } else if f < 0.70 {
            0.35 + (f - 0.15) / 0.55 * 0.25
        } else if f < 0.90 {
            0.60 + (f - 0.70) / 0.20 * 0.30
        } else {
            0.90 + (f - 0.90) / 0.10 * 0.10
        }
    }

    /// Advances the evening and its effect on the world.
    pub fn update(&mut self, dt: f32, world: &mut World) {
        self.time += dt;
        if self.time >= self.duration {
            self.time -= self.duration;
            self.arrival_bearing = self.rng.random_range(0.0..std::f32::consts::TAU);
            self.arrival_radius = self.rng.random_range(180.0..200.0);
            self.arrived = 0;
        }
        match self.phase() {
            Phase::Arrival => {
                world.z0 = self.base_z0;
                // Continuous flux with a soft start and finish, not discrete
                // half-second clouds. Short evenings still fill before display.
                let span = (self.duration * 0.15).min(6.0);
                let u = (self.time / span).clamp(0.0, 1.0);
                let wanted = (world.target_birds as f32 * u * u * (3.0 - 2.0 * u)).floor() as usize;
                let count = wanted.saturating_sub(self.arrived);
                self.spawn_group(world, count);
                self.arrived += count;
            }
            Phase::Display => {
                // Finish the last fractional bird at the phase boundary, even
                // when the clock's last arrival step rounded just below 1.
                let count = world.target_birds.saturating_sub(self.arrived);
                self.spawn_group(world, count);
                self.arrived += count;
            }
            Phase::Roosting => {
                let f = ((self.time / self.duration) - 0.70) / 0.20;
                world.z0 = self.base_z0 + (REED_TOP - self.base_z0) * f.clamp(0.0, 1.0);
                world.remove_landed();
            }
            Phase::Night => {
                world.z0 = REED_TOP;
                world.clear();
            }
        }
    }

    fn spawn_group(&mut self, world: &mut World, count: usize) {
        let outward = Vec3::new(self.arrival_bearing.cos(), self.arrival_bearing.sin(), 0.0);
        let tangent = Vec3::new(-outward.y, outward.x, 0.0);
        // A grazing approach joins the display's turn instead of shooting
        // through its centre. The cross-section depends on total flux, never
        // on how many birds happen to spawn in this integration step.
        let heading = -outward * 0.6 + tangent * 0.8;
        let side = heading.cross(Vec3::Z);
        let base = outward * self.arrival_radius + Vec3::Z * self.base_z0;
        let span = (self.duration * 0.15).min(6.0);
        let radius = (world.target_birds as f32 / (V0 * span * std::f32::consts::PI * 0.5))
            .sqrt()
            .max(3.0);
        // Start perception near the local flux density, rather than letting a
        // new bird scan hundreds of neighbours at the old fixed 5 m radius for
        // its first half-second. The adaptive law is unchanged after birth.
        let u = (self.time / span).clamp(0.0, 1.0);
        let flux = world.target_birds as f32 * 6.0 * u * (1.0 - u) / span;
        let density = flux / (V0 * std::f32::consts::PI * radius * radius * 0.6);
        let perception = (N_C / (4.0 / 3.0 * std::f32::consts::PI * density.max(0.05)))
            .cbrt()
            .clamp(R_H, 5.0);
        for _ in 0..count {
            let angle = self.rng.random_range(0.0..std::f32::consts::TAU);
            let r = radius * self.rng.random::<f32>().sqrt();
            let pos = base
                + side * (r * angle.cos())
                + Vec3::Z * (r * angle.sin() * 0.6)
                + heading * self.rng.random_range(-0.5..0.5);
            let dir = (heading
                + Vec3::new(
                    self.rng.random_range(-0.05..0.05),
                    self.rng.random_range(-0.05..0.05),
                    self.rng.random_range(-0.025..0.025),
                ))
            .normalize_or_zero();
            world.push_bird(pos, dir * V0, &mut self.rng);
            *world.radii.last_mut().expect("just spawned") = perception;
        }
    }
}
