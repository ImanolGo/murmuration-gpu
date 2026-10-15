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
    spawn_timer: f32,
    arrival_bearing: f32,
    arrival_radius: f32,
}

impl Arc {
    /// A new evening of `minutes` minutes.
    pub fn new(minutes: f32, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x00A2_C0FE);
        let arrival_bearing = rng.random_range(0.0..std::f32::consts::TAU);
        let arrival_radius = rng.random_range(280.0..320.0);
        Self {
            duration: (minutes * 60.0).max(1.0),
            time: 0.0,
            base_z0: Z0,
            rng,
            spawn_timer: 0.0,
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
            self.arrival_radius = self.rng.random_range(280.0..320.0);
        }
        match self.phase() {
            Phase::Arrival => {
                world.z0 = self.base_z0;
                self.spawn_timer -= dt;
                while self.spawn_timer <= 0.0 && world.len() < world.target_birds {
                    let remaining = world.target_birds - world.len();
                    let group = self.rng.random_range(50..=300).min(remaining.max(1));
                    self.spawn_group(world, group);
                    self.spawn_timer += 0.5;
                }
                if self.spawn_timer < 0.0 {
                    self.spawn_timer = 0.0;
                }
            }
            Phase::Display => {}
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
        // A shared approach corridor lets arriving groups meet with compatible
        // headings. Random antipodal approaches to unrelated targets created
        // opposing streams that passed through one another instead of gathering.
        let bearing = self.arrival_bearing + self.rng.random_range(-0.045..0.045);
        let radius = self.arrival_radius + self.rng.random_range(-8.0..8.0);
        let height = self.base_z0 + self.rng.random_range(-10.0..10.0);
        let base = Vec3::new(radius * bearing.cos(), radius * bearing.sin(), height);
        let target = Vec3::new(
            self.rng.random_range(-20.0..20.0),
            self.rng.random_range(-20.0..20.0),
            self.base_z0,
        );
        let radius = (count as f32).cbrt() * 1.9;
        for _ in 0..count {
            let offset = crate::social::random_unit(&mut self.rng)
                * radius
                * self.rng.random::<f32>().cbrt();
            let pos = base + Vec3::new(offset.x, offset.y, offset.z * 0.6);
            let toward = target - pos;
            let dir = (toward.normalize_or_zero()
                + Vec3::new(
                    self.rng.random_range(-0.05..0.05),
                    self.rng.random_range(-0.05..0.05),
                    self.rng.random_range(-0.025..0.025),
                ))
            .normalize_or_zero();
            world.push_bird(pos, dir * V0, &mut self.rng);
        }
    }
}
