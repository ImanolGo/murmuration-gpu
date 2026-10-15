//! The falcon: an autonomous hunter, or one driven by the mouse.
//!
//! Mirrors `murmuration::sim::falcon`. The GPU side mirrors it in PLAN.md
//! phase 7; the CPU version is the oracle.

use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::params::*;

/// Who is steering the falcon.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Hunts on its own.
    Autonomous,
    /// Chases the point the user points at.
    Mouse,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Cruise,
    Climb,
    Stoop,
    Retreat,
}

/// A bird of prey.
pub struct Falcon {
    /// Position \[m\].
    pub position: Vec3,
    /// Velocity [m/s].
    pub velocity: Vec3,
    /// Autonomous or mouse-driven.
    pub mode: Mode,
    /// Point being chased (mouse mode) or attacked (autonomous mode).
    pub target: Vec3,
    state: State,
    until: f32,
    dive: Vec3,
    rng: ChaCha8Rng,
}

impl Falcon {
    /// A falcon cruising near `z0`, attacking every 5–20 s.
    pub fn new(seed: u64, z0: f32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xF0_0D_FA_1C);
        let until = rng.random_range(FALCON_ATTACK_MIN..FALCON_ATTACK_MAX);
        Self {
            position: Vec3::new(0.0, -220.0, z0 + 20.0),
            velocity: Vec3::new(0.0, FALCON_CRUISE, 0.0),
            mode: Mode::Autonomous,
            target: Vec3::new(0.0, 0.0, z0),
            state: State::Cruise,
            until,
            dive: Vec3::Y,
            rng,
        }
    }

    /// Current speed [m/s].
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// Advances the falcon by `dt`.
    pub fn update(&mut self, dt: f32, centroid: Vec3, extent: f32, time: f32) {
        if self.mode == Mode::Mouse {
            let toward = self.target - self.position;
            let desired = if toward.length_squared() > 1e-6 {
                toward.normalize() * FALCON_STOOP
            } else {
                Vec3::ZERO
            };
            self.velocity += (desired - self.velocity) * (1.0 - (-dt / 0.3).exp());
            self.position += self.velocity * dt;
            return;
        }

        match self.state {
            State::Cruise => {
                let patrol = centroid + Vec3::new(extent + 80.0, 0.0, 20.0);
                self.steer_toward(patrol, FALCON_CRUISE, dt);
                if time >= self.until {
                    self.target = centroid;
                    self.state = State::Climb;
                }
            }
            State::Climb => {
                self.target = centroid;
                let above = self.target + Vec3::Z * FALCON_ATTACK_HEIGHT;
                self.steer_toward(above, FALCON_CLIMB, dt);
                if (self.position - above).length() < 8.0 {
                    let mut dive = (self.target - self.position).normalize_or_zero();
                    if dive.length_squared() < 0.5 {
                        dive = -Vec3::Z;
                    }
                    self.dive = dive;
                    self.state = State::Stoop;
                }
            }
            State::Stoop => {
                self.velocity = self.dive * FALCON_STOOP;
                self.position += self.velocity * dt;
                if (self.position - self.target).dot(self.dive) > 0.0 {
                    self.state = State::Retreat;
                }
            }
            State::Retreat => {
                let away = (self.position - self.target + Vec3::Z * FALCON_ATTACK_HEIGHT)
                    .normalize_or_zero();
                self.steer_toward(self.position + away * 100.0, FALCON_CLIMB, dt);
                if self.position.z > self.target.z + FALCON_ATTACK_HEIGHT + 20.0 {
                    self.state = State::Cruise;
                    self.until = time + self.rng.random_range(FALCON_ATTACK_MIN..FALCON_ATTACK_MAX);
                }
            }
        }
    }

    fn steer_toward(&mut self, point: Vec3, speed: f32, dt: f32) {
        let toward = point - self.position;
        let desired = if toward.length_squared() > 1e-6 {
            toward.normalize() * speed
        } else {
            Vec3::ZERO
        };
        self.velocity += (desired - self.velocity) * (1.0 - (-dt / 0.4).exp());
        self.position += self.velocity * dt;
    }
}
