//! The simulation: pure, headless and deterministic for a given seed.
//!
//! [`World`] owns the flock as a struct of arrays and advances it in fixed
//! integration steps. It knows nothing about the terminal or the GPU. Mirrors
//! `murmuration::sim::World`; the decisions are applied in index order here
//! instead of with `rayon`, which is bit-identical because each bird reads a
//! snapshot and its own random stream.

use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::arc::Arc;
use crate::escape::EscapeState;
use crate::falcon::Falcon;
use crate::flight::{self, Turning};
use crate::neighbours::Grid;
use crate::params::*;
use crate::social;
use crate::state::{Bird, Snapshot};

/// The bird count the model's initial cloud is sized for: the default of the
/// sibling terminal version's `--analyze`. See [`World::new_density_scaled`].
pub const CLOUD_REFERENCE_BIRDS: f32 = 2000.0;

/// The radius of the model's initial cloud at [`CLOUD_REFERENCE_BIRDS`] \[m\].
/// [`World::new`] uses it directly; `--analyze`'s large-N runs scale it with
/// `n^(1/3)` to hold the cloud's density (see `docs/model.md`).
pub const CLOUD_RADIUS: f32 = 18.0;

/// The flock and its environment.
pub struct World {
    /// Simulated time \[s\].
    pub time: f32,
    /// Preferred flight height, the roost "zero level" \[m\].
    pub z0: f32,
    /// Bird positions \[m\].
    pub position: Vec<Vec3>,
    /// Bird velocities [m/s].
    pub velocity: Vec<Vec3>,
    /// Bank angle per bird \[rad\].
    pub roll: Vec<f32>,
    /// Bank rate per bird [rad/s] (used by the inertial turning mode).
    pub spin: Vec<f32>,
    /// Wingbeat phase per bird \[rad\].
    pub flap_phase: Vec<f32>,
    /// Wingbeat amplitude multiplier per bird (higher while accelerating).
    pub flap_amp: Vec<f32>,
    /// Adaptive interaction radius per bird \[m\].
    pub radii: Vec<f32>,
    /// Steering force held since each bird's last decision \[N\].
    pub steering: Vec<Vec3>,
    /// Simulated time of each bird's next decision \[s\].
    pub next_decision: Vec<f32>,
    /// Neighbour count `|N_i|` from each bird's last decision.
    pub neighbour_count: Vec<u32>,
    /// Centrality `C_i` from each bird's last decision.
    pub centrality: Vec<f32>,
    /// Birds committed to an inward turn after leaving the display area.
    pub returning: Vec<bool>,
    /// Per-bird escape state (alarm, zigzag, flash expansion).
    pub escape: Vec<EscapeState>,
    /// The falcon, if one is hunting.
    pub falcon: Option<Falcon>,
    /// Number of neighbours whose zigzag starts a copy (1–7).
    pub zigzag_k: u32,
    /// How the bank responds to steering (base or inertial).
    pub turning: Turning,
    /// The evening arc, if one is running.
    pub arc: Option<Arc>,
    /// Target flock size the arc aims for during arrival.
    pub target_birds: usize,
    /// Weight of the random force (Eq. 13). Defaults to the model's `W_XI`; it
    /// is a parameter only so the CPU/GPU single-step parity check can switch
    /// the noise off on both sides at once (see `docs/model.md`).
    pub w_xi: f32,
    /// Per-bird random streams, so decisions are order-independent.
    rngs: Vec<ChaCha8Rng>,
    grid: Grid,
    /// The seed this world was created with.
    pub seed: u64,
}

impl World {
    /// Creates a compact, slightly flattened flock near the roost edge, nearly
    /// aligned at cruise speed. The social and return rules develop its display.
    pub fn new(n: usize, seed: u64) -> Self {
        Self::with_cloud_radius(n, seed, CLOUD_RADIUS)
    }

    /// Like [`World::new`] but with the initial cloud's radius chosen by the
    /// caller. The random draw order is unchanged, so [`World::new`] stays
    /// byte-for-byte the model's initial condition; this exists for
    /// [`World::new_density_scaled`].
    pub fn with_cloud_radius(n: usize, seed: u64, cloud_radius: f32) -> Self {
        let z0 = Z0;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut position = Vec::with_capacity(n);
        let mut velocity = Vec::with_capacity(n);
        let mut roll = Vec::with_capacity(n);
        let mut spin = Vec::with_capacity(n);
        let mut flap_phase = Vec::with_capacity(n);
        let mut radii = Vec::with_capacity(n);
        let mut next_decision = Vec::with_capacity(n);
        let mut rngs = Vec::with_capacity(n);
        let bearing = rng.random_range(0.0..std::f32::consts::TAU);
        let outward = Vec3::new(bearing.cos(), bearing.sin(), 0.0);
        let centre = Vec3::new(outward.x * 140.0, outward.y * 140.0, z0);
        for i in 0..n {
            let dir = social::random_unit(&mut rng);
            let rad = cloud_radius * rng.random::<f32>().cbrt();
            position.push(centre + Vec3::new(dir.x * rad, dir.y * rad, dir.z * rad * 0.4));
            // A compact, uniform-velocity flock already near the roost edge, so
            // it does not beeline across the whole display area first.
            velocity.push((Vec3::X + social::random_unit(&mut rng) * 0.05).normalize() * V0);
            roll.push(0.0);
            spin.push(0.0);
            flap_phase.push(rng.random_range(0.0..std::f32::consts::TAU));
            radii.push(5.0);
            next_decision.push(rng.random_range(0.0..DU));
            rngs.push(ChaCha8Rng::seed_from_u64(mix(seed, i as u64)));
        }
        let grid = Grid::new(GRID_CELL);
        Self {
            time: 0.0,
            z0,
            position,
            velocity,
            roll,
            spin,
            flap_phase,
            flap_amp: vec![1.0; n],
            radii,
            steering: vec![Vec3::ZERO; n],
            next_decision,
            neighbour_count: vec![0; n],
            centrality: vec![0.0; n],
            returning: vec![false; n],
            escape: vec![EscapeState::default(); n],
            falcon: None,
            zigzag_k: ZIGZAG_K,
            turning: Turning::Base,
            arc: None,
            target_birds: n,
            w_xi: W_XI,
            rngs,
            grid,
            seed,
        }
    }

    /// A world whose initial cloud has the same *density* as the model's at its
    /// design size ([`CLOUD_REFERENCE_BIRDS`]), by growing the cloud radius
    /// with `n^(1/3)`. Used by the headless `--analyze` run for large flocks.
    ///
    /// The model's initial condition is a fixed 18 m cloud (see [`CLOUD_RADIUS`]),
    /// which is right for the few thousand birds the published version runs.
    /// At 100,000 birds it is ~50× denser than at its design size: the first
    /// `decide` then scans tens of thousands of candidates per bird and the
    /// driver loses the device. Scaling the radius keeps the cloud's shape and
    /// density; the model's own dynamics then evolve it. See `docs/model.md`.
    pub fn new_density_scaled(n: usize, seed: u64) -> Self {
        let scale = (n as f32 / CLOUD_REFERENCE_BIRDS).max(1.0).cbrt();
        Self::with_cloud_radius(n, seed, CLOUD_RADIUS * scale)
    }

    /// Starts a new evening of `minutes` minutes, from an empty sky.
    pub fn enable_arc(&mut self, minutes: f32) {
        self.clear();
        self.arc = Some(Arc::new(minutes, self.seed));
    }

    /// Adds one bird with the given position and velocity.
    pub fn push_bird(&mut self, pos: Vec3, vel: Vec3, rng: &mut ChaCha8Rng) {
        let i = self.len();
        self.position.push(pos);
        self.velocity.push(vel);
        self.roll.push(0.0);
        self.spin.push(0.0);
        self.flap_phase
            .push(rng.random_range(0.0..std::f32::consts::TAU));
        self.flap_amp.push(1.0);
        self.radii.push(5.0);
        self.steering.push(Vec3::ZERO);
        self.next_decision
            .push(self.time + rng.random_range(0.0..DU));
        self.neighbour_count.push(0);
        self.centrality.push(0.0);
        self.returning.push(false);
        self.escape.push(EscapeState::default());
        self.rngs
            .push(ChaCha8Rng::seed_from_u64(mix(self.seed, i as u64)));
    }

    /// Removes the bird at `i`, swapping the last bird into its place.
    fn remove_bird(&mut self, i: usize) {
        self.position.swap_remove(i);
        self.velocity.swap_remove(i);
        self.roll.swap_remove(i);
        self.spin.swap_remove(i);
        self.flap_phase.swap_remove(i);
        self.flap_amp.swap_remove(i);
        self.radii.swap_remove(i);
        self.steering.swap_remove(i);
        self.next_decision.swap_remove(i);
        self.neighbour_count.swap_remove(i);
        self.centrality.swap_remove(i);
        self.returning.swap_remove(i);
        self.escape.swap_remove(i);
        self.rngs.swap_remove(i);
    }

    /// Removes birds that have landed below the reeds near the roost.
    pub fn remove_landed(&mut self) {
        let mut i = 0;
        while i < self.len() {
            let p = self.position[i];
            let horizontal = (p.x * p.x + p.y * p.y).sqrt();
            if p.z < REED_TOP && horizontal < R_ROOST {
                self.remove_bird(i);
            } else {
                i += 1;
            }
        }
    }

    /// Removes every bird.
    pub fn clear(&mut self) {
        self.position.clear();
        self.velocity.clear();
        self.roll.clear();
        self.spin.clear();
        self.flap_phase.clear();
        self.flap_amp.clear();
        self.radii.clear();
        self.steering.clear();
        self.next_decision.clear();
        self.neighbour_count.clear();
        self.centrality.clear();
        self.returning.clear();
        self.escape.clear();
        self.rngs.clear();
    }

    /// Adds `count` birds near the flock centroid.
    pub fn add_birds(&mut self, count: usize, rng: &mut ChaCha8Rng) {
        let centre = if self.is_empty() {
            Vec3::new(0.0, 0.0, self.z0)
        } else {
            self.position.iter().copied().sum::<Vec3>() / self.len() as f32
        };
        // New birds join the flock's heading with a little noise, instead of
        // flying off in a random direction (including straight up or down).
        let mean_dir = if self.is_empty() {
            Vec3::X
        } else {
            flight::forward(self.velocity.iter().copied().sum::<Vec3>())
        };
        for _ in 0..count {
            let pos = centre + social::random_unit(rng) * rng.random_range(0.0..30.0);
            let dir = (mean_dir + social::random_unit(rng) * 0.1).normalize_or_zero();
            let dir = if dir.length_squared() < 0.5 {
                mean_dir
            } else {
                dir
            };
            self.push_bird(pos, dir * V0, rng);
        }
    }

    /// Removes up to `count` birds.
    pub fn remove_birds(&mut self, count: usize) {
        for _ in 0..count {
            if self.is_empty() {
                break;
            }
            let last = self.len() - 1;
            self.remove_bird(last);
        }
    }

    /// Evening brightness: 0 bright gold, 1 night. Display dusk without an arc.
    pub fn evening(&self) -> f32 {
        match &self.arc {
            Some(arc) => arc.evening(),
            None => 0.5,
        }
    }

    /// Adds an autonomous falcon to the world.
    pub fn spawn_falcon(&mut self) {
        if self.falcon.is_none() {
            self.falcon = Some(Falcon::new(self.seed, self.z0));
        }
    }

    /// Number of birds.
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// Whether the flock is empty.
    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    /// Advances the world by one integration step.
    pub fn step(&mut self, dt: f32) {
        self.time += dt;
        if let Some(mut arc) = self.arc.take() {
            arc.update(dt, self);
            self.arc = Some(arc);
        }
        let time = self.time;
        let z0 = self.z0;

        // The falcon hunts.
        if !self.is_empty() {
            let n = self.len() as f32;
            let centroid = self.position.iter().copied().sum::<Vec3>() / n;
            let extent = self
                .position
                .iter()
                .map(|p| (*p - centroid).length())
                .fold(1.0f32, f32::max);
            if let Some(falcon) = &mut self.falcon {
                falcon.update(dt, centroid, extent, time);
            }
        }

        // Alarm, the alarm-triggered manoeuvre, and flash expansion.
        match &self.falcon {
            Some(falcon) => {
                let fpos = falcon.position;
                let fspeed = falcon.speed();
                for i in 0..self.len() {
                    let dist = (self.position[i] - fpos).length();
                    if dist < R_THREAT {
                        // Roll away from the falcon: bank so the bird's up
                        // vector tilts away from it.
                        let side = flight::side_of(flight::forward(self.velocity[i]));
                        let to_falcon = (fpos - self.position[i]).dot(side);
                        let e = &mut self.escape[i];
                        if !e.zigzag_active && time >= e.cooldown_until {
                            e.zigzag_start = time;
                            e.zigzag_active = true;
                            e.zigzag_sign = if to_falcon > 0.0 { -1.0 } else { 1.0 };
                        }
                        e.alarmed = true;
                        if dist < R_FLASH && fspeed > V_FLASH {
                            e.flash_until = time + FLASH_DURATION;
                        }
                    } else {
                        self.escape[i].alarmed = false;
                    }
                }
            }
            None => {
                for e in &mut self.escape {
                    e.alarmed = false;
                }
            }
        }

        self.grid.rebuild(&self.position);
        let k = self.zigzag_k;

        // Decisions for every bird whose reaction is due. They read a snapshot
        // of the flock, so sequential and parallel execution agree exactly.
        {
            let Self {
                position,
                velocity,
                radii,
                steering,
                next_decision,
                neighbour_count,
                centrality,
                returning,
                rngs,
                grid,
                escape,
                falcon,
                w_xi,
                ..
            } = self;
            let mut decisions: Vec<Option<social::Decision>> =
                (0..position.len()).map(|_| None).collect();
            for (i, (slot, rng)) in decisions.iter_mut().zip(rngs.iter_mut()).enumerate() {
                if time >= next_decision[i] {
                    *slot = Some(social::decide(
                        i,
                        position,
                        velocity,
                        radii,
                        grid,
                        rng,
                        z0,
                        falcon.as_ref(),
                        escape,
                        time,
                        k,
                        returning,
                        *w_xi,
                    ));
                }
            }
            for (i, decision) in decisions.iter_mut().enumerate() {
                let Some(d) = decision.take() else {
                    continue;
                };
                steering[i] = d.force;
                radii[i] = d.radius;
                neighbour_count[i] = d.neighbours;
                centrality[i] = d.centrality;
                returning[i] = d.returning;
                if let Some(sign) = d.start_zigzag {
                    escape[i].zigzag_start = time;
                    escape[i].zigzag_active = true;
                    escape[i].zigzag_sign = sign;
                }
                next_decision[i] += DU;
            }
        }

        // End finished zigzags and start their cooldown.
        for e in &mut self.escape {
            if e.zigzag_active && time - e.zigzag_start >= ZIGZAG_DURATION {
                e.zigzag_active = false;
                e.cooldown_until = time + ZIGZAG_COOLDOWN;
            }
        }

        // Flight integration for every bird.
        let turning = self.turning;
        {
            let Self {
                position,
                velocity,
                roll,
                spin,
                flap_phase,
                flap_amp,
                steering,
                escape,
                ..
            } = self;
            for i in 0..position.len() {
                let before = velocity[i];
                let forced = crate::escape::zigzag_roll(&escape[i], time);
                flight::step_bird(
                    &mut position[i],
                    &mut velocity[i],
                    &mut roll[i],
                    &mut spin[i],
                    steering[i],
                    forced,
                    dt,
                    turning,
                );
                // Advance the wingbeat and let it strengthen with acceleration.
                let accel = (velocity[i] - before).length() / dt.max(1e-6);
                flap_phase[i] =
                    (flap_phase[i] + std::f32::consts::TAU * FLAP_HZ * dt) % std::f32::consts::TAU;
                flap_amp[i] = 1.0 + (accel / G).clamp(0.0, 1.0);
            }
        }
    }

    /// Captures the full state so another engine (or a restarted run) can
    /// resume exactly. See [`crate::state`].
    pub fn snapshot(&self) -> Snapshot {
        let birds = (0..self.len())
            .map(|i| Bird {
                pos: self.position[i],
                vel: self.velocity[i],
                roll: self.roll[i],
                spin: self.spin[i],
                flap_phase: self.flap_phase[i],
                flap_amp: self.flap_amp[i],
                radius: self.radii[i],
                steering: self.steering[i],
                next_decision: self.next_decision[i],
                neighbour_count: self.neighbour_count[i],
                centrality: self.centrality[i],
                returning: self.returning[i],
                escape: self.escape[i],
                rng: self.rngs[i].clone(),
            })
            .collect();
        Snapshot {
            time: self.time,
            z0: self.z0,
            seed: self.seed,
            target_birds: self.target_birds,
            zigzag_k: self.zigzag_k,
            turning: self.turning,
            w_xi: self.w_xi,
            birds,
        }
    }

    /// Rebuilds a world from a snapshot produced by [`World::snapshot`].
    pub fn restore(snapshot: &Snapshot) -> Self {
        let mut world = World::new(0, snapshot.seed);
        for b in &snapshot.birds {
            world.position.push(b.pos);
            world.velocity.push(b.vel);
            world.roll.push(b.roll);
            world.spin.push(b.spin);
            world.flap_phase.push(b.flap_phase);
            world.flap_amp.push(b.flap_amp);
            world.radii.push(b.radius);
            world.steering.push(b.steering);
            world.next_decision.push(b.next_decision);
            world.neighbour_count.push(b.neighbour_count);
            world.centrality.push(b.centrality);
            world.returning.push(b.returning);
            world.escape.push(b.escape);
            world.rngs.push(b.rng.clone());
        }
        world.time = snapshot.time;
        world.z0 = snapshot.z0;
        world.target_birds = snapshot.target_birds;
        world.zigzag_k = snapshot.zigzag_k;
        world.turning = snapshot.turning;
        world.w_xi = snapshot.w_xi;
        world
    }
}

/// SplitMix64, for deriving independent per-bird seeds from one master seed.
fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_for_a_seed() {
        let mut a = World::new(200, 7);
        let mut b = World::new(200, 7);
        for _ in 0..100 {
            a.step(DT);
            b.step(DT);
        }
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = World::new(200, 1);
        let mut b = World::new(200, 2);
        for _ in 0..100 {
            a.step(DT);
            b.step(DT);
        }
        assert_ne!(a.position, b.position);
    }

    #[test]
    fn interaction_radius_relaxes() {
        let mut w = World::new(1000, 1);
        for _ in 0..2000 {
            w.step(DT);
        }
        let mean_r = w.radii.iter().sum::<f32>() / w.len() as f32;
        let mean_n = w.neighbour_count.iter().map(|&c| c as f32).sum::<f32>() / w.len() as f32;
        assert!(mean_r < 20.0, "mean R blew up: {mean_r}");
        assert!(mean_n < 40.0, "mean |N| blew up: {mean_n}");
    }

    #[test]
    fn added_birds_follow_the_flock() {
        let mut w = World::new(200, 7);
        for _ in 0..200 {
            w.step(DT);
        }
        let before = w.len();
        let mean_dir = flight::forward(w.velocity.iter().copied().sum::<Vec3>());
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        w.add_birds(50, &mut rng);
        for i in before..w.len() {
            let dir = flight::forward(w.velocity[i]);
            assert!(
                dir.dot(mean_dir) > 0.9,
                "added bird {i} heads {dir} while the flock heads {mean_dir}"
            );
        }
    }

    #[test]
    fn snapshot_restore_continues_exactly() {
        let mut a = World::new(300, 5);
        for _ in 0..500 {
            a.step(DT);
        }
        let snapshot = a.snapshot();
        let mut b = World::restore(&snapshot);
        assert_eq!(a.time, b.time);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.position, b.position);
        for _ in 0..500 {
            a.step(DT);
            b.step(DT);
        }
        assert_eq!(a.position, b.position, "restored world diverged");
        assert_eq!(a.velocity, b.velocity);
    }

    #[test]
    fn wingbeat_phase_advances_and_stays_bounded() {
        let mut w = World::new(100, 1);
        let tau = std::f32::consts::TAU;
        for _ in 0..100 {
            w.step(DT);
        }
        for (&phase, &amp) in w.flap_phase.iter().zip(&w.flap_amp) {
            assert!((0.0..tau).contains(&phase), "phase {phase} out of range");
            assert!((1.0..=2.0).contains(&amp), "amplitude {amp} out of range");
        }
    }
}
