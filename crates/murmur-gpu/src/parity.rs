//! CPU/GPU parity: run the same model on both engines and compare.
//!
//! Flocking is chaotic: two runs that differ only in the random force follow the
//! same trajectory at first and then diverge exponentially, so the comparison
//! has to say *when* and *what* it compares (PLAN.md phase 2):
//!
//! - [`single_step`]: one step from the same state, noise off on both engines,
//!   matches to about 1e-4 relative error per bird. This is the strongest test
//!   that the shader mirrors the reference function for function.
//! - [`lock_in`]: both engines stepped together for a few seconds, before chaos
//!   dominates, must agree on *every* reference metric. At 2,000 birds and the
//!   seeds tried, every metric stays within 10 % for at least 8 s.
//! - [`run_seed`]: 60 simulated seconds. The self-averaging flock statistics
//!   (neighbour count, nearest-neighbour distance, density, speed, height,
//!   extent) agree within 10 %; the global *shape* metrics (`I2/I1`, `I3/I1`,
//!   `flat`) do not self-average over a single 2,000-bird flock's minute — the
//!   CPU's own value swings by an order of magnitude from seed to seed (3.2 to
//!   30.4 for `I2/I1` here) — so they are compared with a documented 50 %
//!   tolerance and left to tighten on the larger runs of phase 4.

use glam::Vec3;
use murmur_ref::World;
use murmur_ref::analysis::{self, Stats};
use murmur_ref::params::DT;

use crate::device::Gpu;
use crate::sim::Sim;

/// How to read a metric from [`Stats`].
type Accessor = fn(&Stats) -> f32;

/// One compared metric: name, reader, relative tolerance and absolute floor
/// (so a near-zero metric is not compared as a pure ratio).
struct Spec {
    name: &'static str,
    accessor: Accessor,
    rtol: f32,
    atol: f32,
}

/// The reference metrics compared, with their tolerances.
const METRICS: &[Spec] = &[
    spec("|N|", |s| s.mean_neighbours, 0.10, 0.5),
    spec("NND", |s| s.mean_nnd, 0.10, 0.05),
    spec("cluster", |s| s.largest_cluster_fraction, 0.10, 0.005),
    spec("nn_ab", |s| s.nn_aligned_fraction, 0.10, 0.02),
    spec("R", |s| s.mean_radius, 0.10, 0.5),
    spec("spread", |s| s.spread, 0.10, 1.0),
    spec("Rh", |s| s.max_radius_h, 0.10, 3.0),
    spec("dz", |s| s.mean_dz, 0.10, 0.5),
    spec("v", |s| s.mean_speed, 0.10, 0.2),
    spec("roll", |s| s.mean_roll, 0.10, 0.02),
    spec("clamp", |s| s.clamp_fraction, 0.10, 0.005),
    spec("hstd", |s| s.height_std, 0.10, 0.3),
    spec("rc", |s| s.centroid_radius, 0.10, 3.0),
    spec("tang", |s| s.tangential, 0.10, 0.5),
    // Chaotic, non-self-averaging global shape (see the module docs).
    spec("I2/I1", |s| s.i2_i1, 0.50, 1.0),
    spec("I3/I1", |s| s.i3_i1, 0.50, 3.0),
    spec("flat", |s| s.flatness, 0.50, 0.05),
];

const fn spec(name: &'static str, accessor: Accessor, rtol: f32, atol: f32) -> Spec {
    Spec {
        name,
        accessor,
        rtol,
        atol,
    }
}

/// One metric's compared values and deviation.
#[derive(Debug, Clone, Copy)]
pub struct Metric {
    /// Metric name.
    pub name: &'static str,
    /// Relative tolerance applied.
    pub rtol: f32,
    /// CPU value.
    pub cpu: f32,
    /// GPU value.
    pub gpu: f32,
    /// `|cpu - gpu|`.
    pub abs: f32,
    /// The bound it must stay under: `rtol·|cpu| + atol`.
    pub bound: f32,
    /// Whether it passed its tolerance.
    pub ok: bool,
}

/// The result of one seed's parity run.
pub struct SeedReport {
    /// The seed used.
    pub seed: u64,
    /// `(time, cpu stats, gpu stats)` once per simulated second.
    pub per_second: Vec<(f32, Stats, Stats)>,
    /// The settled-mean comparison per metric (last half of the run).
    pub settled: Vec<Metric>,
}

impl SeedReport {
    /// Whether every settled metric passed.
    pub fn passed(&self) -> bool {
        self.settled.iter().all(|m| m.ok)
    }
}

/// The error of `a` relative to `b`, with an absolute floor.
fn rel(a: f32, b: f32, atol: f32) -> f32 {
    (a - b).abs() / b.abs().max(atol)
}

fn rel_vec(diff: Vec3, value: Vec3) -> f32 {
    diff.length() / value.length().max(1e-3)
}

/// One step from the same state on both engines, random force off on both.
/// Returns the largest relative error over every bird's position and velocity.
pub fn single_step_max_rel_error(gpu: &Gpu, world: &World) -> f32 {
    let snapshot = world.snapshot();

    let mut sim = Sim::from_snapshot(gpu, &snapshot);
    sim.set_w_xi(gpu, 0.0);
    sim.advance(gpu, 1);
    let gpu_world = World::restore(&sim.snapshot(gpu));

    let mut cpu = World::restore(&snapshot);
    cpu.w_xi = 0.0;
    cpu.step(DT);

    let mut max = 0.0f32;
    for i in 0..cpu.len() {
        max = max.max(rel_vec(
            cpu.position[i] - gpu_world.position[i],
            cpu.position[i],
        ));
        max = max.max(rel_vec(
            cpu.velocity[i] - gpu_world.velocity[i],
            cpu.velocity[i],
        ));
        max = max.max(rel(cpu.radii[i], gpu_world.radii[i], 0.01));
    }
    max
}

/// The same one-step check with the noise on, so the residual is the
/// unmatched-noise floor the statistical parity has to live with.
pub fn single_step_noise_floor(gpu: &Gpu, world: &World) -> f32 {
    let snapshot = world.snapshot();

    let mut sim = Sim::from_snapshot(gpu, &snapshot);
    sim.advance(gpu, 1);
    let gpu_world = World::restore(&sim.snapshot(gpu));

    let mut cpu = World::restore(&snapshot);
    cpu.step(DT);

    let mut max = 0.0f32;
    for i in 0..cpu.len() {
        max = max.max(rel_vec(
            cpu.position[i] - gpu_world.position[i],
            cpu.position[i],
        ));
    }
    max
}

/// Steps both engines together for `seconds`, comparing every metric each
/// second while the trajectories are still locked. Returns the worst deviation
/// seen per metric. Every metric is compared at a tight 10 % here, including
/// the global-shape ones that the 60 s run has to relax: within the lock-in
/// window they agree closely too.
pub fn lock_in(gpu: &Gpu, seed: u64, birds: usize, seconds: u32) -> Vec<Metric> {
    const RTOL: f32 = 0.10;
    let samples = step_together(gpu, seed, birds, seconds);
    METRICS
        .iter()
        .map(|spec| {
            let mut worst = worst_metric(spec, &samples);
            worst.rtol = RTOL;
            worst.bound = RTOL * worst.cpu.abs() + spec.atol;
            worst.ok = worst.abs <= worst.bound;
            worst
        })
        .collect()
}

/// Runs `seconds` simulated seconds on both engines from the same state and
/// compares the settled statistics (last half of the run).
pub fn run_seed(gpu: &Gpu, seed: u64, birds: usize, seconds: u32) -> SeedReport {
    let samples = step_together(gpu, seed, birds, seconds);

    let start = samples.len() / 2;
    let settled: Vec<Metric> = METRICS
        .iter()
        .map(|spec| {
            let cpu = mean(samples[start..].iter().map(|s| (spec.accessor)(&s.1)));
            let on_gpu = mean(samples[start..].iter().map(|s| (spec.accessor)(&s.2)));
            let abs = (cpu - on_gpu).abs();
            let bound = spec.rtol * cpu.abs() + spec.atol;
            Metric {
                name: spec.name,
                rtol: spec.rtol,
                cpu,
                gpu: on_gpu,
                abs,
                bound,
                ok: abs <= bound,
            }
        })
        .collect();

    SeedReport {
        seed,
        per_second: samples,
        settled,
    }
}

/// The sample with the largest relative deviation for one metric.
fn worst_metric(spec: &Spec, samples: &[(f32, Stats, Stats)]) -> Metric {
    let mut worst = Metric {
        name: spec.name,
        rtol: spec.rtol,
        cpu: f32::NAN,
        gpu: f32::NAN,
        abs: 0.0,
        bound: 0.0,
        ok: true,
    };
    let mut worst_rel = -1.0f32;
    for (_, cpu, on_gpu) in samples {
        let c = (spec.accessor)(cpu);
        let g = (spec.accessor)(on_gpu);
        let abs = (c - g).abs();
        let r = rel(g, c, spec.atol);
        if r > worst_rel {
            worst_rel = r;
            worst = Metric {
                name: spec.name,
                rtol: spec.rtol,
                cpu: c,
                gpu: g,
                abs,
                bound: spec.rtol * c.abs() + spec.atol,
                ok: abs <= spec.rtol * c.abs() + spec.atol,
            };
        }
    }
    worst
}

/// Steps the CPU and the GPU one second at a time from the same snapshot.
fn step_together(gpu: &Gpu, seed: u64, birds: usize, seconds: u32) -> Vec<(f32, Stats, Stats)> {
    let mut cpu = World::new(birds, seed);
    let mut sim = Sim::from_snapshot(gpu, &cpu.snapshot());
    let per_second = (1.0 / DT).round() as u32;

    let mut samples = Vec::with_capacity(seconds as usize);
    for _ in 0..seconds {
        for _ in 0..per_second {
            cpu.step(DT);
        }
        sim.advance(gpu, per_second);
        let gpu_world = World::restore(&sim.snapshot(gpu));
        samples.push((
            cpu.time,
            analysis::compute(&cpu),
            analysis::compute(&gpu_world),
        ));
    }
    samples
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut sum = 0.0;
    let mut count = 0u32;
    for v in values {
        sum += v;
        count += 1;
    }
    if count > 0 {
        sum / count as f32
    } else {
        f32::NAN
    }
}

/// A one-line table of metrics.
pub fn table(title: &str, metrics: &[Metric]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "  {title:<22} {:>10} {:>10} {:>10} {:>10} {:>6}",
        "cpu", "gpu", "abs", "bound", "pass"
    );
    for m in metrics {
        let _ = writeln!(
            out,
            "  {:<22} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>6}",
            m.name,
            m.cpu,
            m.gpu,
            m.abs,
            m.bound,
            if m.ok { "ok" } else { "FAIL" },
        );
    }
    out
}
