//! The phase 4 coherence run and its verdict: the criteria a long GPU flock
//! must meet to count as a coherent display over the roost.
//!
//! These thresholds were **written down before the 100,000-bird, 10-minute run**
//! (see `docs/coherence-100k.md`) and are not tuned to its result. They are
//! grounded in the reference's settled values at 2,000 birds
//! (`docs/reference-analyze-2000-42.txt`) and in the model constants, and they
//! check the two failure modes that matter — collapse (spread and radius to
//! zero, `|N|` exploding) and dispersal/splitting (cluster fraction falling,
//! `Rh`/`rc` growing) — not the documented gaps. Elongation (`I2/I1`, `I3/I1`)
//! and milling (`tang`) are the reference's known weak points; they are
//! reported by `--analyze`, never used to fail the run.
//!
//! # Why this is a separate, lighter sampler
//!
//! [`crate::analyze`] prints the *full* reference metric set through
//! [`murmur_ref::analysis::compute`], whose 15 m nearest-neighbour scan is
//! almost all-pairs at a large dense flock: measured at 100,000 birds it costs
//! ~26 s per simulated second, so the 600 s run would take ~5 h and almost all
//! of it would be statistics the criteria do not read.
//!
//! The criteria need only the per-bird fields the simulation already keeps
//! (`|N|`, `R`, `v`, `roll`) plus a few O(N) reductions of the positions and one
//! 5 m cluster scan. [`sample`] computes exactly those, in the same order and
//! with the same formulas as `compute`, so the fields it shares are identical
//! (asserted by `tests/analyze.rs`). The cluster scan is the one field that is
//! not free; it is the same function `compute` calls
//! ([`murmur_ref::analysis::cluster_fraction`]) and still runs every sampled
//! second.

use glam::Vec3;
use murmur_ref::World;
use murmur_ref::analysis;
use murmur_ref::flight::Turning;
use murmur_ref::params::MAX_BANK;

use crate::analyze::{RunError, RunLog};
use crate::device::Gpu;

/// A bird's radius above which a neighbour query stops being cheap \[m\].
///
/// A query at `R` walks every cell covering a `2R` box, so the cost of one bird
/// grows with `R³`: at the grid's 1 m cell, `R = 10 m` is already ~8,000 cells
/// (~300× an ordinary bird) and the model's `R_MAX` is 100 m (~8.1e6 cells).
/// The threshold only splits the diagnostic below; no behaviour depends on it.
pub const WIDE_RADIUS: f32 = 10.0;

/// The subset of flock statistics the coherence criteria read, computed in the
/// same order and with the same formulas as [`murmur_ref::analysis::compute`].
///
/// The field names match [`murmur_ref::analysis::Stats`] so the two can be
/// compared field by field.
#[derive(Debug, Clone, Copy)]
pub struct CoherenceSample {
    /// Simulated time \[s\].
    pub time: f32,
    /// Number of birds.
    pub birds: usize,
    /// Mean topological neighbour count `|N_i|`.
    pub mean_neighbours: f32,
    /// Mean adaptive interaction radius \[m\].
    pub mean_radius: f32,
    /// Mean speed \[m/s\].
    pub mean_speed: f32,
    /// Mean distance of birds from the flock centroid \[m\].
    pub spread: f32,
    /// Mean absolute height above the roost level \[m\].
    pub mean_dz: f32,
    /// Standard deviation of height \[m\].
    pub height_std: f32,
    /// Fraction of birds pinned at the ±`MAX_BANK` clamp.
    pub clamp_fraction: f32,
    /// Horizontal distance of the flock centroid from the roost centre \[m\].
    pub centroid_radius: f32,
    /// Furthest bird from the roost centre in the horizontal plane \[m\].
    pub max_radius_h: f32,
    /// Fraction of birds in the largest single-link cluster at 5 m.
    pub largest_cluster_fraction: f32,
    /// Largest adaptive interaction radius `R` in the flock \[m\].
    ///
    /// This is the cost driver of the whole run, so it is sampled rather than
    /// assumed. A neighbour query walks the cells its radius covers, so the
    /// cost of one bird grows as `R³`; `params::R_MAX = 100 m` means a single
    /// straggler can cost as much as ~300,000 ordinary birds, which is what the
    /// acceptance run was hitting before the query was bounded
    /// (`gather` in `shaders/sim.wgsl`).
    pub max_radius: f32,
    /// Fraction of birds with `R > `[`WIDE_RADIUS`\] \[m\].
    pub wide_radius_fraction: f32,
}

impl CoherenceSample {
    /// A line in the reference `--analyze` format for the fields this sampler
    /// measures, in the reference's field order.
    pub fn line(&self) -> String {
        format!(
            "birds={:5} |N|={:4.2} R={:5.2}m cluster={:5.1}% spread={:5.1}m Rh={:5.1}m dz={:4.1}m clamp={:4.1}% v={:4.1}m/s hstd={:4.1}m rc={:5.1}m Rmax={:6.2}m wide={:4.1}%",
            self.birds,
            self.mean_neighbours,
            self.mean_radius,
            self.largest_cluster_fraction * 100.0,
            self.spread,
            self.max_radius_h,
            self.mean_dz,
            self.clamp_fraction * 100.0,
            self.mean_speed,
            self.height_std,
            self.centroid_radius,
            self.max_radius,
            self.wide_radius_fraction * 100.0,
        )
    }
}

/// The mean of an iterator of `f32`, or `NaN` when it is empty (matching
/// `analysis::mean`).
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

/// Measures the coherence fields of `world` at `time`.
///
/// Mirrors the matching block of [`murmur_ref::analysis::compute`]; see the
/// module docs for why it is a separate function.
pub fn sample(world: &World, time: f32) -> CoherenceSample {
    let pos = &world.position;
    let n = pos.len();
    if n == 0 {
        return CoherenceSample {
            time,
            birds: 0,
            mean_neighbours: f32::NAN,
            mean_radius: f32::NAN,
            mean_speed: f32::NAN,
            spread: f32::NAN,
            mean_dz: f32::NAN,
            height_std: f32::NAN,
            clamp_fraction: f32::NAN,
            centroid_radius: f32::NAN,
            max_radius_h: f32::NAN,
            largest_cluster_fraction: 0.0,
            max_radius: f32::NAN,
            wide_radius_fraction: f32::NAN,
        };
    }
    let nf = n as f32;
    let centroid = pos.iter().copied().sum::<Vec3>() / nf;
    let mean_height = mean(pos.iter().map(|p| p.z));
    CoherenceSample {
        time,
        birds: n,
        mean_neighbours: mean(world.neighbour_count.iter().map(|&c| c as f32)),
        mean_radius: mean(world.radii.iter().copied()),
        mean_speed: mean(world.velocity.iter().map(|v| v.length())),
        spread: mean(pos.iter().map(|p| (*p - centroid).length())),
        mean_dz: mean(pos.iter().map(|p| (p.z - world.z0).abs())),
        height_std: mean(pos.iter().map(|p| (p.z - mean_height).powi(2)))
            .max(0.0)
            .sqrt(),
        clamp_fraction: world
            .roll
            .iter()
            .filter(|r| r.abs() >= MAX_BANK - 1e-3)
            .count() as f32
            / nf,
        centroid_radius: (centroid.x * centroid.x + centroid.y * centroid.y).sqrt(),
        max_radius_h: pos
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .fold(0.0, f32::max),
        largest_cluster_fraction: analysis::cluster_fraction(pos, 5.0),
        max_radius: world.radii.iter().copied().fold(0.0, f32::max),
        wide_radius_fraction: world.radii.iter().filter(|r| **r > WIDE_RADIUS).count() as f32 / nf,
    }
}

/// A finished coherence run: one [`CoherenceSample`] per whole simulated second.
pub struct CoherenceReport {
    /// Birds simulated.
    pub birds: usize,
    /// Seed used.
    pub seed: u64,
    /// Bank model used.
    pub turning: Turning,
    /// One entry per whole simulated second.
    pub per_second: Vec<CoherenceSample>,
    /// Every time the driver hung and the run had to reopen the device.
    pub log: RunLog,
}

impl CoherenceReport {
    /// The last sample, the run's settled state, or `None` for an empty run.
    pub fn summary(&self) -> Option<&CoherenceSample> {
        self.per_second.last()
    }
}

/// Runs the GPU flock for `seconds` simulated seconds, calling `on_sample` once
/// per whole second so a long run can report progress as it goes. Shares the
/// stepping with [`crate::analyze`] through
/// [`crate::analyze::each_second_resilient`], so a driver hang reopens the
/// device and resumes from the last sample instead of ending the run.
pub fn run_streaming(
    gpu: &mut Gpu,
    birds: usize,
    seed: u64,
    seconds: f32,
    turning: Turning,
    mut on_sample: impl FnMut(&CoherenceSample),
    on_resume: impl FnMut(&crate::analyze::Resume),
) -> Result<RunLog, RunError> {
    let mut world = World::new_density_scaled(birds, seed);
    world.turning = turning;
    crate::analyze::each_second_resilient(
        gpu,
        &world,
        seconds,
        |time, on_gpu| on_sample(&sample(on_gpu, time)),
        on_resume,
    )
}

/// Runs the GPU flock for `seconds` simulated seconds, sampling the coherence
/// fields once per whole second.
pub fn run(
    gpu: &mut Gpu,
    birds: usize,
    seed: u64,
    seconds: f32,
    turning: Turning,
) -> Result<CoherenceReport, RunError> {
    let mut per_second = Vec::with_capacity(seconds.max(0.0) as usize);
    let log = run_streaming(
        gpu,
        birds,
        seed,
        seconds,
        turning,
        |s| per_second.push(*s),
        |_| {},
    )?;
    Ok(CoherenceReport {
        birds,
        seed,
        turning,
        per_second,
        log,
    })
}

/// One criterion: its measured value and the range it must lie in.
pub struct Check {
    /// Human-readable name.
    pub name: &'static str,
    /// `"every second"` or `"settled mean"` (last half of the run).
    pub scope: &'static str,
    /// The measured value.
    pub value: f32,
    /// Inclusive lower bound, or `f32::NEG_INFINITY`.
    pub lo: f32,
    /// Inclusive upper bound, or `f32::INFINITY`.
    pub hi: f32,
}

impl Check {
    /// Whether the measured value is inside its declared range.
    pub fn ok(&self) -> bool {
        self.value >= self.lo && self.value <= self.hi
    }

    /// The range as a printable string.
    pub fn range(&self) -> String {
        match (self.lo.is_finite(), self.hi.is_finite()) {
            (true, true) => format!("[{:.2}, {:.2}]", self.lo, self.hi),
            (true, false) => format!(">= {:.2}", self.lo),
            (false, true) => format!("<= {:.2}", self.hi),
            (false, false) => "any".to_string(),
        }
    }
}

/// The verdict over a whole run.
pub struct Verdict {
    /// Every criterion, in the order it is checked.
    pub checks: Vec<Check>,
}

impl Verdict {
    /// Whether every criterion passed.
    pub fn passed(&self) -> bool {
        self.checks.iter().all(Check::ok)
    }

    /// A printable table of the criteria.
    pub fn table(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "  {:<18} {:<13} {:>10} {:>18} {:>6}",
            "criterion", "scope", "value", "range", "pass"
        );
        for c in &self.checks {
            let _ = writeln!(
                out,
                "  {:<18} {:<13} {:>10.3} {:>18} {:>6}",
                c.name,
                c.scope,
                c.value,
                c.range(),
                if c.ok() { "ok" } else { "FAIL" },
            );
        }
        out
    }
}

/// The minimum of `accessor` over every sample. `f32::min` returns the other
/// operand when one is `NaN`, so an unmeasured sample is skipped.
fn every_min(report: &CoherenceReport, accessor: fn(&CoherenceSample) -> f32) -> f32 {
    report
        .per_second
        .iter()
        .map(accessor)
        .fold(f32::INFINITY, f32::min)
}

/// The maximum of `accessor` over every sample (see [`every_min`] for `NaN`).
fn every_max(report: &CoherenceReport, accessor: fn(&CoherenceSample) -> f32) -> f32 {
    report
        .per_second
        .iter()
        .map(accessor)
        .fold(f32::NEG_INFINITY, f32::max)
}

/// The mean of `accessor` over the settled window (last half of the run),
/// ignoring `NaN` samples so a metric that is only measured occasionally still
/// averages over the seconds it was measured.
fn settled_mean(report: &CoherenceReport, accessor: fn(&CoherenceSample) -> f32) -> f32 {
    let samples = &report.per_second;
    if samples.is_empty() {
        return f32::NAN;
    }
    let settled = &samples[samples.len() / 2..];
    let mut sum = 0.0;
    let mut count = 0u32;
    for s in settled {
        let v = accessor(s);
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f32
    } else {
        f32::NAN
    }
}

/// A check over every second of the run.
fn every(name: &'static str, value: f32, lo: f32, hi: f32) -> Check {
    Check {
        name,
        scope: "every second",
        value,
        lo,
        hi,
    }
}

/// A check over the settled (last-half) means.
fn settled(name: &'static str, value: f32, lo: f32, hi: f32) -> Check {
    Check {
        name,
        scope: "settled mean",
        value,
        lo,
        hi,
    }
}

/// Evaluates the declared coherence criteria (see the module docs and
/// `docs/coherence-100k.md`).
pub fn evaluate(report: &CoherenceReport) -> Verdict {
    let inf = f32::INFINITY;
    let checks = vec![
        // Every-second guards: the flock never splits, never leaves the roost
        // area, never flies off the altitude band and never pins its bank.
        every(
            "cluster",
            every_min(report, |s| s.largest_cluster_fraction),
            0.90,
            inf,
        ),
        every("spread max", every_max(report, |s| s.spread), -inf, 250.0),
        every("Rh max", every_max(report, |s| s.max_radius_h), -inf, 375.0),
        every("dz max", every_max(report, |s| s.mean_dz), -inf, 20.0),
        every("hstd", every_min(report, |s| s.height_std), 0.3, 40.0),
        every("hstd max", every_max(report, |s| s.height_std), -inf, 40.0),
        every(
            "clamp max",
            every_max(report, |s| s.clamp_fraction),
            -inf,
            0.05,
        ),
        // Settled means: still one flock, still interacting, still cruising,
        // centred over the roost, and not collapsed into a point.
        settled(
            "cluster",
            settled_mean(report, |s| s.largest_cluster_fraction),
            0.98,
            inf,
        ),
        settled(
            "|N|",
            settled_mean(report, |s| s.mean_neighbours),
            4.0,
            12.0,
        ),
        settled("R", settled_mean(report, |s| s.mean_radius), 0.5, 6.0),
        settled("v", settled_mean(report, |s| s.mean_speed), 7.0, 13.0),
        settled(
            "rc",
            settled_mean(report, |s| s.centroid_radius),
            -inf,
            200.0,
        ),
        settled("spread", settled_mean(report, |s| s.spread), 5.0, inf),
    ];
    Verdict { checks }
}
