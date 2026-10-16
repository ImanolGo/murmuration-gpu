//! `--analyze`: per-second flock statistics from the GPU simulation, in the
//! reference `--analyze` format.
//!
//! This mirrors `murmur-ref`'s `src/bin/analyze.rs` and the terminal version's
//! `--analyze`: the same per-second line ([`murmur_ref::analysis::Stats::line`])
//! and the same summary. The difference is only the engine: the flock is
//! stepped on the GPU and read back once per simulated second, then the *CPU*
//! analysis functions measure it, so the two engines print the same fields and
//! the GPU output can be read against a recorded oracle run.
//!
//! The noise is a documented adaptation (`docs/model.md`): the GPU draws the
//! random force from its own counter-based per-bird stream, not the CPU's
//! ChaCha8. At one seed the two runs are therefore *statistically* comparable,
//! not bit-identical, exactly as `--parity` states. The per-second lines from
//! `--analyze` are the GPU's own trajectory.

use murmur_ref::World;
use murmur_ref::analysis::{self, Stats};
use murmur_ref::flight::Turning;
use murmur_ref::params::DT;
use murmur_ref::stats::{self, FlockStats};

use crate::device::{DeviceError, Gpu, Stalled};
use crate::params::Params;
use crate::sim::{RawState, Sim};

/// One whole-second sample of the GPU flock.
pub struct Sample {
    /// Simulated time \[s\].
    pub time: f32,
    /// The reference statistics at this second.
    pub stats: Stats,
    /// The extended statistics, when `--extended` asked for them.
    pub extended: Option<FlockStats>,
}

/// A finished `--analyze` run.
pub struct Report {
    /// Birds simulated.
    pub birds: usize,
    /// Seed used.
    pub seed: u64,
    /// Bank model used.
    pub turning: Turning,
    /// The statistics of the starting state (the model's initial condition).
    pub initial: Stats,
    /// One entry per whole simulated second.
    pub per_second: Vec<Sample>,
    /// Every time the driver hung and the run had to reopen the device.
    pub log: RunLog,
}

impl Report {
    /// The line printed as `# summary`: the last second sampled, or the initial
    /// state when the run is shorter than a second.
    pub fn summary(&self) -> &Stats {
        self.per_second
            .last()
            .map(|s| &s.stats)
            .unwrap_or(&self.initial)
    }

    /// The extended summary line, when `--extended` was used.
    pub fn summary_extended(&self) -> Option<&FlockStats> {
        self.per_second.last().and_then(|s| s.extended.as_ref())
    }
}

/// Runs the GPU flock for `seconds` simulated seconds, sampling the reference
/// statistics once per whole second. The engine order is the reference's: the
/// initial state is measured before the first step, then after every
/// `1 / DT` steps.
///
/// If the driver hangs the run repairs itself from the last sample
/// ([`each_second_resilient`]).
pub fn run(
    gpu: &mut Gpu,
    birds: usize,
    seed: u64,
    seconds: f32,
    turning: Turning,
    extended: bool,
) -> Result<Report, RunError> {
    let mut world = World::new_density_scaled(birds, seed);
    world.turning = turning;

    let initial = analysis::compute(&world);

    let mut per_second_samples = Vec::with_capacity(seconds.max(0.0) as usize + 1);
    let log = each_second_resilient(
        gpu,
        &world,
        seconds,
        |time, on_gpu| {
            per_second_samples.push(Sample {
                time,
                stats: analysis::compute(on_gpu),
                extended: extended.then(|| stats::compute(on_gpu)),
            });
        },
        |_| {},
    )?;

    Ok(Report {
        birds,
        seed,
        turning,
        initial,
        per_second: per_second_samples,
        log,
    })
}

/// One repair of a long run: the driver hung and the device was reopened.
///
/// Recorded so the acceptance document can say what actually happened rather
/// than only its final numbers.
#[derive(Debug, Clone)]
pub struct Resume {
    /// Simulated time of the last good sample before the hang \[s\].
    pub time: f32,
    /// Why the device was reopened.
    pub reason: String,
}

/// The repairs a long run needed.
#[derive(Debug, Default, Clone)]
pub struct RunLog {
    /// One entry per device reopen, in order.
    pub resumes: Vec<Resume>,
}

impl RunLog {
    /// One `#`-prefixed line per repair, for a run's header.
    pub fn lines(&self) -> Vec<String> {
        self.resumes
            .iter()
            .enumerate()
            .map(|(i, r)| format!("# resume {} at t={:.1}s: {}", i + 1, r.time, r.reason))
            .collect()
    }
}

/// How many repairs may fail in a row (no new sample) before a run gives up.
///
/// A repair that restores nothing means the driver is not coming back, and
/// retrying would loop for ever: `reopen_with_retry` already gives each attempt
/// a fresh device and several goes at opening it, so a few failed repairs are
/// conclusive. Giving up with an error is honest; looping silently is not.
const MAX_RESUME_TRIES: u32 = 3;

/// Why a resilient run stopped.
#[derive(Debug)]
pub enum RunError {
    /// The device could not be reopened after a stall.
    Reopen(DeviceError),
    /// The driver kept hanging without progress: repairs at [`NoProgress::time`]
    /// did not produce a single new sample, so the run was abandoned instead of
    /// looped.
    NoProgress {
        /// Simulated time of the last good sample \[s\].
        time: f32,
        /// Repairs attempted at that time without a new sample.
        tries: u32,
    },
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Reopen(err) => write!(f, "reopening the device after a stall failed: {err}"),
            RunError::NoProgress { time, tries } => write!(
                f,
                "the GPU hung {tries} times at t={time:.1}s with no sample in between; \
                 the driver is not recovering, so the run is stopping (the trajectory is \
                 resumable from the last sample)"
            ),
        }
    }
}

impl std::error::Error for RunError {}

impl From<DeviceError> for RunError {
    fn from(error: DeviceError) -> Self {
        RunError::Reopen(error)
    }
}

/// Steps `world` on the GPU for `seconds` simulated seconds and calls `f` once
/// per whole second with the flock read back to the CPU, reopening the device
/// and resuming from the last sample if it stalls.
///
/// Shared by [`run`] and [`crate::coherence::run`] so the two sample the same
/// trajectory at the same instants. Submissions are batched rather than one
/// command buffer per second: at large bird counts a single `decide` can take
/// seconds, so a whole second's worth of dispatches in one submission can
/// outlive the driver's GPU watchdog and lose the device. Batching keeps every
/// submission short; the sampling is still per second.
///
/// # Resuming after a driver hang
///
/// The development laptop's NVIDIA driver hangs on a long continuous compute
/// run (`Xid 109`, `CTX SWITCH TIMEOUT`; a known driver bug, see
/// [`crate::device::WAIT_BUDGET`]). The hang kills a device but not the run: the
/// flock's state at the last sampled second is kept as a [`RawState`],
/// including each bird's own random stream, so reopening the device and
/// rebuilding the flock from it continues the *same* trajectory — the resumed
/// steps are the steps the uninterrupted run would have taken. The
/// [`RunLog`] records every repair.
pub(crate) fn each_second_resilient(
    gpu: &mut Gpu,
    world: &World,
    seconds: f32,
    mut f: impl FnMut(f32, &World),
    mut on_resume: impl FnMut(&Resume),
) -> Result<RunLog, RunError> {
    let base = world.snapshot();
    let template = Params::from_snapshot(&base);
    let per_second = (1.0 / DT).round() as usize;
    let steps = (seconds / DT).max(0.0) as usize;

    const BATCH: usize = 10;

    // The state a repair resumes from is the world's own starting state, at step
    // zero — every caller hands in a fresh world. A start part-way through (a
    // checkpoint read back from disk) would need that offset added to the step
    // count, the whole-second sampling and `raw.step` together.
    debug_assert!(
        base.time == 0.0,
        "a resilient run starts from a fresh world, not one already at t={}",
        base.time
    );

    // The state at the last sampled second (step zero before the first read) is
    // the one thing a repair needs, because it is where the trajectory resumes.
    let mut raw = RawState {
        birds: Sim::birds_from_snapshot(&base),
        time: base.time,
        step: 0,
    };
    let mut sim = Sim::from_snapshot(gpu, &base);
    let mut log = RunLog::default();
    let mut done = 0;
    // The step count `raw` belongs to. A repair rebuilds the flock from `raw`,
    // so the run's step count has to rewind to this with it; leaving `done` at
    // the failed batch's end desynchronised the clock from the flock (the run
    // then sampled at 0.1 s steps instead of whole seconds).
    let mut sampled = 0;
    let mut tries = 0;

    while done < steps {
        let this = BATCH.min(steps - done);
        let boundary = (done + this).is_multiple_of(per_second);
        match sample_batch(gpu, &mut sim, this, boundary, &mut raw) {
            Ok(Some(on_gpu)) => {
                done += this;
                sampled = done;
                tries = 0;
                f(raw.time, &on_gpu);
            }
            Ok(None) => done += this,
            Err(stalled) => {
                done = sampled;
                tries += 1;
                if tries > MAX_RESUME_TRIES {
                    return Err(RunError::NoProgress {
                        time: raw.time,
                        tries,
                    });
                }
                let resume = Resume {
                    time: raw.time,
                    reason: stalled.to_string(),
                };
                on_resume(&resume);
                log.resumes.push(resume);
                // The device knows how it was opened, so the run can reopen the
                // same request without the caller carrying it.
                *gpu = reopen_with_retry(gpu)?;
                sim = Sim::from_raw(gpu, template, base.seed, &raw);
            }
        }
    }
    Ok(log)
}

/// Opens the device `gpu` was using, retrying briefly.
///
/// After a hang the driver needs a moment to retire the dead context, so a
/// single attempt can fail on a device that is fine a second later; and a run
/// that is hours long must not end because of a moment's flakiness.
fn reopen_with_retry(gpu: &Gpu) -> Result<Gpu, DeviceError> {
    let request = gpu.request();
    const ATTEMPTS: u32 = 5;
    let mut last = None;
    for attempt in 0..ATTEMPTS {
        match request.open() {
            Ok(fresh) => return Ok(fresh),
            Err(error) => {
                last = Some(error);
                if attempt + 1 < ATTEMPTS {
                    std::thread::sleep(std::time::Duration::from_secs(3 * u64::from(attempt + 1)));
                }
            }
        }
    }
    Err(last.expect("at least one open attempt was made"))
}

/// Advances `sim` by `this` steps and, at a whole second, reads the flock back
/// as a [`World`]. `raw` is replaced only when the read succeeds, so a stall
/// always leaves it at the last sample a repair can resume from.
fn sample_batch(
    gpu: &Gpu,
    sim: &mut Sim,
    this: usize,
    boundary: bool,
    raw: &mut RawState,
) -> Result<Option<World>, Stalled> {
    sim.advance(gpu, this as u32);
    if !boundary {
        return Ok(None);
    }
    let next = sim.raw_state(gpu)?;
    let on_gpu = World::restore(&sim.snapshot_from(&next));
    *raw = next;
    Ok(Some(on_gpu))
}
