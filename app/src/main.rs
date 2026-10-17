//! murmuration-gpu — starlings at dusk, on your GPU.
//!
//! Phase 0 scaffold: open a window, create the wgpu device, print the
//! adapter it landed on, and clear to the dusk colour of the reference
//! palette. The simulation and the renderer arrive with the later phases
//! (see PLAN.md). Run on the discrete GPU of a hybrid-graphics laptop with
//! `prime-run` (see docs/setup.md) and check the printed adapter before
//! trusting any measurement.
//!
//! Phase 2 adds a headless `--parity` mode: it runs the CPU oracle and the GPU
//! simulation from the same state and compares the reference statistics each
//! second, exactly what phase 2's acceptance asks for.
//!
//! Phase 3 adds `--bench`: per-pass GPU timestamps at 100k, 1M and 3M birds,
//! from timestamp queries (AGENTS.md forbids any other source for performance).
//!
//! Phase 4 adds `--analyze`: the GPU flock's per-second statistics in the
//! reference `--analyze` format, and `--coherence`: the long-run sampler and the
//! criteria verdict declared in `docs/coherence-100k.md`.
//!
//! Phase 5 adds `--still PATH`: simulate a flock, frame it with the ground
//! camera and render one tone-mapped frame to a PNG (the acceptance still).
//!
//! Usage:
//!   prime-run cargo run -p murmuration-gpu --release                              # window
//!   cargo run -p murmuration-gpu --release -- --parity --birds 2000 --seconds 60
//!   cargo run -p murmuration-gpu --release -- --bench
//!   cargo run -p murmuration-gpu --release -- --analyze 120 --birds 50000
//!   cargo run -p murmuration-gpu --release -- --coherence 600 --birds 100000
//!   cargo run -p murmuration-gpu --release -- --still docs/still.png --birds 1000000

use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

use bytemuck::Zeroable;
use glam::Vec3;
use murmur_gpu::bird::Bird;
use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::sim::Sim;
use murmur_gpu::{FLAP_AMPLITUDE, Turning, World, agitation, analyze, bench, coherence, parity};
use murmur_render::camera::{Camera, FlockView};
use murmur_render::mesh::WINGSPAN;
use murmur_render::png;
use murmur_render::renderer::{LOD_LEVELS, Renderer as FrameRenderer, Scene, TIMESTAMP_SLOTS};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

/// The command line. Hand-rolled for now; PLAN.md phase 8 moves the app to a
/// `clap` CLI matching the terminal version.
#[derive(Debug, Clone)]
struct Args {
    /// Run the headless CPU/GPU parity check instead of the window.
    parity: bool,
    /// Run the headless per-pass benchmark instead of the window.
    bench: bool,
    /// Run the headless `--analyze` for this many simulated seconds.
    analyze: Option<f32>,
    /// Print the extended statistics line each second under `--analyze`.
    extended: bool,
    /// Run the light coherence sampler for this many simulated seconds and
    /// print the criteria verdict.
    coherence: Option<f32>,
    /// Render one still frame of the flock to this PNG path instead of opening
    /// the window.
    still: Option<String>,
    /// Capture a frame sequence into this directory (the phase-6 fly-past
    /// recording and the shimmer statistic).
    capture: Option<String>,
    /// Record one evening as a time-lapse into this directory (PLAN.md phase 8).
    record: Option<String>,
    /// Sweep one bird across every LOD threshold and print the transition
    /// measurement instead of opening the window.
    lod_sweep: bool,
    /// Run the agitation-wave experiment headless and exit.
    agitation: bool,
    /// A falcon hunts from the start.
    falcon: bool,
    /// Hold the display phase: no evening arc.
    no_arc: bool,
    /// Length of one evening in minutes.
    arc_minutes: f32,
    /// Copying threshold k for the zigzag wave (1–7).
    zigzag_k: u32,
    /// Frames in a `--capture` sequence.
    frames: u32,
    /// How far the `--capture` lens zooms in over the recording (1.0 = fixed).
    zoom: f32,
    /// Bank model the shader runs (base or inertial).
    turning: Turning,
    /// Number of birds.
    birds: usize,
    /// Whether `--birds` was given (the benchmark has its own default set).
    birds_given: bool,
    /// First seed; the sweep uses `seed .. seed + seeds`.
    seed: u64,
    /// Simulated seconds per seed.
    seconds: u32,
    /// Number of seeds.
    seeds: u32,
    /// Timed steps per bird count in `--bench`.
    steps: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            parity: false,
            bench: false,
            analyze: None,
            extended: false,
            coherence: None,
            still: None,
            capture: None,
            record: None,
            lod_sweep: false,
            agitation: false,
            falcon: false,
            no_arc: false,
            arc_minutes: 15.0,
            zigzag_k: murmur_ref::params::ZIGZAG_K,
            frames: 120,
            zoom: CAPTURE_ZOOM,
            turning: Turning::Base,
            birds: 2000,
            birds_given: false,
            seed: 1,
            seconds: 60,
            seeds: 5,
            steps: 10,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--parity" => args.parity = true,
            "--bench" => args.bench = true,
            "--analyze" => {
                args.analyze = Some(value(&mut it, "--analyze")?.parse().map_err(err)?);
            }
            "--extended" => args.extended = true,
            "--coherence" => {
                args.coherence = Some(value(&mut it, "--coherence")?.parse().map_err(err)?);
            }
            "--still" => args.still = Some(value(&mut it, "--still")?),
            "--capture" => args.capture = Some(value(&mut it, "--capture")?),
            "--record" => args.record = Some(value(&mut it, "--record")?),
            "--lod-sweep" => args.lod_sweep = true,
            "--agitation" => args.agitation = true,
            "--falcon" => args.falcon = true,
            "--no-arc" => args.no_arc = true,
            "--arc-minutes" => {
                args.arc_minutes = value(&mut it, "--arc-minutes")?.parse().map_err(err)?;
            }
            "--zigzag-k" => {
                args.zigzag_k = value(&mut it, "--zigzag-k")?.parse().map_err(err)?;
                if !(1..=7).contains(&args.zigzag_k) {
                    return Err(format!("--zigzag-k: {} is outside 1-7", args.zigzag_k));
                }
            }
            "--frames" => args.frames = value(&mut it, "--frames")?.parse().map_err(err)?,
            "--zoom" => args.zoom = value(&mut it, "--zoom")?.parse().map_err(err)?,
            "--turning" => {
                let value = value(&mut it, "--turning")?;
                args.turning = match value.as_str() {
                    "base" => Turning::Base,
                    "inertial" => Turning::Inertial,
                    other => return Err(format!("--turning: unknown value {other:?}")),
                };
            }
            "--birds" => {
                args.birds = value(&mut it, "--birds")?.parse().map_err(err)?;
                args.birds_given = true;
            }
            "--seed" => args.seed = value(&mut it, "--seed")?.parse().map_err(err)?,
            "--seconds" => args.seconds = value(&mut it, "--seconds")?.parse().map_err(err)?,
            "--seeds" => args.seeds = value(&mut it, "--seeds")?.parse().map_err(err)?,
            "--steps" => args.steps = value(&mut it, "--steps")?.parse().map_err(err)?,
            "-h" | "--help" => {
                println!(
                    "murmuration-gpu [--parity | --bench | --analyze SECONDS | \
                     --coherence SECONDS | --still PATH | --capture DIR | \
                     --record DIR | --lod-sweep | --agitation] [--birds N] [--seed S] [--seconds T] \
                     [--seeds K] [--steps N] [--frames N] [--zoom F] [--falcon] \
                     [--zigzag-k K] [--extended] [--turning base|inertial]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(args)
}

fn value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn err<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

/// The headless CPU/GPU parity check (PLAN.md phase 2 acceptance).
fn run_parity(args: &Args) -> Result<(), Box<dyn Error>> {
    let gpu = open()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --parity birds={} seed={}..+{} seconds={}",
        args.birds, args.seed, args.seeds, args.seconds
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    let mut all_ok = true;
    for i in 0..args.seeds {
        let seed = args.seed + i as u64;
        let report = parity::run_seed(&gpu, seed, args.birds, args.seconds);
        println!("# seed {seed}");
        if i == 0 {
            // One seed of per-second lines; the rest are summarised by their
            // settled table so the output stays readable.
            for (time, cpu, on_gpu) in &report.per_second {
                println!("t={time:6.1}s cpu {}", cpu.line());
                println!("         gpu {}", on_gpu.line());
            }
        }
        print!(
            "{}",
            parity::table(&format!("seed {seed} settled"), &report.settled)
        );
        all_ok &= report.passed();
    }
    println!("# parity {}", if all_ok { "PASS" } else { "FAIL" });
    Ok(())
}

/// Opens a headless device, or explains why it could not.
fn open() -> Result<Gpu, Box<dyn Error>> {
    match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => Ok(gpu),
        Err(DeviceError::NoAdapter(error)) => {
            Err(format!("no suitable GPU adapter: {error}").into())
        }
        Err(error) => Err(error.into()),
    }
}

/// The headless per-pass benchmark (PLAN.md phase 3 acceptance).
fn run_bench(args: &Args) -> Result<(), Box<dyn Error>> {
    let gpu = match device::headless_with(
        wgpu::PowerPreference::HighPerformance,
        wgpu::Features::TIMESTAMP_QUERY,
    ) {
        Ok(gpu) => gpu,
        Err(DeviceError::NoAdapter(error)) => {
            return Err(format!("no suitable GPU adapter: {error}").into());
        }
        Err(error) => return Err(error.into()),
    };
    let info = gpu.info();
    println!("# murmuration-gpu --bench steps={}", args.steps);
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        println!("# this adapter has no TIMESTAMP_QUERY feature; --bench skipped");
        return Ok(());
    }

    // The acceptance asks for 100k, 1M and 3M; `--birds` picks one instead.
    let counts: Vec<usize> = if args.birds_given {
        vec![args.birds]
    } else {
        vec![100_000, 1_000_000, 3_000_000]
    };
    for birds in counts {
        let report = bench::run(&gpu, birds, args.steps);
        println!(
            "# {} birds, {} timed steps (after warm-up)",
            report.birds, report.steps
        );
        println!("  {:<18} {:>12}", "pass", "ms");
        for pass in &report.passes {
            println!("  {:<18} {:>12.4}", pass.pass, pass.millis);
        }
        println!("  {:<18} {:>12.4}", "step total", report.step_millis());
        println!(
            "  {:<18} {:>12.0}",
            "bird-steps/s",
            report.birds_per_second()
        );
    }
    Ok(())
}

/// The headless per-second statistics (PLAN.md phase 4), in the reference
/// `--analyze` format.
fn run_analyze(args: &Args, seconds: f32) -> Result<(), Box<dyn Error>> {
    let mut gpu = open()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --analyze birds={} seed={} seconds={} turning={:?}",
        args.birds, args.seed, seconds, args.turning
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    let report = analyze::run(
        &mut gpu,
        args.birds,
        args.seed,
        seconds,
        args.turning,
        args.extended,
    )?;
    for line in report.log.lines() {
        println!("{line}");
    }
    for sample in &report.per_second {
        println!("t={:6.1}s {}", sample.time, sample.stats.line());
        if let Some(extended) = &sample.extended {
            println!("        {}", extended.line());
        }
    }
    println!("# summary  {}", report.summary().line());
    Ok(())
}

/// The long coherence run (PLAN.md phase 4): the light per-second sample and the
/// criteria verdict from `docs/coherence-100k.md`.
///
/// This is separate from `--analyze` because the full reference metric set costs
/// ~26 s per simulated second at 100,000 birds; see
/// [`murmur_gpu::coherence`].
fn run_coherence(args: &Args, seconds: f32) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut gpu = open()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --coherence birds={} seed={} seconds={} turning={:?}",
        args.birds, args.seed, seconds, args.turning
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    let mut per_second = Vec::new();
    let log = coherence::run_streaming(
        &mut gpu,
        args.birds,
        args.seed,
        seconds,
        args.turning,
        |sample| {
            println!("t={:6.1}s {}", sample.time, sample.line());
            // Flush each second: a run this long (minutes to hours) must report
            // progress, and a stalled run then shows up as a file that stops
            // growing instead of an empty file that looks the same either way.
            let _ = io::stdout().flush();
            per_second.push(*sample);
        },
        |resume| {
            // The driver hung and the device was reopened; the run continues
            // from the last second and the trajectory is unchanged.
            println!("# resume at t={:.1}s: {}", resume.time, resume.reason);
            let _ = io::stdout().flush();
        },
    )?;
    let report = coherence::CoherenceReport {
        birds: args.birds,
        seed: args.seed,
        turning: args.turning,
        per_second,
        log,
    };
    // The wall clock is part of the result, not decoration: these runs take
    // hours and the cost is the thing phase 4 kept mis-estimating, so the record
    // carries what it actually cost.
    let wall = started.elapsed().as_secs_f64();
    println!(
        "# wall {wall:.0}s for {seconds:.0}s simulated ({:.2} s per simulated second)",
        wall / seconds.max(1.0) as f64
    );
    if !report.log.resumes.is_empty() {
        println!(
            "# the device was reopened {} time(s); the resumed trajectory is the \
             uninterrupted one",
            report.log.resumes.len()
        );
    }
    if let Some(summary) = report.summary() {
        println!("# summary  {}", summary.line());
    }
    let verdict = coherence::evaluate(&report);
    print!("{}", verdict.table());
    println!(
        "# coherence {}",
        if verdict.passed() { "PASS" } else { "FAIL" }
    );
    Ok(())
}

/// The first real frame (PLAN.md phase 5): simulate the flock, frame it with
/// the ground camera and write one tone-mapped PNG.
///
/// The flock is stepped on the GPU, read back once so the camera can frame it,
/// and the adaptive exposure is warmed over a few frames before the captured
/// one, exactly as the windowed renderer does.
fn run_still(args: &Args, path: &str, seconds: f32) -> Result<(), Box<dyn Error>> {
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;
    const DT: f32 = 1.0 / 30.0;

    let started = Instant::now();
    // The frame time is an acceptance measurement, so ask for the timestamp
    // feature; fall back to the plain device (and say the time is unmeasured)
    // when the adapter cannot write stamps.
    let gpu = match open_timed() {
        Ok(gpu) => gpu,
        Err(_) => open()?,
    };
    let info = gpu.info();
    println!(
        "# murmuration-gpu --still {path} birds={} seed={} seconds={} {WIDTH}x{HEIGHT}",
        args.birds, args.seed, seconds
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    let mut world = World::new_density_scaled(args.birds, args.seed);
    world.turning = args.turning;
    let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
    if !args.no_arc {
        sim.enable_arc(&gpu, args.arc_minutes, args.seed);
    }
    if args.falcon {
        sim.spawn_falcon(&gpu, args.seed, world.z0);
    }
    sim.advance_seconds(&gpu, seconds)?;

    let birds = sim.read_birds(&gpu)?;
    let (positions, velocities) = flock_view(&birds);
    let mut camera = Camera::new();
    camera.update(
        &FlockView {
            positions: &positions,
            velocities: &velocities,
            time: sim.time(),
        },
        DT,
        WIDTH as f32 / HEIGHT as f32,
    );

    // The tone-map pass writes to an `*Srgb` target, so the readback holds the
    // display-encoded bytes the PNG needs directly.
    let mut renderer =
        FrameRenderer::new(&gpu, WIDTH, HEIGHT, wgpu::TextureFormat::Rgba8UnormSrgb)?;
    renderer.set_birds(&gpu, sim.bird_buffer(), sim.len() as u32);
    let target = renderer.create_target(&gpu, WIDTH, HEIGHT);
    let falcon = if args.falcon {
        let f = sim.falcon(&gpu)?.position();
        println!("# falcon at ({:.0}, {:.0}, {:.0})", f.x, f.y, f.z);
        Some(f)
    } else {
        None
    };
    let evening = evening_of(args, sim.time());

    let mut scene = Scene::from_camera(
        &camera,
        WIDTH,
        HEIGHT,
        evening,
        renderer.exposure(),
        FLAP_AMPLITUDE,
        sim.time(),
    )
    .with_falcon(falcon)
    .with_reeds(murmur_render::sky::reed_visibility(&camera));
    // The first frame fixes the exposure from the flock's 80th-percentile
    // optical depth; the rest let it settle, and let the temporal accumulation
    // average the jittered samples of the frozen scene.
    const SETTLE: u32 = 24;
    for frame in 0..SETTLE {
        scene.exposure = renderer.exposure();
        renderer.set_scene(&gpu, scene);
        renderer.set_history_weight(Some(1.0 / (frame as f32 + 1.0)));
        renderer.render(&gpu, &target.view, args.birds as u32, DT);
    }
    renderer.set_history_weight(None);

    let rgba = renderer.read_target(&gpu, &target, WIDTH, HEIGHT);
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    png::write_rgb(path, WIDTH, HEIGHT, &rgb)?;
    println!("# exposure {:.3}", renderer.exposure());
    let lod = renderer.lod_counts(&gpu);
    println!(
        "# lod mesh={} impostor={} splat={} (birds {})",
        lod[0], lod[1], lod[2], args.birds
    );
    // The frame time is a GPU measurement or nothing at all (AGENTS.md): it
    // comes from the renderer's timestamp query, never from a wall clock.
    match renderer.frame_times_ms(&gpu) {
        Some(times) => println!("# render {}", timing_line(&times)),
        None => println!("# render unmeasured: this adapter cannot write timestamps"),
    }
    // How long the still cost is part of the record: at a million birds this is
    // a long simulation, not a screenshot.
    let wall = started.elapsed().as_secs_f64();
    println!(
        "# wall {wall:.0}s to simulate {seconds:.0}s and render ({:.2} s per simulated second)",
        wall / seconds.max(1.0) as f64
    );
    println!("# wrote {path}");
    Ok(())
}

/// Opens a headless device that can also write GPU timestamps, or explains why
/// it could not. Every frame time this app reports comes from a timestamp query
/// (AGENTS.md), so the render modes ask for the feature up front and say so when
/// the adapter cannot provide it.
fn open_timed() -> Result<Gpu, Box<dyn Error>> {
    match device::headless_with(
        wgpu::PowerPreference::HighPerformance,
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    ) {
        Ok(gpu) => Ok(gpu),
        Err(DeviceError::NoAdapter(error)) => {
            Err(format!("no suitable GPU adapter: {error}").into())
        }
        Err(error) => Err(error.into()),
    }
}

/// The frame's GPU timings, in the order the passes run. Slot 0 is the frame's
/// start, so this is a frame time, not a wall clock.
fn timing_line(times: &[f32; TIMESTAMP_SLOTS]) -> String {
    format!(
        "total {:.2} ms/frame: lod {:.2}, birds {:.2}, exposure {:.2}, taa {:.2}, tonemap {:.2}",
        times[5],
        times[1],
        times[2] - times[1],
        times[3] - times[2],
        times[4] - times[3],
        times[5] - times[4],
    )
}

/// What a frame-to-frame change amounts to, in equivalent black pixels, and the
/// number of pixels whose worst channel moved by more than 16.
///
/// Black pixels are the unit: one fully dark pixel of the whole frame counts as
/// one. A bird 12 px across is worth about ten of them, so a change of one or
/// two is a small fraction of a bird and a change of ten is a whole one. This is
/// the shimmer statistic: a bird crossing a pixel boundary between two frames
/// shows up here whether or not the eye would call it a flicker.
fn change(a: &[u8], b: &[u8]) -> (f32, u32) {
    const SPARKLE: i32 = 16;
    let mut sum = 0u64;
    let mut sparkles = 0u32;
    for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        let mut worst = 0;
        for k in 0..3 {
            let d = (i32::from(pa[k]) - i32::from(pb[k])).abs();
            sum += d as u64;
            worst = worst.max(d);
        }
        if worst > SPARKLE {
            sparkles += 1;
        }
    }
    // Absorbed light over all three channels, in black pixels: 3 channels of a
    // fully dark pixel are worth 3 x 255.
    (sum as f32 / (3.0 * 255.0), sparkles)
}

/// The light a frame absorbs against a bird-free `background`, in equivalent
/// black pixels: the bird's total ink on the screen.
fn ink(background: &[u8], frame: &[u8]) -> f32 {
    let mut sum = 0u64;
    for (b, f) in background
        .as_chunks::<4>()
        .0
        .iter()
        .zip(frame.as_chunks::<4>().0)
    {
        for k in 0..3 {
            sum += u64::from(b[k].saturating_sub(f[k]));
        }
    }
    sum as f32 / (3.0 * 255.0)
}

/// Which LOD levels drew a bird, as `m`, `i`, `s` joined by `+`.
fn level_tag(counts: &[u32; LOD_LEVELS]) -> String {
    let mut tag = String::new();
    for (count, name) in counts.iter().zip(["m", "i", "s"]) {
        if *count > 0 {
            if !tag.is_empty() {
                tag.push('+');
            }
            tag.push_str(name);
        }
    }
    if tag.is_empty() {
        tag.push('-');
    }
    tag
}

/// One `--capture` pass: the per-frame shimmer statistic and the GPU render
/// time, with the PNGs written when `dir` is given.
struct CaptureReport {
    /// Change against the previous frame, per frame, in black pixels.
    diffs: Vec<f32>,
    /// Pixels whose worst channel moved by more than 16, per frame.
    sparkles: Vec<u32>,
    /// GPU render time of each frame, from its timestamps.
    render_ms: Vec<f32>,
    /// LOD instance counts of the first and last frame: `(first, last)`.
    lod: ([u32; LOD_LEVELS], [u32; LOD_LEVELS]),
}

impl CaptureReport {
    fn mean(values: &[f32]) -> f32 {
        values.iter().sum::<f32>() / values.len().max(1) as f32
    }

    fn worst(values: &[f32]) -> f32 {
        values.iter().copied().fold(0.0f32, f32::max)
    }

    fn line(&self, name: &str) -> String {
        let sparkles: Vec<f32> = self.sparkles.iter().map(|&s| s as f32).collect();
        let render = if self.render_ms.is_empty() {
            "unmeasured".to_string()
        } else {
            format!("{:.2} ms/frame", Self::mean(&self.render_ms))
        };
        format!(
            "shimmer {name}: change {:.1} black px/frame (worst {:.1}), over-16 pixels {:.0}/frame, \
             lod {} -> {}, render {render}",
            Self::mean(&self.diffs),
            Self::worst(&self.diffs),
            Self::mean(&sparkles),
            lod_line(&self.lod.0),
            lod_line(&self.lod.1),
        )
    }
}

/// `mesh=… impostor=… splat=…`, the LOD instance counts of one frame.
fn lod_line(counts: &[u32; LOD_LEVELS]) -> String {
    format!(
        "mesh={} impostor={} splat={}",
        counts[0], counts[1], counts[2]
    )
}

/// What a `--capture` pass is: the moving flock or the frozen one, with the
/// temporal accumulation on or off, and whether it writes the recording.
#[derive(Clone)]
struct CapturePass {
    /// `true` for the static camera on a distant flock.
    frozen: bool,
    /// `true` to accumulate, `false` to show the raw frames.
    taa: bool,
    /// The directory the frames are written to, for the recorded pass.
    dir: Option<String>,
}

impl CapturePass {
    fn name(&self) -> &'static str {
        match (self.frozen, self.taa) {
            (false, true) => "fly-past taa",
            (false, false) => "fly-past raw",
            (true, true) => "static   taa",
            (true, false) => "static   raw",
        }
    }
}

/// The recorded fly-past and the shimmer statistics (PLAN.md phase 6
/// acceptance).
///
/// Two sequences are captured, each twice, once with the temporal accumulation
/// on and once without.
///
/// * The fly-past: the flock is simulated and framed, and the lens zooms in over
///   the capture, so the flock grows and every bird crosses every LOD threshold
///   within the recording. That is what makes a pop visible if there is one, and
///   it is the "recorded fly-past" the acceptance asks for.
/// * The static camera on a distant flock: the same flock frozen, framed four
///   times wider so the birds are sub-pixel, and sampled frame after frame with
///   the temporal-AA jitter moving. Nothing in the scene moves, so every
///   frame-to-frame difference is a sampling artefact of sub-pixel birds — the
///   shimmer the acceptance asks about, isolated from motion.
///
/// The frames written to `dir` are the accumulated fly-past.
fn run_capture(args: &Args, dir: &str, seconds: f32) -> Result<(), Box<dyn Error>> {
    const WIDTH: u32 = 960;
    const HEIGHT: u32 = 540;
    let gpu = open_timed()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --capture {dir} birds={} seed={} seconds={} frames={} {WIDTH}x{HEIGHT}",
        args.birds, args.seed, seconds, args.frames
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    println!(
        "# the moving flock steps six integration steps (0.03 s) per frame, played back at \
         30 fps; the lens zooms 1.00 -> {:.2}, so every bird grows {:.1}x and \
         crosses every LOD threshold in view",
        args.zoom,
        1.0 / args.zoom
    );

    let started = Instant::now();
    // The static passes get their own directories: their frames are the two
    // halves of the shimmer comparison.
    let static_taa = format!("{dir}/static-taa");
    let static_raw = format!("{dir}/static-raw");
    for path in [dir, static_taa.as_str(), static_raw.as_str()] {
        std::fs::create_dir_all(path)?;
    }
    let passes = [
        CapturePass {
            frozen: false,
            taa: true,
            dir: Some(dir.to_string()),
        },
        CapturePass {
            frozen: false,
            taa: false,
            dir: None,
        },
        CapturePass {
            frozen: true,
            taa: true,
            dir: Some(static_taa),
        },
        CapturePass {
            frozen: true,
            taa: false,
            dir: Some(static_raw),
        },
    ];
    let mut reports = Vec::new();
    for pass in passes {
        let report = capture_pass(&gpu, args, seconds, pass.clone())?;
        println!("# {}", report.line(pass.name()));
        reports.push(report);
    }

    let ratio = |with: &CaptureReport, without: &CaptureReport| {
        CaptureReport::mean(&with.diffs) / CaptureReport::mean(&without.diffs).max(1e-6)
    };
    println!(
        "# temporal accumulation removes {:.0}% of the in-motion change ({:.2}x) and {:.0}% of \
         the static flock's sub-pixel change ({:.2}x)",
        100.0 * (1.0 - ratio(&reports[0], &reports[1])),
        ratio(&reports[0], &reports[1]),
        100.0 * (1.0 - ratio(&reports[2], &reports[3])),
        ratio(&reports[2], &reports[3]),
    );
    println!(
        "# {} frames written to {dir} in {:.0}s",
        args.frames,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// One pass over the capture frames, from a fresh, identically seeded
/// simulation, so the two accumulated-or-not passes see the same frames.
fn capture_pass(
    gpu: &Gpu,
    args: &Args,
    seconds: f32,
    pass: CapturePass,
) -> Result<CaptureReport, Box<dyn Error>> {
    const WIDTH: u32 = 960;
    const HEIGHT: u32 = 540;
    const DT: f32 = 1.0 / 30.0;
    // One frame of the recording is six integration steps (0.03 s), played back
    // at 30 fps.
    const STEPS: u32 = 6;
    // The static flock is framed this many times wider, so its birds are
    // sub-pixel: a distant flock.
    const DISTANT: f32 = 4.0;
    // Enough frames to see the jitter's whole cycle and to let the accumulation
    // converge.
    const STATIC_FRAMES: u32 = 30;

    let mut world = World::new_density_scaled(args.birds, args.seed);
    world.turning = args.turning;
    let mut sim = Sim::from_snapshot(gpu, &world.snapshot());
    if !args.no_arc {
        sim.enable_arc(gpu, args.arc_minutes, args.seed);
    }
    if args.falcon {
        sim.spawn_falcon(gpu, args.seed, world.z0);
    }
    sim.advance_seconds(gpu, seconds)?;

    let birds = sim.read_birds(gpu)?;
    let (positions, velocities) = flock_view(&birds);
    let mut camera = Camera::new();
    camera.update(
        &FlockView {
            positions: &positions,
            velocities: &velocities,
            time: sim.time(),
        },
        DT,
        WIDTH as f32 / HEIGHT as f32,
    );

    let mut renderer = FrameRenderer::new(gpu, WIDTH, HEIGHT, wgpu::TextureFormat::Rgba8UnormSrgb)?;
    renderer.set_birds(gpu, sim.bird_buffer(), sim.len() as u32);
    let target = renderer.create_target(gpu, WIDTH, HEIGHT);
    let mut scene = Scene::from_camera(
        &camera,
        WIDTH,
        HEIGHT,
        evening_of(args, sim.time()),
        renderer.exposure(),
        FLAP_AMPLITUDE,
        sim.time(),
    );
    // The exposure is part of the model's look: settle it before the recording
    // starts, so the frames differ by the birds and not by a drifting exposure.
    for _ in 0..12 {
        scene.exposure = renderer.exposure();
        renderer.set_scene(gpu, scene);
        renderer.render(gpu, &target.view, sim.len() as u32, DT);
    }

    let mut report = CaptureReport {
        diffs: Vec::new(),
        sparkles: Vec::new(),
        render_ms: Vec::new(),
        lod: ([0; LOD_LEVELS], [0; LOD_LEVELS]),
    };
    let frames = if pass.frozen {
        STATIC_FRAMES
    } else {
        args.frames
    };
    // The static pass looks at the same flock from the same camera, four times
    // wider, with the lens and everything else fixed.
    let mut frozen_scene = {
        let mut frozen = scene;
        frozen.fov *= DISTANT;
        frozen
    };
    let mut previous: Option<Vec<u8>> = None;
    for frame in 0..frames {
        if pass.frozen {
            // Nothing moves: the same flock, the same camera, the same lens. What
            // is left between frames is the jitter's sampling of sub-pixel
            // shapes.
            frozen_scene.exposure = renderer.exposure();
            scene = frozen_scene;
        } else {
            sim.advance(gpu, STEPS);
            let birds = sim.read_birds(gpu)?;
            let (positions, velocities) = flock_view(&birds);
            camera.update(
                &FlockView {
                    positions: &positions,
                    velocities: &velocities,
                    time: sim.time(),
                },
                DT,
                WIDTH as f32 / HEIGHT as f32,
            );
            scene = Scene::from_camera(
                &camera,
                WIDTH,
                HEIGHT,
                evening_of(args, sim.time()),
                renderer.exposure(),
                FLAP_AMPLITUDE,
                sim.time(),
            )
            .with_falcon(if sim.falcon_active() {
                Some(sim.falcon(gpu)?.position())
            } else {
                None
            })
            .with_reeds(murmur_render::sky::reed_visibility(&camera));
            let t = if frames > 1 {
                frame as f32 / (frames - 1) as f32
            } else {
                0.0
            };
            scene.fov *= 1.0 - (1.0 - args.zoom) * t;
        }
        renderer.set_scene(gpu, scene);
        renderer.set_history_weight(if pass.taa { None } else { Some(1.0) });
        renderer.render(gpu, &target.view, sim.len() as u32, DT);
        let counts = renderer.lod_counts(gpu);
        if frame == 0 {
            report.lod.0 = counts;
        }
        report.lod.1 = counts;
        if let Some(times) = renderer.frame_times_ms(gpu) {
            report.render_ms.push(times[TIMESTAMP_SLOTS - 1]);
        }
        let rgba = renderer.read_target(gpu, &target, WIDTH, HEIGHT);
        if let Some(before) = &previous {
            let (diff, sparkles) = change(before, &rgba);
            report.diffs.push(diff);
            report.sparkles.push(sparkles);
        }
        if let Some(dir) = &pass.dir {
            let rgb: Vec<u8> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            png::write_rgb(format!("{dir}/frame-{frame:04}.png"), WIDTH, HEIGHT, &rgb)?;
        }
        previous = Some(rgba);
    }
    Ok(report)
}

/// One row of the `--lod-sweep` table.
struct SweepRow {
    rho: f32,
    tag: String,
    /// The bird's ink in black pixels.
    ink: f32,
    /// The change against the previous row's frame, in black pixels.
    diff: f32,
    sparkle: u32,
}

/// The LOD transition measurement (PLAN.md phase 6 acceptance): one bird, its
/// projected wingspan swept across every threshold, and what the picture does at
/// each step.
///
/// The bird sits on the view axis and flies across the view, so it stays at the
/// same screen point while ρ changes: the sky, the pose and the exposure are the
/// same at every step, and the only thing that changes is the level the bird is
/// drawn at. The temporal accumulation and the sample jitter are both off, so a
/// step measures the level change and nothing else.
///
/// A bird that changes level must change the picture no more than growing by the
/// same amount does. The verdict compares the worst step across a threshold with
/// the worst step beside it, at the same sizes: the steps away from a threshold
/// are what that growth costs, so a threshold step has to stay in the same
/// league. The ink column is the bird's total darkening, which the shapes are
/// not required to keep equal — the levels carry the same optical *mass*, and
/// the reference's disc spreads it differently from a silhouette.
fn run_lod_sweep(args: &Args) -> Result<(), Box<dyn Error>> {
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    /// Focal length in pixels: fixed, so a distance is a projected size.
    const FOCAL: f32 = 700.0;
    const RHO_MIN: f32 = 0.25;
    const RHO_MAX: f32 = 12.0;
    const RHO_STEP: f32 = 0.25;
    /// The control windows: steps of the same growth away from the thresholds.
    const CONTROL: f32 = 2.0;
    /// How many times a control step a threshold step may be before the change
    /// counts as a jump rather than as growth. A step that moves the picture
    /// twice as much as the same bird growing by the same amount elsewhere is
    /// the point at which the eye reads a level change and not the bird.
    const JUMP: f32 = 2.0;

    let gpu = open_timed()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --lod-sweep {WIDTH}x{HEIGHT} rho {RHO_MIN}..{RHO_MAX} step {RHO_STEP}"
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    // A camera whose focal length is fixed, looking up at 45 degrees, so the
    // sweep is in projected size rather than in lens angle.
    let mut camera = Camera::new();
    camera.forward = Vec3::new(0.0, 0.702, 0.712).normalize();
    camera.up = Vec3::new(0.0, -0.712, 0.702).normalize();
    camera.right = camera.forward.cross(camera.up).normalize();
    camera.fov = 2.0 * (HEIGHT as f32 * 0.5 / FOCAL).atan();
    let eye = camera.eye;
    let forward = camera.forward;
    // The bird crosses the view at the observer's level: seen from below with
    // its wings spread, which is how the flock is mostly seen, and with the same
    // pose at every step.
    let across = Vec3::new(0.0, 0.0, 1.0).cross(forward).normalize();

    let mut world = World::new_density_scaled(1, args.seed);
    world.turning = args.turning;
    let evening = world.evening();

    let mut bird = Bird::zeroed();
    bird.flap_amp = 1.0;
    bird.vel = (across * 8.0).to_array();
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lod-sweep: one bird"),
        size: std::mem::size_of::<Bird>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut renderer =
        FrameRenderer::new(&gpu, WIDTH, HEIGHT, wgpu::TextureFormat::Rgba8UnormSrgb)?;
    renderer.set_birds(&gpu, &buffer, 1);
    renderer.set_jitter(Some([0.0, 0.0]));
    let target = renderer.create_target(&gpu, WIDTH, HEIGHT);
    let mut scene = Scene::from_camera(
        &camera,
        WIDTH,
        HEIGHT,
        evening,
        renderer.exposure(),
        FLAP_AMPLITUDE,
        0.0,
    );

    // The bird is placed on the view axis at the distance its projected wingspan
    // asks for: `rho = WINGSPAN * focal / depth`, the classifier's own formula.
    let place = |bird: &mut Bird, rho: f32| {
        bird.pos = (eye + forward * (WINGSPAN * FOCAL / rho)).to_array();
    };

    // The exposure is held at the floor a dense flock pins it to. A single bird
    // cannot give the histogram a sensible 80th percentile, so letting the
    // exposure adapt here would measure the exposure and not the levels.
    renderer.set_exposure(murmur_render::renderer::EXPOSURE_MIN);
    scene.exposure = renderer.exposure();
    println!(
        "# exposure fixed at {:.3} (the floor a dense flock settles on)",
        renderer.exposure()
    );

    // The sky alone, for the ink measure: the bird pushed behind the observer,
    // where no level draws it.
    bird.pos = (eye - forward * 10.0).to_array();
    gpu.queue
        .write_buffer(&buffer, 0, bytemuck::bytes_of(&bird));
    renderer.set_scene(&gpu, scene);
    renderer.render(&gpu, &target.view, 1, 0.0);
    let background = renderer.read_target(&gpu, &target, WIDTH, HEIGHT);

    println!("#   rho  level      ink   d(ink)   change  sparkle");
    let mut rows: Vec<SweepRow> = Vec::new();
    let mut previous: Option<Vec<u8>> = None;
    let mut rho = RHO_MIN;
    while rho <= RHO_MAX + 1e-6 {
        place(&mut bird, rho);
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&bird));
        renderer.set_scene(&gpu, scene);
        renderer.set_history_weight(Some(1.0));
        renderer.render(&gpu, &target.view, 1, 0.0);
        let counts = renderer.lod_counts(&gpu);
        let rgba = renderer.read_target(&gpu, &target, WIDTH, HEIGHT);
        let ink = ink(&background, &rgba);
        let (diff, sparkle) = match &previous {
            Some(before) => change(before, &rgba),
            None => (0.0, 0),
        };
        let tag = level_tag(&counts);
        let d_ink = rows.last().map_or(0.0, |last| ink - last.ink);
        println!("# {rho:6.2}  {tag:<6} {ink:6.2} {d_ink:+8.2} {diff:8.2} {sparkle:8}");
        rows.push(SweepRow {
            rho,
            tag,
            ink,
            diff,
            sparkle,
        });
        previous = Some(rgba);
        rho += RHO_STEP;
    }

    // The mesh cross-fade, the hard splat/impostor threshold, and the control
    // windows beside each: growth of the same bird, at the same sizes.
    let mesh_lo = murmur_render::renderer::RHO_MESH - murmur_render::renderer::LOD_BAND;
    let mesh_hi = murmur_render::renderer::RHO_MESH + murmur_render::renderer::LOD_BAND;
    let splat = murmur_render::renderer::RHO_SPLAT;
    let mesh_worst = worst_step(&rows, mesh_lo, mesh_hi);
    let mesh_above = worst_step(&rows, mesh_hi, mesh_hi + CONTROL);
    let mesh_below = worst_step(&rows, splat + RHO_STEP, mesh_lo);
    let control = mesh_above.diff.max(mesh_below.diff);
    let control_sparkle = if mesh_above.diff >= mesh_below.diff {
        mesh_above.sparkle
    } else {
        mesh_below.sparkle
    };
    let splat_worst = worst_step(&rows, splat - RHO_STEP, splat + RHO_STEP);
    let splat_above = worst_step(&rows, splat + RHO_STEP, splat + CONTROL + RHO_STEP);
    let splat_below = worst_step(&rows, RHO_MIN, splat - RHO_STEP);
    let splat_control = splat_above.diff.max(splat_below.diff);

    println!(
        "# mesh cross-fade [{mesh_lo:.2}, {mesh_hi:.2}] px: worst step {:.2} black px at rho {:.2} \
         ({} pixels over 16)",
        mesh_worst.diff, mesh_worst.rho, mesh_worst.sparkle
    );
    println!(
        "# control: growth of the same bird over [.. {mesh_lo:.2}] and [{mesh_hi:.2} .. {:.2}]: \
         worst {control:.2} black px ({control_sparkle} pixels over 16)",
        mesh_hi + CONTROL,
    );
    println!(
        "# threshold steps are {:.2}x the control steps; {}",
        mesh_worst.diff / control.max(1e-6),
        if mesh_worst.diff <= JUMP * control {
            "no step at the mesh threshold is a jump"
        } else {
            "A STEP AT THE MESH THRESHOLD IS A JUMP"
        }
    );
    println!(
        "# splat/impostor at {splat:.2} px: step {:.2} black px at rho {:.2} ({} pixels over 16), \
         control {splat_control:.2}; {}",
        splat_worst.diff,
        splat_worst.rho,
        splat_worst.sparkle,
        if splat_worst.diff <= JUMP * splat_control {
            "no step is a jump"
        } else {
            "A STEP AT THE SPLAT THRESHOLD IS A JUMP"
        }
    );
    // The levels the cross-fade runs between, and the mass argument for the ink
    // it moves: the mesh is drawn by a silhouette whose mean projected area is
    // `mesh::mean_projected_area()`, so at equal mass it concentrates the ink the
    // reference's disc spreads out. That is the documented adaptation; this line
    // is its size.
    let at = |rho: f32| {
        rows.iter()
            .find(|r| r.rho >= rho)
            .map_or((0.0, String::from("-")), |r| (r.ink, r.tag.clone()))
    };
    let (ink_lo, tag_lo) = at(mesh_lo);
    let (ink_hi, tag_hi) = at(mesh_hi);
    println!(
        "# ink over the sweep {:.2} black px (rho {RHO_MIN}) -> {:.2} (rho {RHO_MAX}); across \
         the cross-fade {ink_lo:.2} [{tag_lo}] -> {ink_hi:.2} [{tag_hi}]",
        rows.first().map_or(0.0, |r| r.ink),
        rows.last().map_or(0.0, |r| r.ink),
    );
    // What the fade does to the ink, and the reading of it. The reference's disc
    // gives a resolved bird more ink than its own silhouette can hold, so a bird
    // that resolves gets thinner: the mass is the same, the area is not.
    // `docs/model.md` records the adaptation; this is its size, spread over the
    // steps the bird takes to cross the band.
    println!(
        "# the cross-fade moves the ink {ink_lo:.2} -> {ink_hi:.2} black px ({:.2}x) over {:.1} px \
         of growth, spread over {:.0} steps of {RHO_STEP:.2} px",
        ink_lo / ink_hi.max(1e-6),
        mesh_hi - mesh_lo,
        (mesh_hi - mesh_lo) / RHO_STEP,
    );
    Ok(())
}

/// What a step did: the difference, where it ended, and the pixels it moved by
/// more than 16.
struct Step {
    diff: f32,
    rho: f32,
    sparkle: u32,
}

/// The worst step difference whose ρ interval touches `[lo, hi]`, and the ρ it
/// ended at. Row `i`'s difference is against row `i - 1`, so it belongs to the
/// interval between them.
fn worst_step(rows: &[SweepRow], lo: f32, hi: f32) -> Step {
    let mut worst = Step {
        diff: 0.0,
        rho: 0.0,
        sparkle: 0,
    };
    for pair in rows.windows(2) {
        let (before, after) = (&pair[0], &pair[1]);
        if before.rho <= hi && after.rho >= lo && after.diff > worst.diff {
            worst = Step {
                diff: after.diff,
                rho: after.rho,
                sparkle: after.sparkle,
            };
        }
    }
    worst
}

/// Records one evening as a time-lapse (PLAN.md phase 8): `--frames` frames
/// spread evenly over `--arc-minutes`, each rendered and written as a PNG in
/// `dir`. The GIF or WebP is then made with ffmpeg, as `scripts/evening.sh`
/// does, so the recording is of the real app and not of a separate renderer.
fn run_record(args: &Args, dir: &str) -> Result<(), Box<dyn Error>> {
    const WIDTH: u32 = 960;
    const HEIGHT: u32 = 540;
    const DT: f32 = 1.0 / 30.0;
    if args.no_arc {
        return Err("--record needs the evening arc; drop --no-arc".into());
    }
    let frames = args.frames.max(2);
    let duration = (args.arc_minutes * 60.0).max(1.0);
    let per_frame = duration / frames as f32;

    let gpu = open_timed()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --record {dir} birds={} seed={} minutes={} frames={frames} {WIDTH}x{HEIGHT}",
        args.birds, args.seed, args.arc_minutes
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    std::fs::create_dir_all(dir)?;

    let mut world = World::new_density_scaled(args.birds, args.seed);
    world.turning = args.turning;
    let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
    sim.enable_arc(&gpu, args.arc_minutes, args.seed);
    if args.falcon {
        sim.spawn_falcon(&gpu, args.seed, world.z0);
    }

    let mut renderer =
        FrameRenderer::new(&gpu, WIDTH, HEIGHT, wgpu::TextureFormat::Rgba8UnormSrgb)?;
    renderer.set_birds(&gpu, sim.bird_buffer(), sim.len() as u32);
    let target = renderer.create_target(&gpu, WIDTH, HEIGHT);
    let mut camera = Camera::new();
    let mut scene;

    for frame in 0..frames {
        sim.advance_seconds(&gpu, per_frame)?;
        let birds = sim.read_birds(&gpu)?;
        let (positions, velocities) = flock_view(&birds);
        camera.update(
            &FlockView {
                positions: &positions,
                velocities: &velocities,
                time: sim.time(),
            },
            DT,
            WIDTH as f32 / HEIGHT as f32,
        );
        let falcon = if sim.falcon_active() {
            Some(sim.falcon(&gpu)?.position())
        } else {
            None
        };
        scene = Scene::from_camera(
            &camera,
            WIDTH,
            HEIGHT,
            evening_of(args, sim.time()),
            renderer.exposure(),
            FLAP_AMPLITUDE,
            sim.time(),
        )
        .with_falcon(falcon)
        .with_reeds(murmur_render::sky::reed_visibility(&camera));
        let evening = scene.evening;
        renderer.set_scene(&gpu, scene);
        renderer.render(&gpu, &target.view, sim.len() as u32, DT);
        let rgba = renderer.read_target(&gpu, &target, WIDTH, HEIGHT);
        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        png::write_rgb(format!("{dir}/frame-{frame:04}.png"), WIDTH, HEIGHT, &rgb)?;
        if frame % 20 == 0 {
            println!(
                "# frame {frame:4} t={:6.1}s live {} evening {evening:.2}",
                sim.time(),
                birds.iter().filter(|b| b.live != 0).count(),
            );
        }
    }
    println!("# {} frames written to {dir}", frames);
    Ok(())
}

/// The agitation-wave experiment (PLAN.md phase 7): how fast a zigzag copies
/// through the flock.
///
/// Runs the same experiment on the CPU oracle and on the GPU, seeding a patch of
/// zigzags on one side of a settled flock and fitting the wave front's arrival
/// time against distance. The sibling terminal version reports 6.8–7.6 m/s at
/// 2,000 birds with `k = 2`; both lines are printed so the GPU's speed can be
/// read against that range rather than against itself.
fn run_agitation(args: &Args) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let gpu = open()?;
    let info = gpu.info();
    println!(
        "# murmuration-gpu --agitation birds={} seed={} k={}",
        args.birds, args.seed, args.zigzag_k
    );
    println!(
        "# adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    // The CPU oracle mirrors the terminal's run (same seeding, same fit).
    let mut world = World::new(args.birds, args.seed);
    world.zigzag_k = args.zigzag_k;
    let cpu = murmur_ref::escape::measure_agitation(&mut world, 4.0);
    println!("# cpu {}", agitation_line(&cpu));
    // The GPU runs the same experiment from its own settled state.
    let mut world = World::new(args.birds, args.seed);
    world.zigzag_k = args.zigzag_k;
    let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
    let gpu_wave = agitation::measure(&gpu, &mut sim, 4.0)?;
    println!("# gpu {}", agitation_line(&gpu_wave));

    let in_range = (3.0..=25.0).contains(&gpu_wave.speed);
    let tracks = (gpu_wave.speed - cpu.speed).abs() / cpu.speed.max(1e-6);
    println!(
        "# gpu speed {:.2} m/s is {} the reference's 3-25 m/s range, {:.0}% from the CPU \
         oracle's {:.2} m/s",
        gpu_wave.speed,
        if in_range { "inside" } else { "OUTSIDE" },
        tracks * 100.0,
        cpu.speed,
    );
    println!("# wall {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

/// One `--agitation` result line, in the terminal version's format.
fn agitation_line(wave: &murmur_ref::escape::AgitationWave) -> String {
    format!(
        "wave speed = {:.2} m/s, R2 = {:.3}, samples = {}, reached = {:.0}%, expansion = {:.0}%",
        wave.speed,
        wave.r2,
        wave.samples,
        wave.peak_alarmed * 100.0,
        wave.peak_expansion * 100.0,
    )
}

/// The camera's view of the flock: the live birds only. The evening arc parks
/// the slots it has not filled (and the birds that have landed) at the origin,
/// and the camera must not frame them.
fn flock_view(birds: &[Bird]) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut positions = Vec::with_capacity(birds.len());
    let mut velocities = Vec::with_capacity(birds.len());
    for b in birds.iter().filter(|b| b.live != 0) {
        positions.push(Vec3::from_array(b.pos));
        velocities.push(Vec3::from_array(b.vel));
    }
    (positions, velocities)
}

/// How far into the evening the sky is, from the arc's own clock: `0` at the
/// gold of the arrival, `1` at night. Without an arc the sky holds the model's
/// mid-evening value.
fn evening_of(args: &Args, time: f32) -> f32 {
    if args.no_arc {
        return 0.5;
    }
    let mut arc = murmur_ref::arc::Arc::new(args.arc_minutes, args.seed);
    arc.time = time % arc.duration;
    arc.evening()
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    };
    if args.parity {
        if let Err(error) = run_parity(&args) {
            eprintln!("murmuration-gpu: --parity failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.bench {
        if let Err(error) = run_bench(&args) {
            eprintln!("murmuration-gpu: --bench failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(seconds) = args.analyze {
        if let Err(error) = run_analyze(&args, seconds) {
            eprintln!("murmuration-gpu: --analyze failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(seconds) = args.coherence {
        if let Err(error) = run_coherence(&args, seconds) {
            eprintln!("murmuration-gpu: --coherence failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(path) = &args.still {
        let seconds = args.seconds as f32;
        if let Err(error) = run_still(&args, path, seconds) {
            eprintln!("murmuration-gpu: --still failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(dir) = &args.capture {
        let seconds = args.seconds as f32;
        if let Err(error) = run_capture(&args, dir, seconds) {
            eprintln!("murmuration-gpu: --capture failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.lod_sweep {
        if let Err(error) = run_lod_sweep(&args) {
            eprintln!("murmuration-gpu: --lod-sweep failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(dir) = &args.record {
        if let Err(error) = run_record(&args, dir) {
            eprintln!("murmuration-gpu: --record failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.agitation {
        if let Err(error) = run_agitation(&args) {
            eprintln!("murmuration-gpu: --agitation failed: {error}");
            std::process::exit(1);
        }
        return;
    }

    let event_loop = EventLoop::new().expect("opening the window event loop");
    let mut app = App {
        args,
        renderer: None,
    };
    event_loop
        .run_app(&mut app)
        .expect("running the window event loop");
}

/// The running app: the parsed arguments (the flock's size and seed) and the
/// renderer, once winit has delivered the first `resumed`.
struct App {
    args: Args,
    renderer: Option<Renderer>,
}
/// Window, surface, GPU, flock and renderer, wired together.
struct Renderer {
    window: Arc<Window>,
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    sim: Sim,
    frame: FrameRenderer,
    camera: Camera,
    birds: usize,
    /// The run's seed and preferred height, for a window-spawned falcon.
    seed: u64,
    z0: f32,
    /// The parsed arguments, for the evening's clock.
    args: Args,
    /// Wall clock of the previous frame, so the flock steps in real time.
    last: Instant,
}

/// The largest catch-up the window will simulate in one frame, so a stall does
/// not turn into one enormous submission.
const MAX_FRAME_STEPS: u32 = 40;

/// How far the `--capture` lens zooms in over the recording: the framing field
/// of view is scaled by this at the last frame, so the flock grows and every
/// bird crosses every LOD threshold in view.
const CAPTURE_ZOOM: f32 = 0.35;

impl Renderer {
    /// Creates the window, the surface and the device, builds the flock and
    /// the renderer, and prints which adapter the device landed on.
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("murmuration-gpu")
                    .with_inner_size(LogicalSize::new(1280.0, 720.0)),
            )?,
        );
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
                Box::new(event_loop.owned_display_handle()),
            ));
        let surface = instance.create_surface(window.clone())?;
        let gpu = device::for_surface(&instance, &surface, wgpu::PowerPreference::HighPerformance)?;

        let info = gpu.info();
        println!(
            "murmuration-gpu: adapter \"{}\" on {:?} ({:?}), driver \"{}\" {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width, size.height)
            .ok_or("the adapter cannot present to this window")?;
        config.usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&gpu.device, &config);
        println!(
            "murmuration-gpu: surface {}x{} {:?}, present mode {:?}",
            config.width, config.height, config.format, config.present_mode
        );

        let mut world = World::new_density_scaled(args.birds, args.seed);
        world.turning = args.turning;
        let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
        if !args.no_arc {
            sim.enable_arc(&gpu, args.arc_minutes, args.seed);
        }
        if args.falcon {
            sim.spawn_falcon(&gpu, args.seed, world.z0);
        }

        let mut frame = FrameRenderer::new(&gpu, config.width, config.height, config.format)?;
        frame.set_birds(&gpu, sim.bird_buffer(), sim.len() as u32);

        Ok(Self {
            window,
            gpu,
            surface,
            config,
            sim,
            frame,
            camera: Camera::new(),
            birds: args.birds,
            last: Instant::now(),
            seed: args.seed,
            z0: world.z0,
            args: args.clone(),
        })
    }

    /// Recreates the swap chain and the render targets after a resize.
    fn resize(&mut self) {
        let size = self.window.inner_size();
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.gpu.device, &self.config);
            self.frame.resize(&self.gpu, size.width, size.height);
            self.window.request_redraw();
        }
    }

    /// Clears the frame to the dusk colour and presents it.
    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                // Nothing to present; try again on the next redraw request.
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                drop(texture);
                self.surface.configure(&self.gpu.device, &self.config);
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.gpu.device, &self.config);
                self.window.request_redraw();
                return Ok(());
            }
            // No error scope is registered, so validation errors panic before
            // this variant can be produced.
            wgpu::CurrentSurfaceTexture::Validation => unreachable!(
                "validation errors panic without an error scope, so this cannot be returned"
            ),
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .gpu
                    .instance
                    .create_surface(self.window.clone())
                    .map_err(|_| "recreating the lost surface failed")?;
                self.surface.configure(&self.gpu.device, &self.config);
                self.window.request_redraw();
                return Ok(());
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Step the flock in real time, then let the camera frame it. The GPU
        // readback drives the camera for now; phase 9 removes the round-trip.
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().clamp(1e-3, 0.25);
        self.last = now;
        let steps = ((dt / self.sim.params.dt) as u32).clamp(1, MAX_FRAME_STEPS);
        self.sim.advance(&self.gpu, steps);

        let birds = self.sim.read_birds(&self.gpu)?;
        let (positions, velocities) = flock_view(&birds);
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        self.camera.update(
            &FlockView {
                positions: &positions,
                velocities: &velocities,
                time: self.sim.time(),
            },
            dt,
            aspect,
        );

        // The bird buffer ping-pongs each step, so re-point the coverage pass.
        self.frame
            .set_birds(&self.gpu, self.sim.bird_buffer(), self.sim.len() as u32);
        let falcon = if self.sim.falcon_active() {
            Some(self.sim.falcon(&self.gpu)?.position())
        } else {
            None
        };
        self.frame.set_scene(
            &self.gpu,
            Scene::from_camera(
                &self.camera,
                self.config.width,
                self.config.height,
                evening_of(&self.args, self.sim.time()),
                self.frame.exposure(),
                FLAP_AMPLITUDE,
                self.sim.time(),
            )
            .with_falcon(falcon)
            .with_reeds(murmur_render::sky::reed_visibility(&self.camera)),
        );
        self.frame.render(&self.gpu, &view, self.birds as u32, dt);

        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
        self.window.request_redraw();
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        match Renderer::new(event_loop, &self.args) {
            Ok(renderer) => {
                renderer.window.request_redraw();
                self.renderer = Some(renderer);
            }
            Err(err) => {
                eprintln!("murmuration-gpu: initialising the GPU failed: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Character(ref key) if key.as_str() == "q" => event_loop.exit(),
                    // `f` calls the falcon (the terminal version's key).
                    Key::Character(ref key) if key.as_str() == "f" => {
                        renderer
                            .sim
                            .spawn_falcon(&renderer.gpu, renderer.seed, renderer.z0);
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Moving the mouse makes you the falcon, spawning one on the
                // first movement (the terminal version's behaviour).
                if !renderer.sim.falcon_active() {
                    renderer
                        .sim
                        .spawn_falcon(&renderer.gpu, renderer.seed, renderer.z0);
                }
                let flock = World::restore(&renderer.sim.snapshot(&renderer.gpu));
                let plane = if flock.position.is_empty() {
                    glam::Vec3::Z * renderer.z0
                } else {
                    flock.position.iter().copied().sum::<glam::Vec3>() / flock.position.len() as f32
                };
                let target = renderer.camera.cursor_to_plane(
                    position.x as f32,
                    position.y as f32,
                    renderer.config.width as f32,
                    renderer.config.height as f32,
                    plane,
                );
                renderer.sim.steer_falcon(&renderer.gpu, target);
            }
            WindowEvent::Resized(_) => renderer.resize(),
            WindowEvent::Occluded(false) => renderer.window.request_redraw(),
            WindowEvent::RedrawRequested => {
                if let Err(err) = renderer.render() {
                    eprintln!("murmuration-gpu: rendering failed: {err}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}
