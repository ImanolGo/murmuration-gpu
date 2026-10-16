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

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::sim::Sim;
use murmur_gpu::{FLAP_AMPLITUDE, Turning, World, analyze, bench, coherence, parity};
use murmur_render::camera::{Camera, FlockView};
use murmur_render::png;
use murmur_render::renderer::{Renderer as FrameRenderer, Scene};
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
#[derive(Debug)]
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
                     --coherence SECONDS | --still PATH] [--birds N] \
                     [--seed S] [--seconds T] [--seeds K] [--steps N] [--extended] \
                     [--turning base|inertial]"
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
    let gpu = open()?;
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
    let evening = world.evening();
    let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
    sim.advance_seconds(&gpu, seconds)?;

    let flock = World::restore(&sim.snapshot(&gpu));
    let mut camera = Camera::new();
    camera.update(
        &FlockView {
            positions: &flock.position,
            velocities: &flock.velocity,
            time: sim.time(),
        },
        DT,
        WIDTH as f32 / HEIGHT as f32,
    );

    // The tone-map pass writes to an `*Srgb` target, so the readback holds the
    // display-encoded bytes the PNG needs directly.
    let mut renderer =
        FrameRenderer::new(&gpu, WIDTH, HEIGHT, wgpu::TextureFormat::Rgba8UnormSrgb)?;
    renderer.set_birds(&gpu, sim.bird_buffer());
    let target = renderer.create_target(&gpu, WIDTH, HEIGHT);

    let mut scene = Scene::from_camera(
        &camera,
        WIDTH,
        HEIGHT,
        evening,
        renderer.exposure(),
        FLAP_AMPLITUDE,
        sim.time(),
    );
    // The first frame fixes the exposure from the flock's 80th-percentile
    // optical depth; the rest let it settle before the captured frame.
    for _ in 0..4 {
        scene.exposure = renderer.exposure();
        renderer.set_scene(&gpu, scene);
        renderer.render(&gpu, &target.view, args.birds as u32, DT);
    }

    let rgba = renderer.read_target(&gpu, &target, WIDTH, HEIGHT);
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    png::write_rgb(path, WIDTH, HEIGHT, &rgb)?;
    println!("# exposure {:.3}", renderer.exposure());
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
    /// Evening brightness for the sky, fixed for the run (the simulation's
    /// initial value; the evening arc arrives in a later phase).
    evening: f32,
    birds: usize,
    /// Wall clock of the previous frame, so the flock steps in real time.
    last: Instant,
}

/// The largest catch-up the window will simulate in one frame, so a stall does
/// not turn into one enormous submission.
const MAX_FRAME_STEPS: u32 = 40;

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
        let evening = world.evening();
        let sim = Sim::from_snapshot(&gpu, &world.snapshot());

        let mut frame = FrameRenderer::new(&gpu, config.width, config.height, config.format)?;
        frame.set_birds(&gpu, sim.bird_buffer());

        Ok(Self {
            window,
            gpu,
            surface,
            config,
            sim,
            frame,
            camera: Camera::new(),
            evening,
            birds: args.birds,
            last: Instant::now(),
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

        let flock = World::restore(&self.sim.snapshot(&self.gpu));
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        self.camera.update(
            &FlockView {
                positions: &flock.position,
                velocities: &flock.velocity,
                time: self.sim.time(),
            },
            dt,
            aspect,
        );

        // The bird buffer ping-pongs each step, so re-point the coverage pass.
        self.frame.set_birds(&self.gpu, self.sim.bird_buffer());
        self.frame.set_scene(
            &self.gpu,
            Scene::from_camera(
                &self.camera,
                self.config.width,
                self.config.height,
                self.evening,
                self.frame.exposure(),
                FLAP_AMPLITUDE,
                self.sim.time(),
            ),
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
                    _ => {}
                }
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
