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
//! Usage:
//!   prime-run cargo run -p murmuration-gpu --release                              # window
//!   cargo run -p murmuration-gpu --release -- --parity --birds 2000 --seconds 60

use std::error::Error;
use std::sync::Arc;

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::parity;
use murmur_render::sky;
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
    /// Number of birds.
    birds: usize,
    /// First seed; the sweep uses `seed .. seed + seeds`.
    seed: u64,
    /// Simulated seconds per seed.
    seconds: u32,
    /// Number of seeds.
    seeds: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            parity: false,
            birds: 2000,
            seed: 1,
            seconds: 60,
            seeds: 5,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--parity" => args.parity = true,
            "--birds" => args.birds = value(&mut it, "--birds")?.parse().map_err(err)?,
            "--seed" => args.seed = value(&mut it, "--seed")?.parse().map_err(err)?,
            "--seconds" => args.seconds = value(&mut it, "--seconds")?.parse().map_err(err)?,
            "--seeds" => args.seeds = value(&mut it, "--seeds")?.parse().map_err(err)?,
            "-h" | "--help" => {
                println!(
                    "murmuration-gpu [--parity] [--birds N] [--seed S] [--seconds T] [--seeds K]"
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

    let event_loop = EventLoop::new().expect("opening the window event loop");
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("running the window event loop");
}

/// The running app: nothing until winit delivers the first `resumed`.
#[derive(Default)]
struct App {
    renderer: Option<Renderer>,
}

/// Window, surface and GPU, wired together.
struct Renderer {
    window: Arc<Window>,
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl Renderer {
    /// Creates the window, the surface and the device, and prints which
    /// adapter the device landed on.
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
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

        Ok(Self {
            window,
            gpu,
            surface,
            config,
        })
    }

    /// Recreates the swap chain configuration after a resize.
    fn resize(&mut self) {
        let size = self.window.inner_size();
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.gpu.device, &self.config);
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
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("murmuration-gpu: frame"),
            });
        {
            let _dusk = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dusk clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(sky::dusk_clear(self.config.format)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.gpu.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
        // Keep the loop alive until there is a simulation to drive it.
        self.window.request_redraw();
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        match Renderer::new(event_loop) {
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
