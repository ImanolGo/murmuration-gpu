//! Instrumented CPU frame profiler for the real window (not a GPU benchmark).
//! Samples include surface acquisition, submission, and presentation. Profiling
//! never waits on GPU readbacks during playback; optional GPU data is read at exit.
use murmur_gpu::{
    device::{Gpu, WAIT_BUDGET},
    sim::Sim,
};
use std::time::Instant;

const CAPACITY: usize = 16384;
const STRIDE: u64 = 256;

struct GpuProbe {
    queries: wgpu::QuerySet,
    history: wgpu::Buffer,
}

pub const STAGES: [&str; 7] = [
    "surface",
    "sim_submit",
    "tracking",
    "camera",
    "bindings",
    "render_submit",
    "present",
];

pub struct Profile {
    started: Instant,
    seconds: f32,
    samples: Vec<([f64; 7], f64, u32)>,
    last_frame: Instant,
    gpu: Option<GpuProbe>,
}

impl Profile {
    pub fn new(gpu: &Gpu, seconds: f32) -> Self {
        Self {
            started: Instant::now(),
            last_frame: Instant::now(),
            seconds,
            samples: Vec::new(),
            gpu: gpu
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
                .then(|| GpuProbe {
                    queries: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("window simulation timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    }),
                    history: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("window simulation timestamp history"),
                        size: CAPACITY as u64 * STRIDE,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                }),
        }
    }

    pub fn advance(&self, gpu: &Gpu, sim: &mut Sim, steps: u32) {
        if let Some(probe) = &self.gpu
            && self.samples.len() < CAPACITY
        {
            sim.advance_profiled(
                gpu,
                steps,
                &probe.queries,
                &probe.history,
                self.samples.len() as u64 * STRIDE,
            );
            return;
        }
        sim.advance(gpu, steps);
    }

    pub fn record(&mut self, stages: [f64; 7], steps: u32) -> bool {
        let now = Instant::now();
        let interval = (now - self.last_frame).as_secs_f64() * 1000.0;
        self.last_frame = now;
        self.samples.push((stages, interval, steps));
        now.duration_since(self.started).as_secs_f32() >= self.seconds
    }

    pub fn report(&self, gpu: &Gpu, simulated: f32) {
        let wall = self.started.elapsed().as_secs_f64();
        println!(
            "# window CPU profiler: {} presented frames / {wall:.3}s = {:.2} fps; simulated {simulated:.3}s",
            self.samples.len(),
            self.samples.len() as f64 / wall
        );
        println!("# CPU wall spans include driver waits, not GPU execution times");
        for (i, name) in STAGES.iter().enumerate() {
            let data: Vec<f64> = self.samples.iter().map(|s| s.0[i]).collect();
            print_stats(name, data);
        }
        print_stats("frame_interval", self.samples.iter().map(|s| s.1).collect());
        let steps: u32 = self.samples.iter().map(|s| s.2).sum();
        println!(
            "# fixed steps {steps}; mean {:.2} per frame",
            steps as f64 / self.samples.len().max(1) as f64
        );
        println!(
            "# first frames: interval_ms, surface_ms, sim_submit_ms, tracking_ms, camera_ms, bindings_ms, render_submit_ms, present_ms, steps"
        );
        for (stages, interval, steps) in self.samples.iter().take(20) {
            println!(
                "# sample {interval:.3}, {}, {steps}",
                stages
                    .iter()
                    .map(|s| format!("{s:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if let Some(probe) = &self.gpu {
            let count = self.samples.len().min(CAPACITY);
            let bytes = count as u64 * STRIDE;
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("window timestamp readback at exit"),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&probe.history, 0, &staging, 0, bytes);
            gpu.queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r);
            });
            gpu.wait_bounded(WAIT_BUDGET).expect("profile readback");
            rx.recv().unwrap().unwrap();
            let range = staging.get_mapped_range(..).unwrap();
            let ticks: &[u64] = bytemuck::cast_slice(&range);
            let period = f64::from(gpu.queue.get_timestamp_period()) / 1e6;
            let batches: Vec<f64> = (0..count)
                .map(|i| ticks[i * 32 + 1].wrapping_sub(ticks[i * 32]) as f64 * period)
                .collect();
            let per_step: Vec<f64> = batches
                .iter()
                .zip(&self.samples)
                .filter(|(_, s)| s.2 > 0)
                .map(|(t, s)| t / f64::from(s.2))
                .collect();
            println!("# GPU timestamp simulation batch / per-step ms (read only after playback):");
            print_stats("GPU_sim_batch", batches.clone());
            print_stats("GPU_sim_step", per_step);
            for (i, t) in batches.iter().take(20).enumerate() {
                println!(
                    "# GPU sample {i} steps {} batch {t:.3} ms",
                    self.samples[i].2
                );
            }
            drop(range);
            staging.unmap();
        }
    }
}

fn print_stats(name: &str, mut data: Vec<f64>) {
    if data.is_empty() {
        return;
    }
    data.sort_by(f64::total_cmp);
    let at = |q: f64| data[((data.len() - 1) as f64 * q) as usize];
    println!(
        "# profile {name:14} mean {:8.3} ms p50 {:8.3} p95 {:8.3} p99 {:8.3} max {:8.3}",
        data.iter().sum::<f64>() / data.len() as f64,
        at(0.5),
        at(0.95),
        at(0.99),
        data.last().unwrap()
    );
}
