//! The GPU simulation: the compute pipelines, the ping-pong bird buffers and
//! the spatial grid.
//!
//! Each step mirrors the reference's step (see `shaders/sim.wgsl` and
//! `shaders/grid.wgsl`): `tick` advances the shared clock, the grid is rebuilt
//! from the current positions, `decide` reads the whole flock from `src` and
//! writes `dst` (a faithful simultaneous snapshot, like the reference reading
//! `self` once), and `integrate` advances `dst` in place.
//!
//! The bird state is double-buffered and the two directions have their own bind
//! groups, so a step needs no queue write: many steps can be encoded into one
//! command buffer. [`Sim::advance`] is that loop; [`Sim::advance_timeline`]
//! stamps each pass with GPU timestamps for `--bench`.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use murmur_ref::state::Snapshot;
use wgpu::{
    BindGroup, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder, CommandEncoderDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor, QuerySet,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::bird::Bird;
use crate::device::Gpu;
use crate::grid::{
    BINDING_BLOCK_SUMS, BINDING_BUCKET, BINDING_COUNTS, BINDING_OFFSETS, BINDING_SORTED,
    COMMON_WGSL, Grid,
};
use crate::params::{Params, turning_from_code};
use crate::pass::{Timeline, dispatch};

/// Threads per workgroup, matching `@workgroup_size` in `shaders/sim.wgsl`.
const WORKGROUP: u32 = 256;

/// The per-bird step, mirrored from `shaders/sim.wgsl`.
const SIM_WGSL: &str = include_str!("../shaders/sim.wgsl");

/// The shared clock buffer, matching `struct Clock` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Clock {
    /// Simulated time \[s\].
    pub time: f32,
    /// Integration steps taken.
    pub step: u32,
    pub pad0: u32,
    pub pad1: u32,
}

/// A running GPU flock: pipelines, buffers, the grid and the step index.
pub struct Sim {
    /// The parameters block sent to the shader.
    pub params: Params,
    /// The spatial grid, rebuilt every step.
    pub grid: Grid,
    params_buffer: Buffer,
    birds: [Buffer; 2],
    clock: Buffer,
    readback: Buffer,
    bind_groups: [BindGroup; 2],
    decide: ComputePipeline,
    decide_bruteforce: ComputePipeline,
    integrate: ComputePipeline,
    tick: ComputePipeline,
    n: usize,
    /// Index into `birds` holding the current state (the next `src`).
    cur: usize,
    time: f32,
    step: u32,
    seed: u64,
}

impl Sim {
    /// Builds a GPU flock for `params`, with an empty (zeroed) bird buffer.
    pub fn new(gpu: &Gpu, params: Params) -> Self {
        let device = &gpu.device;
        let source = format!("{COMMON_WGSL}\n{SIM_WGSL}");
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sim"),
            source: ShaderSource::Wgsl(source.into()),
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sim: bind group layout"),
            entries: &[
                entry(0, BufferBindingType::Uniform),
                entry(1, BufferBindingType::Storage { read_only: true }),
                entry(2, BufferBindingType::Storage { read_only: false }),
                entry(3, BufferBindingType::Storage { read_only: false }),
                entry(
                    BINDING_BUCKET,
                    BufferBindingType::Storage { read_only: false },
                ),
                entry(
                    BINDING_COUNTS,
                    BufferBindingType::Storage { read_only: false },
                ),
                entry(
                    BINDING_OFFSETS,
                    BufferBindingType::Storage { read_only: false },
                ),
                entry(
                    BINDING_SORTED,
                    BufferBindingType::Storage { read_only: false },
                ),
                entry(
                    BINDING_BLOCK_SUMS,
                    BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sim: pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let decide = pipeline("decide_kernel");
        let decide_bruteforce = pipeline("decide_bruteforce_kernel");
        let integrate = pipeline("integrate_kernel");
        let tick = pipeline("tick_kernel");

        let n = params.n as usize;
        let bird_bytes = (n.max(1) * size_of::<Bird>()) as u64;
        let storage = || {
            device.create_buffer(&BufferDescriptor {
                label: Some("sim: birds"),
                size: bird_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let birds = [storage(), storage()];
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sim: params"),
            size: size_of::<Params>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let clock = device.create_buffer(&BufferDescriptor {
            label: Some("sim: clock"),
            size: size_of::<Clock>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("sim: readback"),
            size: bird_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid = Grid::new(gpu, &layout, n);

        let bind = |src: usize, dst: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sim: bind group"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: birds[src].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: birds[dst].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: clock.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: BINDING_BUCKET,
                        resource: grid.bucket.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: BINDING_COUNTS,
                        resource: grid.counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: BINDING_OFFSETS,
                        resource: grid.offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: BINDING_SORTED,
                        resource: grid.sorted.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: BINDING_BLOCK_SUMS,
                        resource: grid.block_sums.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_groups = [bind(0, 1), bind(1, 0)];

        let sim = Self {
            params,
            grid,
            params_buffer,
            birds,
            clock,
            readback,
            bind_groups,
            decide,
            decide_bruteforce,
            integrate,
            tick,
            n,
            cur: 0,
            time: params.time,
            step: 0,
            seed: 0,
        };
        sim.write_params(gpu);
        sim.write_clock(gpu, params.time, 0);
        sim
    }

    /// Builds a GPU flock from an imported snapshot, so it starts from exactly
    /// the CPU state.
    pub fn from_snapshot(gpu: &Gpu, snapshot: &Snapshot) -> Self {
        let mut sim = Self::new(gpu, Params::from_snapshot(snapshot));
        sim.seed = snapshot.seed;
        let birds: Vec<Bird> = snapshot
            .birds
            .iter()
            .enumerate()
            .map(|(i, b)| Bird::from_ref(b, i as u32, snapshot.seed))
            .collect();
        gpu.queue
            .write_buffer(&sim.birds[0], 0, bytemuck::cast_slice(&birds));
        sim.cur = 0;
        sim
    }

    /// Overwrites the current bird buffer with `birds` (benchmarks and tests).
    pub fn write_birds(&self, gpu: &Gpu, birds: &[Bird]) {
        gpu.queue
            .write_buffer(&self.birds[self.cur], 0, bytemuck::cast_slice(birds));
    }

    /// Number of birds.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the flock is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Simulated time \[s\] after the steps run so far.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Overrides the random-force weight on the GPU (single-step parity check).
    pub fn set_w_xi(&mut self, gpu: &Gpu, w_xi: f32) {
        self.params = self.params.with_w_xi(w_xi);
        self.write_params(gpu);
    }

    /// Encodes and submits `steps` integration steps in one command buffer.
    pub fn advance(&mut self, gpu: &Gpu, steps: u32) {
        self.run(gpu, steps, false);
    }

    /// The same loop with the brute-force neighbour search, for the phase 3
    /// acceptance test (the grid is not rebuilt).
    pub fn advance_bruteforce(&mut self, gpu: &Gpu, steps: u32) {
        self.run(gpu, steps, true);
    }

    /// Runs exactly one step with every pass stamped into `query_set`, for
    /// `--bench`. The slots are those of [`crate::pass::STEP_PASSES`].
    pub fn advance_timeline(&mut self, gpu: &Gpu, query_set: &QuerySet) {
        let mut timeline = Some(Timeline::new(query_set));
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sim: timed step"),
            });
        self.record_step(&mut encoder, false, &mut timeline);
        gpu.queue.submit(Some(encoder.finish()));
        self.cur ^= 1;
        self.time += self.params.dt;
        self.step += 1;
    }

    /// Rebuilds the grid from the current positions on its own (tests).
    pub fn rebuild_grid(&self, gpu: &Gpu) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sim: grid"),
            });
        self.grid
            .record(&mut encoder, &self.bind_groups[self.cur], self.n, &mut None);
        gpu.queue.submit(Some(encoder.finish()));
    }

    fn run(&mut self, gpu: &Gpu, steps: u32, bruteforce: bool) {
        if steps == 0 {
            return;
        }
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sim: step"),
            });
        let mut timeline = None;
        for _ in 0..steps {
            self.record_step(&mut encoder, bruteforce, &mut timeline);
            self.cur ^= 1;
        }
        gpu.queue.submit(Some(encoder.finish()));
        self.time += steps as f32 * self.params.dt;
        self.step += steps;
    }

    /// Records one step for the current direction.
    fn record_step(
        &self,
        encoder: &mut CommandEncoder,
        bruteforce: bool,
        timeline: &mut Option<Timeline<'_>>,
    ) {
        let bind_group = &self.bind_groups[self.cur];
        let workgroups = self.n.div_ceil(WORKGROUP as usize) as u32;
        dispatch(encoder, &self.tick, bind_group, 1, timeline.as_mut());
        if !bruteforce {
            self.grid.record(encoder, bind_group, self.n, timeline);
        }
        let decide = if bruteforce {
            &self.decide_bruteforce
        } else {
            &self.decide
        };
        dispatch(encoder, decide, bind_group, workgroups, timeline.as_mut());
        dispatch(
            encoder,
            &self.integrate,
            bind_group,
            workgroups,
            timeline.as_mut(),
        );
    }

    /// Copies the current bird state back to the host.
    pub fn read_birds(&mut self, gpu: &Gpu) -> Vec<Bird> {
        let bytes = (self.n * size_of::<Bird>()) as u64;
        if bytes == 0 {
            return Vec::new();
        }
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sim: readback"),
            });
        encoder.copy_buffer_to_buffer(&self.birds[self.cur], 0, &self.readback, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the readback to finish");
        rx.recv()
            .expect("the poll ran the mapping callback before returning")
            .expect("mapping the readback buffer for reading");

        let range = self
            .readback
            .get_mapped_range(..)
            .expect("reading the mapped range");
        let birds: Vec<Bird> = bytemuck::cast_slice(&range).to_vec();
        drop(range);
        self.readback.unmap();
        birds
    }

    /// Reads the current state back as a [`Snapshot`] the CPU oracle can analyse
    /// (its random streams are placeholders; analysis never reads them).
    pub fn snapshot(&mut self, gpu: &Gpu) -> Snapshot {
        let birds = self.read_birds(gpu);
        Snapshot {
            time: self.time,
            z0: self.params.z0,
            seed: self.seed,
            target_birds: self.n,
            zigzag_k: self.params.zigzag_k,
            turning: turning_from_code(self.params.turning),
            w_xi: self.params.w_xi,
            birds: birds.iter().map(|b| b.to_ref(self.seed)).collect(),
        }
    }

    fn write_params(&self, gpu: &Gpu) {
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    fn write_clock(&self, gpu: &Gpu, time: f32, step: u32) {
        let clock = Clock {
            time,
            step,
            pad0: 0,
            pad1: 0,
        };
        gpu.queue
            .write_buffer(&self.clock, 0, bytemuck::bytes_of(&clock));
    }
}

/// A storage or uniform entry visible to the compute stage.
fn entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
