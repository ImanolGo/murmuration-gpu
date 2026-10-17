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

use crate::BIRD_WGSL;
use crate::arc::ArcState;
use crate::bird::Bird;
use crate::device::{Gpu, Stalled, WAIT_BUDGET};
use crate::falcon::{Falcon, Mode};
use crate::grid::{
    BINDING_BLOCK_SUMS, BINDING_BUCKET, BINDING_COUNTS, BINDING_OFFSETS, BINDING_SORTED,
    COMMON_WGSL, Grid,
};
use crate::params::{Params, turning_from_code};
use crate::pass::{Timeline, dispatch};

/// Threads per workgroup, matching `@workgroup_size` in `shaders/sim.wgsl`.
const WORKGROUP: u32 = 256;

/// Workgroups of `decide_wide_kernel`; `WIDE_GROUPS` in `shaders/sim.wgsl`.
const WIDE_GROUPS: u32 = 64;

/// The per-bird step, mirrored from `shaders/sim.wgsl`.
const SIM_WGSL: &str = include_str!("../shaders/sim.wgsl");

/// The falcon's reduction and update kernels (PLAN.md phase 7), appended to the
/// simulation module so they share `Params`, `Bird`, `Clock` and the falcon
/// binding.
const FALCON_WGSL: &str = include_str!("../shaders/falcon.wgsl");

/// The evening arc's kernels (PLAN.md phase 8), in the same module for the same
/// reason. They read the flock through their own read-write binding instead of
/// the sim's read-only one, because the arc adds and retires birds in place.
const ARC_WGSL: &str = include_str!("../shaders/arc.wgsl");

/// The shared clock buffer, matching `struct Clock` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Clock {
    /// Simulated time \[s\].
    pub time: f32,
    /// Integration steps taken.
    pub step: u32,
    /// Neighbour queries that fell back to the brute-force walk on the last
    /// step (`gather` in `sim.wgsl`); see [`Sim::wide_queries`].
    pub wide: u32,
    pub pad0: u32,
}

/// The flock as the shader holds it, with the clock: what [`Sim::raw_state`]
/// returns and [`Sim::from_raw`] consumes.
///
/// Unlike [`Snapshot`] this keeps each bird's own random stream, so a flock
/// rebuilt from it continues the same trajectory rather than restarting the
/// random force. A run repairs itself after a driver hang by keeping the last
/// `RawState` and rebuilding from it.
#[derive(Clone, Debug)]
pub struct RawState {
    /// The bird records, each with its own random stream.
    pub birds: Vec<Bird>,
    /// Simulated time \[s\].
    pub time: f32,
    /// Integration steps taken.
    pub step: u32,
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
    falcon: Buffer,
    arc: Buffer,
    arc_bind: [BindGroup; 2],
    readback: Buffer,
    bind_groups: [BindGroup; 2],
    decide: ComputePipeline,
    /// Wide-radius stragglers, one workgroup each (`decide_wide_kernel`).
    decide_wide: ComputePipeline,
    prepare_decisions: ComputePipeline,
    decide_bruteforce: ComputePipeline,
    integrate: ComputePipeline,
    tick: ComputePipeline,
    falcon_reduce: ComputePipeline,
    falcon_update: ComputePipeline,
    arc_update: ComputePipeline,
    arc_landed: ComputePipeline,
    n: usize,
    /// Index into `birds` holding the current state (the next `src`).
    cur: usize,
    time: f32,
    step: u32,
    seed: u64,
    /// Whether a falcon is hunting, so the step knows to run its passes without
    /// a readback.
    falcon_active: bool,
    /// Whether an evening arc is running.
    arc_active: bool,
}

impl Sim {
    /// Builds a GPU flock for `params`, with an empty (zeroed) bird buffer.
    pub fn new(gpu: &Gpu, params: Params) -> Self {
        let device = &gpu.device;
        let source = format!("{COMMON_WGSL}\n{BIRD_WGSL}\n{SIM_WGSL}\n{FALCON_WGSL}\n{ARC_WGSL}");
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
                entry(9, BufferBindingType::Storage { read_only: false }),
                entry(11, BufferBindingType::Storage { read_only: false }),
            ],
        });
        // The arc adds and retires birds, so it binds the flock read-write
        // (binding 10) plus the clock and its own state.
        let arc_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sim: arc bind group layout"),
            entries: &[
                entry(0, BufferBindingType::Uniform),
                entry(3, BufferBindingType::Storage { read_only: false }),
                entry(10, BufferBindingType::Storage { read_only: false }),
                entry(11, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let arc_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sim: arc pipeline layout"),
            bind_group_layouts: &[Some(&arc_bgl)],
            immediate_size: 0,
        });
        let arc_pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&arc_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let arc_update = arc_pipeline("arc_kernel");
        let arc_landed = arc_pipeline("landed_kernel");
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
        let decide_wide = pipeline("decide_wide_kernel");
        let prepare_decisions = pipeline("prepare_decisions_kernel");
        let decide_bruteforce = pipeline("decide_bruteforce_kernel");
        let integrate = pipeline("integrate_kernel");
        let tick = pipeline("tick_kernel");
        let falcon_reduce = pipeline("falcon_reduce_kernel");
        let falcon_update = pipeline("falcon_kernel");

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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("sim: readback"),
            size: bird_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let falcon = device.create_buffer(&BufferDescriptor {
            label: Some("sim: falcon"),
            size: size_of::<Falcon>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&falcon, 0, bytemuck::bytes_of(&Falcon::absent()));
        let arc = device.create_buffer(&BufferDescriptor {
            label: Some("sim: arc"),
            size: size_of::<ArcState>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&arc, 0, bytemuck::bytes_of(&ArcState::absent(params.z0)));

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
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: falcon.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: arc.as_entire_binding(),
                    },
                ],
            })
        };
        // The arc's bind groups name the flock buffer it writes, one per
        // ping-pong direction.
        let bind_arc = |cur: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sim: arc bind group"),
                layout: &arc_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: clock.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: birds[cur].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: arc.as_entire_binding(),
                    },
                ],
            })
        };
        let arc_bind = [bind_arc(0), bind_arc(1)];
        let bind_groups = [bind(0, 1), bind(1, 0)];

        let sim = Self {
            params,
            grid,
            params_buffer,
            birds,
            clock,
            falcon,
            arc,
            arc_bind,
            readback,
            bind_groups,
            decide,
            decide_wide,
            prepare_decisions,
            decide_bruteforce,
            integrate,
            tick,
            falcon_reduce,
            falcon_update,
            arc_update,
            arc_landed,
            n,
            cur: 0,
            time: params.time,
            step: 0,
            seed: 0,
            falcon_active: false,
            arc_active: false,
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
        let birds = Self::birds_from_snapshot(snapshot);
        gpu.queue
            .write_buffer(&sim.birds[0], 0, bytemuck::cast_slice(&birds));
        sim.cur = 0;
        sim
    }

    /// The bird records a [`Sim`] holds after [`Sim::from_snapshot`]: the CPU
    /// birds with each random stream derived from the snapshot's seed.
    ///
    /// Exposed so a long run can keep the state it would have at step zero and
    /// resume from there if the very first readback is the one that stalls.
    pub fn birds_from_snapshot(snapshot: &Snapshot) -> Vec<Bird> {
        snapshot
            .birds
            .iter()
            .enumerate()
            .map(|(i, b)| Bird::from_ref(b, i as u32, snapshot.seed))
            .collect()
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

    /// Exact integration step index, without floating-point clock drift.
    pub fn steps_taken(&self) -> u32 {
        self.step
    }

    /// The storage buffer holding the flock's current state, so the renderer
    /// can draw the birds without a host round-trip.
    pub fn bird_buffer(&self) -> &Buffer {
        &self.birds[self.cur]
    }

    /// Overrides the random-force weight on the GPU (single-step parity check).
    pub fn set_w_xi(&mut self, gpu: &Gpu, w_xi: f32) {
        self.params = self.params.with_w_xi(w_xi);
        self.write_params(gpu);
    }

    /// Spawns an autonomous falcon (`world.rs::spawn_falcon`). Idempotent: a run
    /// already hunting keeps its falcon.
    pub fn spawn_falcon(&mut self, gpu: &Gpu, seed: u64, z0: f32) {
        if self.falcon_active {
            return;
        }
        self.write_falcon(gpu, &Falcon::spawn(seed, z0));
    }

    /// Starts an evening of `minutes` minutes (`world.rs::enable_arc`): the
    /// flock is cleared and the arc streams arriving birds into the free slots.
    pub fn enable_arc(&mut self, gpu: &Gpu, minutes: f32, seed: u64) {
        let z0 = self.params.z0;
        // The reference clears the flock, so every slot starts empty.
        let empty = vec![Bird::zeroed(); self.n];
        gpu.queue
            .write_buffer(&self.birds[self.cur], 0, bytemuck::cast_slice(&empty));
        let arc = ArcState::new(minutes, seed, z0);
        self.arc_active = arc.is_present();
        gpu.queue
            .write_buffer(&self.arc, 0, bytemuck::bytes_of(&arc));
    }

    /// Whether an evening arc is running.
    pub fn arc_active(&self) -> bool {
        self.arc_active
    }

    /// The evening's current state (a small blocking readback).
    pub fn arc_state(&self, gpu: &Gpu) -> Result<ArcState, Stalled> {
        read_small::<ArcState>(gpu, &self.arc, "arc")
    }

    /// Writes a falcon record over the current one (window steering, tests). Its
    /// `present` field decides whether the falcon passes run.
    pub fn write_falcon(&mut self, gpu: &Gpu, falcon: &Falcon) {
        self.falcon_active = falcon.is_present();
        gpu.queue
            .write_buffer(&self.falcon, 0, bytemuck::bytes_of(falcon));
    }

    /// Whether a falcon is hunting.
    pub fn falcon_active(&self) -> bool {
        self.falcon_active
    }

    /// Reads the falcon's current state (a small blocking readback).
    pub fn falcon(&self, gpu: &Gpu) -> Result<Falcon, Stalled> {
        read_small::<Falcon>(gpu, &self.falcon, "falcon")
    }

    /// Points a mouse-steered falcon at `target` without any GPU readback or stall.
    /// Writes only `mode` (offset 28) and `aim` (offset 32) into the GPU falcon buffer.
    pub fn steer_falcon(&mut self, gpu: &Gpu, target: glam::Vec3) {
        if !self.falcon_active {
            return;
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct SteerWrite {
            mode: u32,
            aim: [f32; 3],
        }
        let payload = SteerWrite {
            mode: Mode::Mouse.code(),
            aim: target.to_array(),
        };
        gpu.queue
            .write_buffer(&self.falcon, 28, bytemuck::bytes_of(&payload));
    }

    /// The GPU buffer holding the falcon state, for non-blocking tracking.
    pub fn falcon_buffer(&self) -> &Buffer {
        &self.falcon
    }

    /// Encodes and submits `steps` integration steps in one command buffer.
    pub fn advance(&mut self, gpu: &Gpu, steps: u32) {
        self.run(gpu, steps, false, None);
    }

    /// Whole-batch GPU timestamp span for the window profiler. No readback or
    /// wait is performed here; results are resolved into a caller-owned history.
    pub fn advance_profiled(
        &mut self,
        gpu: &Gpu,
        steps: u32,
        queries: &QuerySet,
        history: &Buffer,
        offset: u64,
    ) {
        self.run(gpu, steps, false, Some((queries, history, offset)));
    }

    /// Advances the flock by `seconds` of simulated time, submitting in short
    /// batches so a single command buffer never outlives the driver's GPU
    /// watchdog at large bird counts (the same reason [`crate::analyze`]
    /// batches its steps), and **draining the queue after every batch**.
    ///
    /// The drain is not a nicety. Submitting the whole interval and waiting once
    /// leaves all of it outstanding: at a million birds that is minutes of queued
    /// work, and the single wait then times out against [`WAIT_BUDGET`] even
    /// though the GPU is working normally — which is exactly how the first
    /// million-bird still "stalled". Waiting per batch keeps every wait as short
    /// as the ones a resilient run makes, and reports a real hang as [`Stalled`]
    /// instead of freezing.
    pub fn advance_seconds(&mut self, gpu: &Gpu, seconds: f32) -> Result<(), Stalled> {
        let steps = (seconds / self.params.dt).max(0.0) as u32;
        self.advance_steps_bounded(gpu, steps)
    }

    /// Like `advance_seconds`, but preserves an exact integer recording timeline.
    pub fn advance_steps_bounded(&mut self, gpu: &Gpu, steps: u32) -> Result<(), Stalled> {
        let mut done = 0;
        while done < steps {
            let this = 10.min(steps - done);
            self.advance(gpu, this);
            gpu.wait_bounded(WAIT_BUDGET)?;
            done += this;
        }
        Ok(())
    }

    /// The same loop with the brute-force neighbour search, for the phase 3
    /// acceptance test (the grid is not rebuilt).
    pub fn advance_bruteforce(&mut self, gpu: &Gpu, steps: u32) {
        self.run(gpu, steps, true, None);
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

    fn run(
        &mut self,
        gpu: &Gpu,
        steps: u32,
        bruteforce: bool,
        probe: Option<(&QuerySet, &Buffer, u64)>,
    ) {
        if steps == 0 && probe.is_none() {
            return;
        }
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sim: step"),
            });
        let mut timeline = None;
        if let Some((queries, _, _)) = probe {
            encoder.write_timestamp(queries, 0);
        }
        for _ in 0..steps {
            self.record_step(&mut encoder, bruteforce, &mut timeline);
            self.cur ^= 1;
        }
        if let Some((queries, history, offset)) = probe {
            encoder.write_timestamp(queries, 1);
            encoder.resolve_query_set(queries, 0..2, history, offset);
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
        // The evening advances first (`world.rs::step` runs the arc before the
        // falcon): one invocation sets the roost height and spawns an arriving
        // group into the free slots, and one pass per bird retires the birds that
        // have landed. Untimed, like the falcon's, so the `--bench` timeline
        // stays the step's own passes.
        if self.arc_active {
            dispatch(encoder, &self.arc_update, &self.arc_bind[self.cur], 1, None);
            dispatch(
                encoder,
                &self.arc_landed,
                &self.arc_bind[self.cur],
                workgroups,
                None,
            );
        }
        // The falcon hunts after the arc and before the alarm.
        if self.falcon_active {
            dispatch(encoder, &self.falcon_reduce, bind_group, 1, None);
            dispatch(encoder, &self.falcon_update, bind_group, 1, None);
        }
        if !bruteforce {
            self.grid.record(encoder, bind_group, self.n, timeline);
            encoder.clear_buffer(&self.grid.counts, 0, Some(8));
            dispatch(
                encoder,
                &self.prepare_decisions,
                bind_group,
                workgroups,
                timeline.as_mut(),
            );
        }
        let decide = if bruteforce {
            &self.decide_bruteforce
        } else {
            &self.decide
        };
        dispatch(encoder, decide, bind_group, workgroups, timeline.as_mut());
        if !bruteforce {
            dispatch(
                encoder,
                &self.decide_wide,
                bind_group,
                WIDE_GROUPS,
                timeline.as_mut(),
            );
        }
        dispatch(
            encoder,
            &self.integrate,
            bind_group,
            workgroups,
            timeline.as_mut(),
        );
    }

    /// Copies the current bird state back to the host.
    ///
    /// The wait is bounded ([`WAIT_BUDGET`]): if the driver has hung, the
    /// mapping callback never runs and an indefinite poll would block for ever.
    /// A stalled readback is reported so the caller can reopen the device.
    pub fn read_birds(&mut self, gpu: &Gpu) -> Result<Vec<Bird>, Stalled> {
        let bytes = (self.n * size_of::<Bird>()) as u64;
        if bytes == 0 {
            return Ok(Vec::new());
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
        gpu.wait_bounded(WAIT_BUDGET)?;
        rx.recv_timeout(WAIT_BUDGET)
            .map_err(|_| Stalled {
                waited: WAIT_BUDGET,
            })?
            .expect("mapping the readback buffer for reading");

        let range = self
            .readback
            .get_mapped_range(..)
            .expect("reading the mapped range");
        let birds: Vec<Bird> = bytemuck::cast_slice(&range).to_vec();
        drop(range);
        self.readback.unmap();
        Ok(birds)
    }

    /// How many neighbour queries took the brute-force fallback on the last
    /// step, read back from the clock.
    ///
    /// A query falls back once its radius covers more cells than the flock has
    /// birds (see `gather` in `sim.wgsl`). A bird that wide is a straggler whose
    /// adaptive radius has grown away from the flock, and since the grid walk
    /// costs a bucket lookup per cell in the radius box, `R³`, one such bird can
    /// cost as much as hundreds of thousands of ordinary ones — it is the run's
    /// main cost driver, and zero means every query used the grid.
    pub fn wide_queries(&mut self, gpu: &Gpu) -> Result<u32, Stalled> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sim: clock readback"),
            });
        encoder.copy_buffer_to_buffer(&self.clock, 0, &self.readback, 0, size_of::<Clock>() as u64);
        gpu.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        gpu.wait_bounded(WAIT_BUDGET)?;
        rx.recv_timeout(WAIT_BUDGET)
            .map_err(|_| Stalled {
                waited: WAIT_BUDGET,
            })?
            .expect("mapping the readback buffer for reading");

        let range = self
            .readback
            .get_mapped_range(..)
            .expect("reading the mapped range");
        let wide = bytemuck::from_bytes::<Clock>(&range[..size_of::<Clock>()]).wide;
        drop(range);
        self.readback.unmap();
        Ok(wide)
    }

    /// Reads the current state back as a [`Snapshot`] the CPU oracle can analyse
    /// (its random streams are placeholders; analysis never reads them).
    pub fn snapshot(&mut self, gpu: &Gpu) -> Snapshot {
        let raw = self
            .raw_state(gpu)
            .expect("reading the flock back for a snapshot");
        self.snapshot_from(&raw)
    }

    /// The flock as the shader holds it, plus the clock.
    ///
    /// Unlike [`Snapshot`], whose birds carry placeholder random streams for the
    /// CPU oracle, this keeps each bird's own stream, so a [`Sim`] rebuilt from
    /// it ([`Sim::from_raw`]) continues the *same* trajectory instead of
    /// restarting the random force. That is what makes a long run resumable
    /// after the driver hangs it: the resumed steps are the steps the
    /// uninterrupted run would have taken.
    pub fn raw_state(&mut self, gpu: &Gpu) -> Result<RawState, Stalled> {
        let birds = self.read_birds(gpu)?;
        Ok(RawState {
            birds,
            time: self.time,
            step: self.step,
        })
    }

    /// Rebuilds a flock on `gpu` from a [`RawState`], continuing its trajectory.
    ///
    /// `template` supplies every parameter that does not change over a run
    /// ([`Params::from_snapshot`]); its clock is replaced by `raw`'s.
    pub fn from_raw(gpu: &Gpu, template: Params, seed: u64, raw: &RawState) -> Self {
        let mut params = template;
        params.time = raw.time;
        let mut sim = Self::new(gpu, params);
        sim.seed = seed;
        sim.time = raw.time;
        sim.step = raw.step;
        sim.write_params(gpu);
        sim.write_clock(gpu, raw.time, raw.step);
        gpu.queue
            .write_buffer(&sim.birds[0], 0, bytemuck::cast_slice(&raw.birds));
        sim.cur = 0;
        sim
    }

    /// The CPU-facing [`Snapshot`] for a state already read back by
    /// [`Sim::raw_state`], with this flock's own clock and knobs. The random
    /// streams become placeholders, exactly as in [`Sim::snapshot`].
    pub fn snapshot_from(&self, raw: &RawState) -> Snapshot {
        Snapshot {
            time: raw.time,
            z0: self.params.z0,
            seed: self.seed,
            target_birds: self.n,
            zigzag_k: self.params.zigzag_k,
            turning: turning_from_code(self.params.turning),
            w_xi: self.params.w_xi,
            birds: raw.birds.iter().map(|b| b.to_ref(self.seed)).collect(),
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
            wide: 0,
            pad0: 0,
        };
        gpu.queue
            .write_buffer(&self.clock, 0, bytemuck::bytes_of(&clock));
    }
}

/// Reads a small buffer back to the host: the falcon and the evening arc. The
/// wait is bounded like every other readback ([`WAIT_BUDGET`]).
fn read_small<T: bytemuck::Pod>(gpu: &Gpu, buffer: &Buffer, label: &str) -> Result<T, Stalled> {
    let size = size_of::<T>() as u64;
    let staging = gpu.device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    gpu.wait_bounded(WAIT_BUDGET)?;
    rx.recv_timeout(WAIT_BUDGET)
        .map_err(|_| Stalled {
            waited: WAIT_BUDGET,
        })?
        .expect("mapping the small buffer for reading");
    let range = staging
        .get_mapped_range(..)
        .expect("reading the small buffer");
    let value = *bytemuck::from_bytes::<T>(&range[..size as usize]);
    drop(range);
    staging.unmap();
    Ok(value)
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
