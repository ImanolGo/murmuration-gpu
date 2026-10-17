//! Non-blocking camera telemetry. Sampling affects framing only, never forces.

use std::sync::mpsc::{self, Receiver, TryRecvError};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use murmur_gpu::{BIRD_WGSL, device::Gpu};

const SAMPLES: u32 = 4096;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TrackingPoint {
    pos: [f32; 3],
    live: u32,
    vel: [f32; 3],
    pad: u32,
}

/// A bounded, single-flight readback. A slow GPU cannot build a queue of stale
/// camera requests: keep using the last sample until the pending one completes.
pub struct FlockTracker {
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_cache: Vec<(usize, wgpu::BindGroup)>,
    pending: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pending_time: f32,
    sample_time: f32,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    falcon_pos: Option<Vec3>,
    falcon_vel: Vec3,
}

impl FlockTracker {
    pub fn new(gpu: &Gpu) -> Self {
        let source = format!("{BIRD_WGSL}\n{}", include_str!("../shaders/tracking.wgsl"));
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("camera sampling"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("camera sampling"),
                layout: None,
                module: &module,
                entry_point: Some("sample_flock"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bird_bytes = u64::from(SAMPLES) * size_of::<TrackingPoint>() as u64;
        let staging_bytes = bird_bytes + size_of::<murmur_gpu::falcon::Falcon>() as u64;
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera samples"),
            size: bird_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera staging"),
            size: staging_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            output,
            staging,
            bind_cache: Vec::new(),
            pending: None,
            pending_time: 0.0,
            sample_time: 0.0,
            positions: Vec::new(),
            velocities: Vec::new(),
            falcon_pos: None,
            falcon_vel: Vec3::ZERO,
        }
    }

    /// Poll without waiting. Returns true only when a new sample was received.
    pub fn refresh(&mut self, gpu: &Gpu) -> Result<bool, String> {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let Some(pending) = &self.pending else {
            return Ok(false);
        };
        match pending.try_recv() {
            Err(TryRecvError::Empty) => return Ok(false),
            Err(e) => return Err(e.to_string()),
            Ok(result) => result.map_err(|e| e.to_string())?,
        }
        let range = self
            .staging
            .get_mapped_range(..)
            .map_err(|e| e.to_string())?;
        let bird_bytes = u64::from(SAMPLES) * size_of::<TrackingPoint>() as u64;
        let points: &[TrackingPoint] = bytemuck::cast_slice(&range[..bird_bytes as usize]);
        self.positions.clear();
        self.velocities.clear();
        for p in points.iter().filter(|p| p.live != 0) {
            self.positions.push(Vec3::from_array(p.pos));
            self.velocities.push(Vec3::from_array(p.vel));
        }
        if range.len() >= bird_bytes as usize + size_of::<murmur_gpu::falcon::Falcon>() {
            let falcon: &murmur_gpu::falcon::Falcon = bytemuck::from_bytes(
                &range[bird_bytes as usize
                    ..bird_bytes as usize + size_of::<murmur_gpu::falcon::Falcon>()],
            );
            self.falcon_pos = falcon.is_present().then(|| falcon.position());
            self.falcon_vel = Vec3::from_array(falcon.vel);
        }
        self.sample_time = self.pending_time;
        drop(range);
        self.staging.unmap();
        self.pending = None;
        Ok(true)
    }

    /// Start one request if idle. The caller keeps rendering in either case.
    pub fn request(&mut self, gpu: &Gpu, birds: &wgpu::Buffer, time: f32) {
        self.request_with_falcon(gpu, birds, None, time);
    }

    /// Start one request if idle, optionally copying the falcon state in the same non-blocking staging buffer.
    pub fn request_with_falcon(
        &mut self,
        gpu: &Gpu,
        birds: &wgpu::Buffer,
        falcon: Option<&wgpu::Buffer>,
        time: f32,
    ) {
        if self.pending.is_some() {
            return;
        }
        let key = std::ptr::from_ref(birds) as usize;
        let bind = if let Some((_, b)) = self.bind_cache.iter().find(|(k, _)| *k == key) {
            b.clone()
        } else {
            let b = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera samples"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: birds.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.output.as_entire_binding(),
                    },
                ],
            });
            if self.bind_cache.len() >= 4 {
                self.bind_cache.clear();
            }
            self.bind_cache.push((key, b.clone()));
            b
        };
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("camera samples"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(SAMPLES.div_ceil(256), 1, 1);
        }
        let bird_bytes = u64::from(SAMPLES) * size_of::<TrackingPoint>() as u64;
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, bird_bytes);
        if let Some(falcon_buf) = falcon {
            encoder.copy_buffer_to_buffer(
                falcon_buf,
                0,
                &self.staging,
                bird_bytes,
                size_of::<murmur_gpu::falcon::Falcon>() as u64,
            );
        }
        gpu.queue.submit(Some(encoder.finish()));
        let (tx, rx) = mpsc::channel();
        self.staging
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        self.pending = Some(rx);
        self.pending_time = time;
    }

    /// Predict only the short telemetry delay; do not extrapolate stale samples
    /// across a long pause or an entire evening.
    pub fn predicted_positions(&self, time: f32) -> Vec<Vec3> {
        let dt = (time - self.sample_time).clamp(0.0, 0.25);
        self.positions
            .iter()
            .zip(&self.velocities)
            .map(|(p, v)| *p + *v * dt)
            .collect()
    }

    /// Predicted falcon position from the last non-blocking telemetry sample.
    pub fn predicted_falcon(&self, time: f32) -> Option<Vec3> {
        let dt = (time - self.sample_time).clamp(0.0, 0.25);
        self.falcon_pos.map(|p| p + self.falcon_vel * dt)
    }

    /// The falcon's velocity from the last telemetry sample, if one is hunting.
    pub fn falcon_velocity(&self) -> Option<Vec3> {
        self.falcon_pos.map(|_| self.falcon_vel)
    }

    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }
}
