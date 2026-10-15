//! A GPU smoke test: request a headless device and round-trip a buffer
//! through the queue.
//!
//! Skipped with a clear message when the machine has no adapter, so `cargo
//! test` still passes on GPU-less machines (PLAN.md phase 0). The adapter
//! actually used is printed, because on hybrid-graphics laptops the choice
//! matters (AGENTS.md).

use murmur_gpu::device::{self, DeviceError};
use wgpu::{BufferDescriptor, BufferUsages, MapMode};

#[test]
fn headless_device_roundtrips_a_buffer() {
    let gpu = match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => gpu,
        Err(DeviceError::NoAdapter(err)) => {
            eprintln!("SKIP: no suitable GPU adapter was found ({err}); GPU test not run");
            return;
        }
        Err(err) => panic!("an adapter was found but opening its device failed: {err}"),
    };
    let info = gpu.info();
    eprintln!(
        "headless device: {} ({:?}, {:?}), driver {} {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    // Write a pattern, copy it back out and compare: the smallest end-to-end
    // check that queue writes, command submission and buffer mapping all work
    // on this machine.
    const N: usize = 1024;
    let bytes = (N * 4) as u64;
    let pattern: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
    let storage = gpu.device.create_buffer(&BufferDescriptor {
        label: Some("smoke: storage"),
        size: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = gpu.device.create_buffer(&BufferDescriptor {
        label: Some("smoke: staging"),
        size: bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&storage, 0, bytemuck::cast_slice(&pattern));
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, bytes);
    gpu.queue.submit(Some(encoder.finish()));

    let (mapped, done) = std::sync::mpsc::channel();
    staging.map_async(MapMode::Read, .., move |result| {
        let _ = mapped.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("waiting for the GPU queue to drain");
    done.recv()
        .expect("the poll ran the mapping callback before returning")
        .expect("mapping the staging buffer for reading");

    let range = staging
        .get_mapped_range(..)
        .expect("reading the mapped range");
    let read: &[f32] = bytemuck::cast_slice(&range[..]);
    assert_eq!(read, &pattern[..]);
}
