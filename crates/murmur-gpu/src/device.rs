//! Requesting a wgpu device, shared by the windowed app and headless tests.
//!
//! Both paths ask for the same [`wgpu::PowerPreference`]. On the development
//! laptop (hybrid graphics) the windowed app is launched with `prime-run` so
//! the NVIDIA adapter is the one that answers; whatever adapter was found is
//! printed at startup and must be checked before believing any measurement
//! (see AGENTS.md and docs/setup.md).

use std::fmt;

use wgpu::{Adapter, Device, Instance, PowerPreference, Queue, RequestAdapterOptions, Surface};

/// A wgpu instance together with the adapter, device and queue opened on it.
pub struct Gpu {
    /// The instance the adapter and any surface belong to. Kept here so the
    /// app can recreate a lost surface (`Instance::create_surface`).
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl Gpu {
    /// Name, backend, device kind and driver of the adapter behind this device.
    pub fn info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

/// Why a device could not be opened.
#[derive(Debug)]
pub enum DeviceError {
    /// No adapter satisfied the request.
    NoAdapter(wgpu::RequestAdapterError),
    /// The adapter was found but rejected the device request.
    NoDevice(wgpu::RequestDeviceError),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NoAdapter(err) => write!(f, "no suitable GPU adapter: {err}"),
            DeviceError::NoDevice(err) => {
                write!(f, "the GPU adapter refused the device request: {err}")
            }
        }
    }
}

impl std::error::Error for DeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DeviceError::NoAdapter(err) => Some(err),
            DeviceError::NoDevice(err) => Some(err),
        }
    }
}

/// Requests a device that can present to `surface`.
pub fn for_surface(
    instance: &Instance,
    surface: &Surface<'_>,
    power: PowerPreference,
) -> Result<Gpu, DeviceError> {
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: power,
        compatible_surface: Some(surface),
        ..Default::default()
    }))
    .map_err(DeviceError::NoAdapter)?;
    open(instance.clone(), adapter, wgpu::Features::empty())
}

/// Requests a headless device, without a surface, for tests and benches.
///
/// Fails with [`DeviceError::NoAdapter`] when the machine has no usable
/// adapter (for example a GPU-less CI runner). Callers are expected to skip
/// GPU work with a clear message in that case rather than fail the suite;
/// an adapter that refuses a device, on the other hand, is a real error.
pub fn headless(power: PowerPreference) -> Result<Gpu, DeviceError> {
    headless_with(power, wgpu::Features::empty())
}

/// Like [`headless`], asking for `features` as well.
///
/// The adapter's own set is intersected in first, so asking for a feature the
/// adapter lacks opens the device without it rather than failing; callers that
/// need it (the benchmark needs `TIMESTAMP_QUERY`) must check
/// [`Gpu::adapter`]'s features and say so when it is missing.
pub fn headless_with(power: PowerPreference, features: wgpu::Features) -> Result<Gpu, DeviceError> {
    let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: power,
        ..Default::default()
    }))
    .map_err(DeviceError::NoAdapter)?;
    open(instance, adapter, features)
}

/// Opens the device on `adapter` and bundles it with the instance.
///
/// The device asks for the adapter's own limits rather than the WebGPU
/// defaults. The bird record is 112 bytes, so a three-million-bird run holds
/// 336 MB in one storage buffer: past the default 256 MB `max_buffer_size` and
/// 128 MB `max_storage_buffer_binding_size`. This is a native application run
/// on a known GPU (PLAN.md's target hardware), not a portable web page, so the
/// WebGPU portability floor is not the right bound here. The adapter's limits
/// are what the driver reports it can do and are always requestable.
fn open(
    instance: Instance,
    adapter: Adapter,
    features: wgpu::Features,
) -> Result<Gpu, DeviceError> {
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("murmuration-gpu: device"),
        required_features: features & adapter.features(),
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .map_err(DeviceError::NoDevice)?;
    Ok(Gpu {
        instance,
        adapter,
        device,
        queue,
    })
}
