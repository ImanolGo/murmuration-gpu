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
    open(instance.clone(), adapter)
}

/// Requests a headless device, without a surface, for tests and benches.
///
/// Fails with [`DeviceError::NoAdapter`] when the machine has no usable
/// adapter (for example a GPU-less CI runner). Callers are expected to skip
/// GPU work with a clear message in that case rather than fail the suite;
/// an adapter that refuses a device, on the other hand, is a real error.
pub fn headless(power: PowerPreference) -> Result<Gpu, DeviceError> {
    let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: power,
        ..Default::default()
    }))
    .map_err(DeviceError::NoAdapter)?;
    open(instance, adapter)
}

/// Opens the device on `adapter` and bundles it with the instance.
fn open(instance: Instance, adapter: Adapter) -> Result<Gpu, DeviceError> {
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("murmuration-gpu: device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
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
