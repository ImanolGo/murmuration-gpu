//! Requesting a wgpu device, shared by the windowed app and headless tests.
//!
//! Both paths ask for the same [`wgpu::PowerPreference`]. On the development
//! laptop (hybrid graphics) the windowed app is launched with `prime-run` so
//! the NVIDIA adapter is the one that answers; whatever adapter was found is
//! printed at startup and must be checked before believing any measurement
//! (see AGENTS.md and docs/setup.md).

use std::fmt;
use std::time::Duration;

use wgpu::{Adapter, Device, Instance, PowerPreference, Queue, RequestAdapterOptions, Surface};

/// How long a wait may block before the device is called stalled.
///
/// The longest submission in the project is a ten-step batch of the largest
/// simulated flock, a few seconds of GPU time; a wait an order of magnitude
/// longer than the work it covers cannot mean "slow", only "stopped". On the
/// development laptop that happens for real: the NVIDIA driver hangs on a long
/// continuous compute run with `Xid 109` ("CTX SWITCH TIMEOUT", a known Linux
/// driver bug, not a driver our shaders can upset), after which the submission's
/// fence is never signalled. `poll(wait_indefinitely)` then blocks for ever; a
/// bounded wait turns it into a [`Stalled`] the caller can act on — drop the
/// device and open a new one.
pub const WAIT_BUDGET: Duration = Duration::from_secs(60);

/// The GPU stopped making progress inside the wait budget.
///
/// This is the signature of the driver hang described on [`WAIT_BUDGET`], so a
/// long run treats it as "this device is dead, reopen and carry on" rather than
/// as a fatal error; the trajectory is repaired from the last sample
/// ([`crate::analyze::each_second_resilient`], [`crate::sim::RawState`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stalled {
    /// How long the wait was allowed to block before giving up.
    pub waited: Duration,
}

impl fmt::Display for Stalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the GPU made no progress for {:.0} s (a hung submission; the NVIDIA Xid 109 hang)",
            self.waited.as_secs_f32()
        )
    }
}

impl std::error::Error for Stalled {}

/// A wgpu instance together with the adapter, device and queue opened on it.
pub struct Gpu {
    /// The instance the adapter and any surface belong to. Kept here so the
    /// app can recreate a lost surface (`Instance::create_surface`).
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    /// How this device was asked for, so a long headless run can open a fresh
    /// one after a driver hang ([`DeviceRequest::open`]).
    request: DeviceRequest,
}

impl Gpu {
    /// Name, backend, device kind and driver of the adapter behind this device.
    pub fn info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// The request that opened this device, for reopening it after a stall.
    pub fn request(&self) -> DeviceRequest {
        self.request
    }

    /// Waits for the most recent submission to complete, giving up after
    /// `budget` and reporting [`Stalled`] instead of blocking for ever.
    ///
    /// The indefinite form hangs the process when the driver hangs (see
    /// [`WAIT_BUDGET`]); every wait on a path a long run takes must be bounded.
    pub fn wait_bounded(&self, budget: Duration) -> Result<(), Stalled> {
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(budget),
        }) {
            // `QueueEmpty` and `WaitSucceeded` both mean the wait was satisfied.
            Ok(_) => Ok(()),
            // `Timeout` is the stall; `WrongSubmissionIndex` cannot happen for a
            // wait that names no submission, so treating it as a stall is safe.
            Err(_) => Err(Stalled { waited: budget }),
        }
    }
}

/// A device request that can be replayed, so a run can reopen the same device.
///
/// [`DeviceRequest::open`] is headless (no surface): only the headless runs
/// (`--analyze`, `--coherence`) resume, and the windowed app never calls it.
#[derive(Debug, Clone, Copy)]
pub struct DeviceRequest {
    power: PowerPreference,
    features: wgpu::Features,
}

impl DeviceRequest {
    /// Opens a new headless device with the same request as the original.
    ///
    /// The caller must drop the stalled `Gpu` it replaces; holding one dead
    /// context while opening another is harmless (the two coexist), but the
    /// dead one is of no further use.
    pub fn open(&self) -> Result<Gpu, DeviceError> {
        headless_with(self.power, self.features)
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
    for_surface_with(instance, surface, power, wgpu::Features::empty())
}

/// Surface device with optional profiling features, matching headless_with.
pub fn for_surface_with(
    instance: &Instance,
    surface: &Surface<'_>,
    power: PowerPreference,
    features: wgpu::Features,
) -> Result<Gpu, DeviceError> {
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: power,
        compatible_surface: Some(surface),
        ..Default::default()
    }))
    .map_err(DeviceError::NoAdapter)?;
    open(instance.clone(), adapter, features, power)
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
    open(instance, adapter, features, power)
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
    power: PowerPreference,
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
    let features = features & adapter.features();
    Ok(Gpu {
        instance,
        adapter,
        device,
        queue,
        request: DeviceRequest { power, features },
    })
}
