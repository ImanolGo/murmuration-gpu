//! The GPU simulation of murmuration-gpu.
//!
//! Phase 0 provides device creation only: one request path for the windowed
//! app ([`device::for_surface`]) and one for headless tests and benches
//! ([`device::headless`], no surface, skipped with a clear message when the
//! machine has no adapter). The compute passes arrive with the later phases
//! and mirror the reference model's functions one by one, keeping its names
//! and order of operations.

pub mod device;
