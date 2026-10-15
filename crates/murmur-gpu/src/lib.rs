//! The GPU simulation of murmuration-gpu.
//!
//! Phase 0 provides device creation only: one request path for the windowed
//! app ([`device::for_surface`]) and one for headless tests and benches
//! ([`device::headless`], no surface, skipped with a clear message when the
//! machine has no adapter). The compute passes arrive with the later phases
//! and mirror the reference model's functions one by one, keeping its names
//! and order of operations.
//!
//! Phase 2 adds the first compute passes ([`sim`]) with a brute-force
//! neighbour search, the GPU record ([`bird`]) and uniform block ([`params`]),
//! and the CPU/GPU comparison ([`parity`]).
//!
//! Phase 3 adds the spatial grid ([`grid`]) that replaces the brute-force
//! neighbour search with a hash-bucket counting sort, the optional per-pass
//! timestamps ([`pass`], used by [`bench`]) and the benchmark itself.

pub mod bench;
pub mod bird;
pub mod device;
pub mod grid;
pub mod params;
pub mod parity;
pub mod pass;
pub mod sim;
