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
//!
//! Phase 4 adds the headless `--analyze` run ([`analyze`]) in the reference
//! output format and the long-run coherence sampler and verdict
//! ([`coherence`]). `--analyze` measures the full reference metric set;
//! `--coherence` measures only what the criteria read, because the full set's
//! 15 m nearest-neighbour scan costs ~26 s per simulated second at 100,000
//! birds.

pub mod agitation;
pub mod analyze;
pub mod arc;
pub mod bench;
pub mod bird;
pub mod coherence;
pub mod device;
pub mod falcon;
pub mod grid;
pub mod params;
pub mod parity;
pub mod pass;
pub mod sim;

/// The WGSL `Bird` record and the flight-frame helpers, shared with the
/// renderer so the GPU layout has one definition across both crates
/// (`crates/murmur-gpu/shaders/bird.wgsl`).
pub const BIRD_WGSL: &str = include_str!("../shaders/bird.wgsl");

/// The bank model the shader runs, re-exported so the app's CLI does not need
/// its own dependency on the reference crate.
pub use murmur_ref::World;
pub use murmur_ref::flight::Turning;
/// The wing-flap amplitude the renderer's footprint needs, re-exported for the
/// same reason as [`Turning`].
pub use murmur_ref::params::FLAP_AMPLITUDE;
