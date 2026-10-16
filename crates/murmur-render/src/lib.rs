//! Rendering for murmuration-gpu.
//!
//! Phase 0 carried only the dusk palette the app clears to. Phase 5 adds the
//! first real frame: a ground observer [`camera`], the dusk [`sky`] gradient by
//! viewing elevation, per-bird optical-depth coverage accumulated into an HDR
//! target, and a tone-mapped output. It reuses the *ideas* of cuda-particles'
//! HDR pipeline (accumulate in HDR, tone map after accumulation), not its code.
//! Bird LODs and TAA arrive with phase 6.

pub mod camera;
pub mod png;
pub mod renderer;
pub mod sky;
