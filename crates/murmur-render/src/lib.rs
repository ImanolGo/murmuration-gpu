//! Rendering for murmuration-gpu.
//!
//! Phase 0 carried only the dusk palette the app clears to. Phase 5 adds the
//! first real frame: a ground observer [`camera`], the dusk [`sky`] gradient by
//! viewing elevation, per-bird optical-depth coverage accumulated into an HDR
//! target, and a tone-mapped output. It reuses the *ideas* of cuda-particles'
//! HDR pipeline (accumulate in HDR, tone map after accumulation), not its code.
//! Phase 6 adds the birds themselves: a low-poly [`mesh`] starling, an impostor
//! quad and the phase-5 splat, chosen per bird by a compute pass that writes the
//! indirect draws, plus temporal anti-aliasing.

pub mod atmosphere;
pub mod camera;
pub mod mesh;
pub mod png;
pub mod renderer;
pub mod sky;
pub mod tracking;
