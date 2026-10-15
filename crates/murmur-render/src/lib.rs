//! Rendering for murmuration-gpu.
//!
//! Phase 0 carries only the dusk palette the app clears to. The sky, bird
//! LODs, multiplicative darkness accumulation, TAA and tone mapping arrive
//! with phases 5–8 and reuse the *ideas* of cuda-particles' HDR pipeline
//! (accumulate in HDR, tone map after accumulation), not its code.

pub mod sky;
