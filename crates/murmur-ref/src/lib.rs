//! The CPU reference model, kept as the oracle for the GPU simulation.
//!
//! PLAN.md phase 1 brings the terminal project's model in here, either by
//! depending on the `murmuration` crate or by vendoring its `sim` module with
//! attribution. Its force functions, parameters and documented adaptations
//! (murmuration's `DEVELOPMENT.md`) are the specification the GPU shaders
//! mirror one function at a time, so nothing in this crate may ever be tuned
//! to make the GPU side look better (see AGENTS.md, "The model is the
//! specification").
//!
//! Status: phase 0 scaffold. The model itself is not vendored yet; the choice
//! between depending and vendoring is pending (see docs/progress.md).
