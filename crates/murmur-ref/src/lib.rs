//! The CPU reference model, kept as the oracle for the GPU simulation.
//!
//! This is a phase 1 reimplementation of the model from the sibling
//! [`murmuration`](https://github.com/ImanolGo/murmuration) project (MIT): same
//! force functions, names, constants and order of operations, without the
//! terminal stack and without depending on that crate. Its force functions,
//! parameters and documented adaptations (murmuration's `DEVELOPMENT.md`,
//! collected in `docs/model.md`) are the specification the GPU shaders mirror
//! one function at a time, so nothing in this crate may ever be tuned to make
//! the GPU side look better (see AGENTS.md, "The model is the specification").
//!
//! Layout mirrors the reference module for module: [`params`], [`neighbours`],
//! [`flight`], [`social`], [`escape`], [`falcon`], [`arc`], [`world`]. On top
//! of that, [`analysis`] reproduces the reference `--analyze` metrics (so the
//! output can be diffed against the recorded oracle) and [`stats`] adds the
//! extra statistics PLAN.md phase 1 asks for.

pub mod analysis;
pub mod arc;
pub mod escape;
pub mod falcon;
pub mod flight;
pub mod neighbours;
pub mod params;
pub mod social;
pub mod state;
pub mod stats;
pub mod world;

pub use state::{Bird, Snapshot, export, import};
pub use world::World;
