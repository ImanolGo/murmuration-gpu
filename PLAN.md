# PLAN — murmuration-gpu

This is the working plan for a coding agent building murmuration-gpu. Work
through the phases in order. Each phase ends with acceptance criteria. A phase
is done only when every criterion is met and verified by running something,
not by reading the code. Read [AGENTS.md](AGENTS.md) before starting.

## Context

- Source model: [ImanolGo/murmuration](https://github.com/ImanolGo/murmuration),
  a Rust, deterministic, fixed-step StarDisplay-style flock (terminal
  renderer). Read its README, `DEVELOPMENT.md`, `PLAN.md` and `src/` before
  phase 1. Its force functions, parameters and documented adaptations are
  the specification for this project.
- Prior GPU work: [ImanolGo/cuda-particles](https://github.com/ImanolGo/cuda-particles)
  (CUDA and OpenGL). Reuse its ideas (HDR accumulation, tone mapping, fixed
  step capped at 1/120 s, display-free GPU regression tests, recording
  script), not its code or API.
- Target hardware: Arch Linux, NVIDIA GTX 1650 Mobile (Turing, 4 GB, about
  128 GB/s), hybrid graphics. Vulkan backend, launched with `prime-run`.
- Stack: Rust 1.95+, latest stable `wgpu`, `winit`, `bytemuck`, `glam`,
  `clap`. WGSL shaders. Check current API docs for `wgpu` and `winit`
  rather than relying on memory; both change between versions.

## Global targets (estimates to verify, not promises)

| Birds     | Target on GTX 1650 Mobile |
| --------- | ------------------------- |
| 100,000   | 60 fps, sim + render      |
| 1,000,000 | 60 fps sim, 30+ fps total |
| 3,000,000 | 30 fps, reduced effects   |

If a target is missed after phase 9, report the measured numbers and the
bottleneck. Do not hide it.

## Data layout

Bird state lives in GPU storage buffers, double-buffered (read current,
write next). Start with this layout and change it only with a benchmark:

```
struct Bird {            // 64 bytes, 16-byte aligned
  pos: vec3<f32>,  speed: f32,
  fwd: vec3<f32>,  roll: f32,      // unit forward vector, bank angle
  vel: vec3<f32>,  radius: f32,    // adaptive perception radius
  phase: f32, agitation: f32, flags: u32, id: u32,
}
```

`Params` is one uniform struct shared by Rust (`#[repr(C)]`, `bytemuck`) and
WGSL, generated or checked by a test so the two never drift. The original
`id` must survive every reorder so that parity tests and the falcon can
track individual birds.

---

## Phase 0 — Scaffold

- Cargo workspace: `crates/murmur-ref`, `crates/murmur-gpu`,
  `crates/murmur-render`, `app`.
- `app` opens a `winit` window, creates a `wgpu` device and clears to a dusk
  colour. Prints adapter name and backend at startup.
- Headless device creation (no surface) works for tests and benches.
- CI: fmt, clippy with `-D warnings`, tests (GPU tests skipped when no
  adapter is found, with a clear message).
- `docs/setup.md`: Vulkan drivers on Arch, `prime-run`, checking with
  `vulkaninfo`.

**Accept:** `prime-run cargo run --release` shows the window on the NVIDIA
adapter; `cargo test` passes headless.

## Phase 1 — Reference model as oracle (done)

- Bring the CPU model into `murmur-ref`. First check whether the published
  `murmuration` crate exposes the simulation as a library API. If it does,
  depend on it. If not, vendor the simulation module with attribution and
  strip the terminal code.
- **Update (2026-10-15):** the check found a third case — the simulation is
  a public, pure module
  (`sim::` references neither `cli` nor `render`, and `main.rs` drives it
  directly) but the crate cannot be built without the terminal dependencies
  (`ratatui`, `crossterm`, `clap`, `signal-hook` are unconditional;
  `sim::flight::Turning` derives `clap::ValueEnum`; the published crate has no
  features). A minimal default-`tui`-feature patch was drafted and offered, but
  the maintainer chose to **reimplement the model in `murmur-ref`** — no
  dependency and no vendoring, so `murmur-gpu` stays independent of the
  terminal crate. This supersedes the "depend or vendor" wording above.
- Add a flock statistics module (or reuse the existing one): nearest-neighbour
  distance, mean number of interacting neighbours, polarisation, flock
  extent and aspect ratios, border vs centre density, speed distribution.
- Add `export(state) -> Vec<Bird>` and `import(&[Bird])` so CPU and GPU
  can start from the same state. (Implemented as a `Snapshot`: birds plus the
  scalar world state, which a bare `Vec<Bird>` cannot carry. See
  `docs/model.md`.)

**Accept:** a headless run of `murmur-ref` at 2,000 birds, seed 42, reproduces
the statistics reported by the terminal version's `--analyze` (the oracle
baseline is recorded in `docs/reference-analyze-2000-42.txt`).
**Met (2026-10-15):** all 120 per-second lines are byte-for-byte identical to
the baseline; `tests/oracle.rs` asserts it (t = 20 s and 60 s every run, the
full 120 s with `--ignored`). Details in `docs/progress.md`.

## Phase 2 — GPU simulation, brute force

- Port the per-bird force step to WGSL one function at a time, keeping the
  same names and order as the CPU code. Neighbour search is O(N²) here on
  purpose: it is the simplest correct version.
- Fixed-step loop, accumulator, step capped at 1/120 s.
- `--parity`: run CPU and GPU from the same imported state for 60 simulated
  seconds; compare statistics each second.

**Accept:** at 2,000 birds the GPU statistics stay within agreed tolerances
of the CPU ones (start with ±10 % on each metric, then tighten) across five
seeds. Single-step test: one step from the same state matches the CPU to
about 1e-4 relative error per bird.

## Phase 3 — Spatial grid and topological neighbours

- Uniform grid sized from the typical perception radius. Passes:
  1. compute cell index per bird;
  2. histogram with atomics;
  3. exclusive prefix scan (workgroup scan plus block sums);
  4. scatter into sorted order and reorder the bird state.
- Neighbour pass: each bird scans the cells its adaptive radius covers,
  keeps the k = 7 nearest in a small register array, and updates its radius
  towards the target neighbour count, as in the model.
- Unbounded worlds: hash cells into a fixed-size table, or recentre the grid
  on the flock each step. Choose one, document why.
- Unit tests for scan and sort with random inputs, compared to CPU results.

**Accept:** at 2,000 birds the neighbour sets match brute force for at least
99 % of birds. Parity from phase 2 still passes. `--bench` reports per-pass
GPU timestamps at 100k, 1M and 3M birds.

## Phase 4 — Full flight model

- All remaining behaviour from the reference: boundary-weighted cohesion,
  banked turning from requested roll, cruise speed, preferred altitude, roost
  attraction and committed returns, random perturbation with a seeded
  per-bird generator (for example PCG keyed by `id` and step).
- `--analyze` in the app, matching the reference output format.

**Accept:** parity passes with all behaviours on. At 100k birds the flock
keeps a coherent display over the roost for 10 simulated minutes without
collapsing or dispersing, judged by the statistics.

## Phase 5 — First render: sky and coverage

- Ground observer camera, following the flock and framing it, as in the
  terminal version.
- Dusk sky: gradient by viewing elevation and time of evening first. A
  physically inspired atmosphere can come later.
- Every bird as a point that deposits coverage into an HDR target with
  multiplicative blending (`dst *= 1 - coverage`). Coverage depends on
  distance and wing exposure (Costanzo et al.).
- Tone map and output.

**Accept:** at 1M birds the image reads as a dark, dense flock with lighter
edges against the sky, similar in feel to the reference photographs. Capture
stills into `docs/`.

## Phase 6 — Birds

- Low-poly starling mesh (10–20 triangles), instanced. Wingbeat in the vertex
  shader from `phase`; orientation from `fwd` and `roll`.
- LOD by projected size, chosen in a compute pass that writes indirect draw
  arguments and per-LOD instance lists: mesh, flapping impostor quad,
  coverage point.
- Temporal anti-aliasing for sub-pixel birds, with care around shimmer.

**Accept:** no popping visible between LOD levels in a recorded fly-past; no
obvious shimmer in a static camera shot of a distant flock; frame time at
1M birds reported.

## Phase 7 — Falcon and agitation

- Falcon with patrol, climb, dive and retreat, autonomous or mouse-steered.
- Flee forces, close-threat expansion and copied zigzags with the
  `zigzag-k` threshold. Copying reads the previous state buffer, so it
  propagates one neighbour per step, as in the model.
- `--agitation` experiment matching the reference output.

**Accept:** a falcon attack produces visible dark waves; agitation
propagation speed is within the range reported by the reference at the same
bird count.

## Phase 8 — The evening

- Arrival, display, roosting and night phases, as in the terminal version.
- Reeds in the foreground with wind, stars at night.
- `--seed`, `--birds`, `--falcon`, `--no-arc`, `--arc-minutes`, matching the
  terminal CLI where they make sense.
- Recording script writing a GIF/WebP and a still into `docs/`, captured
  from the running binary at a fixed simulation step.

**Accept:** a full evening runs unattended; the README's GIF is recorded
from the real app.

## Phase 9 — Performance

- Profile with timestamp queries and Nsight Graphics or RenderDoc.
- Candidates, each kept only if a benchmark shows a gain: sorting less often
  than every step, half-precision for render-only fields, structure-of-arrays
  layout, subgroup operations in the scan, merging passes.
- Update the README with measured numbers and hardware.

**Accept:** the global targets table is either met or replaced with measured
numbers and the bottleneck for each.

## Out of scope for now

- Exact global k-nearest-neighbour search.
- Bit-exact GPU determinism.
- A full implementation of the inertial spin model (only the simplified
  experiment from the reference, if time allows).
- Web/WASM builds (possible later through wgpu's WebGPU backend).
