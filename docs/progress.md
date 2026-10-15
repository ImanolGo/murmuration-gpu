# Progress log

Working record, one section per phase: what was built, what was measured,
what is still open. Honesty rules apply (AGENTS.md): numbers below come from
the recorded runs, on the machine named in each section.

## Phase 0 — scaffold (2026-10-15)

**Built**

- Cargo workspace: `app` (package `murmuration-gpu`), `crates/murmur-ref`,
  `crates/murmur-gpu`, `crates/murmur-render`. Edition 2024, Rust 1.95+.
- `murmur-gpu::device`: one device-request path for the windowed app
  (`for_surface`) and one for headless tests and benches (`headless`), both
  asking for `PowerPreference::HighPerformance`, with typed errors
  (`DeviceError`).
- `murmur-render::sky`: the dusk palette shared with the reference
  (murmuration `render/sky.rs` `DUSK_STOPS`) and `dusk_clear(format)`,
  which encodes the clear colour correctly for `*Srgb` and plain surface
  formats.
- `app`: a winit window, a wgpu device + surface, a clear to the dusk
  colour, and an adapter line at startup. Quits on `q`, `Esc` or close.
- A GPU smoke test (`murmur-gpu/tests/gpu.rs`): headless device + buffer
  round-trip through the queue; skips with a message when no adapter is
  found.
- CI (`.github/workflows/ci.yml`): fmt, clippy with `-D warnings`, tests on
  Linux/macOS/Windows; llvmpipe on the Linux runner so the GPU test runs
  there instead of skipping; an msrv build at 1.95.
- `docs/setup.md`: Vulkan ICDs on Arch, `prime-run`, checking the adapter.

**Measured (2026-10-15, Arch Linux, GTX 1650 Mobile, Wayland)**

- `prime-run cargo run --release` (windowed) prints:

  ```
  murmuration-gpu: adapter "NVIDIA GeForce GTX 1650 with Max-Q Design" on Vulkan (DiscreteGpu), driver "NVIDIA" 615.78.08
  murmuration-gpu: surface 1268x1380 Bgra8UnormSrgb, present mode Mailbox
  ```

  and presented frames for the whole 10 s probe with no surface errors.
  Honest caveat: the mechanics (window, adapter, swap chain, present loop)
  are verified by that output; the on-screen pixels were not captured, so
  the dusk colour itself is checked by unit test, not by eye.
- `cargo test` (headless, no prime-run): 3 passed, 0 failed. The headless
  request picked the GTX 1650 (Vulkan) even without `prime-run`;
  under `prime-run`, the same. One harmless `MESA-EGL: DRI2` warning
  appears during enumeration on the Intel side.
- `cargo fmt --all --check` and
  `cargo clippy --workspace --all-targets -- -D warnings`: clean.

**Notes**

- "Latest stable" from PLAN.md resolved to wgpu 30.0.1 + winit 0.30.13.
  Their APIs differ from older material: `request_adapter` returns
  `Result`, surface acquisition is the `CurrentSurfaceTexture` enum,
  presentation is `queue.present`, mapping is
  `map_async(mode, range, callback)` and `get_mapped_range` returns
  `Result`, render passes gained `depth_slice`/`multiview_mask`. The code
  follows wgpu 30's own `hello_windows` example rather than memory.
- `surface.get_default_config` picked present mode `Mailbox`; phase 5
  should choose the present mode deliberately.
- `vulkaninfo` is not installed on this machine; `docs/setup.md` says how
  (`pacman -S vulkan-tools`).

**Open**

- (none) — committed with this phase. `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test`
  were re-run clean at commit time (3 tests; the GPU test ran on the
  GTX 1650, not skipped).

## Phase 1 — reference model as oracle (done, 2026-10-15)

### The library check

PLAN.md asked whether `murmuration` exposes its simulation as a library
usable without the terminal code. The answer was a third case, not a plain
yes/no (murmuration 0.2.0):

- Yes: `murmuration::sim` is public and pure — `World`, `params`, `social`,
  `flight`, `neighbours`, `escape`, `falcon`, `arc`, `turns`, `analysis` —
  and it never references `crate::cli` or `crate::render`. `main.rs` drives
  exactly this API for `--analyze`.
- No: the crate cannot be built without the terminal stack. `ratatui`,
  `crossterm`, `clap` and `signal-hook` are unconditional dependencies,
  `sim::flight::Turning` even derives `clap::ValueEnum`, every module is
  `#[doc(hidden)]` with a "no stability guarantee" note, and the published
  crate declares no features.

The smallest upstream change (a default `tui` feature, ~15 lines) was
drafted and offered, but the maintainer chose instead to **reimplement the
model in `murmur-ref`**: no dependency on `murmuration`, no vendored copy,
`murmur-gpu` fully independent of the terminal crate. This supersedes the
"depend or vendor" wording in PLAN.md phase 1.

### Built

- `crates/murmur-ref/src/`: `params`, `neighbours`, `flight`, `social`,
  `escape`, `falcon`, `arc`, `world` — the model, mirroring the reference
  module for module, without `clap` or any terminal crate. Dependencies are
  only `glam`, `rand` and `rand_chacha`, pinned to the versions the
  reference resolves to (glam 0.34.1, rand 0.9.5, rand_chacha 0.9.0), so the
  arithmetic and random streams are identical.
- `analysis`: the reference `--analyze` metrics and `Stats::line()`, kept
  byte-for-byte comparable with the terminal output.
- `stats`: the extra statistics PLAN.md phase 1 lists — nearest-neighbour
  distance over all birds, polarisation, flock extent and aspect ratios,
  border-vs-centre density, and a speed distribution (`mean ± std`,
  p10/p50/p90). The reference `analysis.rs` has none of these except NND
  and neighbour count.
- `state`: `export`/`import` as a `Snapshot` (birds plus the world clock,
  `z0` and the model knobs). See `docs/model.md` for why it is not a bare
  `Vec<Bird>`.
- `src/bin/analyze.rs`: headless `--analyze` printing the reference format,
  plus `--extended` for the `stats` line.

### Acceptance — reproducing the oracle

`murmuration --analyze 120 --birds 2000 --seed 42` was recorded as the
baseline before the decision (`docs/reference-analyze-2000-42.txt`; 16.3 s
wall in the terminal project). Running the reimplementation:

```sh
cargo run -p murmur-ref --bin analyze --release -- \
    --birds 2000 --seed 42 --seconds 120
```

All **120 per-second lines are byte-for-byte identical** to the recorded
baseline (checked with `diff` on the `t=` lines). The settled means
(t = 30–120 s) match the values tabulated earlier: |N| 6.62, NND 0.77 m,
flat 0.90, nn_ab 0.095, hstd 2.06 m, cluster ≥ 99.9 %, v 10.0, I2/I1 3.43.
The reference took 16.3 s for the run; this crate takes 23.9 s in release
(sequential decisions instead of `rayon`; same results).

`tests/oracle.rs` asserts exact equality at t = 20 s and t = 60 s on every
`cargo test`; the full 120 s check is `#[ignore]`d and passes with
`cargo test -p murmur-ref --test oracle -- --ignored` (≈23 s).

### Deviations recorded

`docs/model.md` collects the reference's documented adaptations to the paper
(banked coordinated turn, the `1/cos β` lift, the stable adaptive radius,
all-round cohesion with `W_COH = 0.65`, the capped vertical roost force, the
committed signed roost return, the separation Gaussian factor of 2, the
initial condition) and the four `murmur-ref`-specific choices: sequential
decisions, an extra `--extended` line, the `Snapshot` exchange type, and the
deferral of the turn/agitation experiments to phase 7. The reference's known
gaps (the flock mills over the roost rather than crossing it; long runs grow
elongated and can split) are recorded there too, unfixed on purpose.

### Open

- (none) for phase 1. Phase 2 is the first GPU simulation (brute-force
  neighbour search) and the `--parity` harness, which will use `state::export`
  and `state::import`.

## Phase 2 — GPU simulation, brute force (2026-10-15)

**Built**

- `murmur-gpu::params`: the 192-byte `#[repr(C)]` `Params` uniform block,
  built field by field from `murmur_ref::params`, so the constants have one
  home and cannot drift. `turning` is the shader's `u32` discriminant.
- `murmur-gpu::bird`: the 112-byte `#[repr(C)]` `Bird` record the shaders read
  and write, with conversions to and from `murmur_ref::state::Bird`.
- `murmur-gpu::sim`: `Sim` owns the compute pipelines, the double-buffered
  bird storage, the shared clock buffer and the readback buffer. Three tiny
  passes per step (`tick`, `decide`, `integrate`) with the two buffer
  directions having their own bind groups, so **no queue write per step**:
  `advance(steps)` encodes many steps into one command buffer.
- `crates/murmur-gpu/shaders/sim.wgsl`: `flight::{forward, side_of, body_up,
  step_bird}`, `params::radius_step`, `escape::zigzag_roll` and
  `social::decide`, mirrored function for function, with a header naming each
  one. The neighbour search is brute force (O(N²)), as phase 2 asks.
- `murmur-gpu::parity`: the CPU/GPU comparison — `single_step_max_rel_error`,
  `single_step_noise_floor`, `lock_in`, `run_seed` and a metric `table`.
- `app`: a headless `--parity` mode (`--birds`, `--seed`, `--seconds`,
  `--seeds`), and the `w_xi` seam: `World::w_xi` / `Snapshot::w_xi`
  (default `params::W_XI`) lets the single-step check switch the random force
  off on both engines. Default preserves the oracle exactly.
- Tests: `tests/layout.rs` (size, offsets and field names checked against the
  WGSL structs — the AGENTS.md drift guard) and `tests/parity.rs`.

**Measured** — GTX 1650 Mobile, driver 615.78.08, Vulkan, 2,000 birds. The
adapter is printed at the top of every run.

- **Single step, noise off:** `1.065e-7` relative error per bird (positions
  and velocities), against the `1e-4` target. This is the deterministic match:
  the shader reproduces the reference forces and integration to float noise.
- **Single step, noise on:** `7.2e-8` (relative to a bird's position). The
  residual is the unmatched-noise floor, not a model error.
- **Lock-in, t = 1–5 s:** every reference metric within 10 % on all five seeds
  (worst deviations in `docs/parity-2000-five-seeds.txt`). The two engines are
  on the same trajectory until chaos amplifies the tiny RNG difference.
- **Statistical parity, 60 s × 5 seeds:** settled means (t = 30–60 s) within
  10 % for every self-averaging metric and within the relaxed 50 % for the
  global-shape metrics `I2/I1`, `I3/I1`, `flat`. Full per-metric table in
  `docs/parity-2000-five-seeds.txt`; the run takes 134 s.
- Reproduction: `cargo test -p murmur-gpu --test parity` (fast) and
  `cargo test -p murmur-gpu --test parity -- --ignored` (the 5-seed
  acceptance), or the app's
  `cargo run -p murmuration-gpu --release -- --parity --birds 2000 --seed 42`.
- **Window path re-checked** (phase 0 regression after the CLI change):
  `prime-run cargo run -p murmuration-gpu --release` prints `adapter "NVIDIA
  GeForce GTX 1650 with Max-Q Design" on Vulkan (DiscreteGpu), driver "NVIDIA"
  615.78.08` and creates an 948×1020 `Bgra8UnormSrgb` surface (present mode
  Mailbox). No performance numbers are claimed here; those wait for the
  timestamp queries of a later phase.
- **Oracle unchanged:** the full 120 s `murmur-ref` oracle is still
  byte-for-byte identical after the `w_xi` seam
  (`cargo test -p murmur-ref --test oracle -- --ignored`).

**PLAN changes recorded** (also in `PLAN.md`, phase 2)

- **Tolerances depend on the observable.** A single 2,000-bird flock is
  chaotic in its global shape but self-averaging in its local structure, so
  "±10 % on each metric" is two different targets. Local statistics
  (`|N|`, NND, cluster, alignment, radius, spread, extent, height, speed,
  bank, centroid radius) hit ±10 %; the shape metrics (`I2/I1`, `I3/I1`,
  `flat`) do not self-average over a minute at this size — the CPU's own
  `I2/I1` ranged 3.2–30.4 across the five seeds — so they are compared at
  ±50 % and are left to tighten on phase 4's larger, longer runs. A
  short-horizon **lock-in** check (all metrics at ±10 % for the first seconds)
  keeps a tight comparison and is what actually proves the dynamics match.
- **Noise is an adaptation, not a target.** The GPU uses a counter-based PCG
  stream per bird keyed by `(seed, id)`; the CPU's `ChaCha8` stream position
  cannot be extracted. Bit-exact noise parity stays out of scope (as the stack
  notes say), so short-horizon parity is exact only with the noise off.
- **`w_xi` is a test seam**, not a model knob: default `params::W_XI`, so the
  phase 1 oracle is unchanged (re-verified byte for byte after the edit).
- **The accumulator is deferred.** `Sim::advance` runs the fixed 5 ms step
  now; the wall-clock accumulator and the 1/120 s presentation cap belong to
  the real-time render loop and arrive with the first render.
- **Documented commands use `-p`.** The workspace has two binaries
  (`murmuration-gpu` and `murmur-ref`'s `analyze`), so `cargo run` at the root
  is ambiguous; `README.md`, `AGENTS.md` and the app now say
  `cargo run -p murmuration-gpu`.

**Open**

- The brute-force neighbour search is O(N²) on purpose (phase 2). Phase 3
  replaces it with the spatial grid and re-runs this parity.
- The 112-byte bird record carries fields the later phases need; phase 9 may
  shrink it with the benchmark.
- `nn_ab` and the shape metrics are noisy at 2,000 birds; the parity table
  shows the absolute bound next to the difference so a loose metric is visible
  rather than hidden.
