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

## Phase 3 — spatial grid and topological neighbours (2026-10-15)

**Built**

- `shaders/common.wgsl`: `Params`, `Bird` and the cell/bucket hash, prepended
  (in Rust, with `format!`) to both shader modules so the structs have one
  definition. `tests/layout.rs` now checks against this file; `Params` grew to
  208 bytes with `buckets` and `inv_cell`.
- `shaders/grid.wgsl`: the grid build — `grid_clear`, `grid_histogram`,
  `grid_scan_blocks` (level-1 exclusive scan, 2048 buckets per workgroup),
  `grid_scan_totals` (one workgroup scans the block totals, in place),
  `grid_scan_add` (adds the block offsets back and re-seeds the scatter
  cursor), `grid_scatter`, `grid_sort_buckets` (fixed order inside a bucket).
  Two buffers play two roles each, so the pipeline needs only eight storage
  buffers per stage, the WebGPU limit.
- `shaders/sim.wgsl`: `decide` now walks the cells the radius covers through
  `offsets`/`sorted` and keeps every bird inside the radius, mirroring
  `Grid::for_each_within`; the neighbour sums are factored into value-returning
  `acc_add`/`cen_add` helpers. A second entry point, `decide_bruteforce_kernel`,
  keeps the phase 2 O(N²) walk for the acceptance test. The distance test is
  `d² ≤ r²`, the oracle's test.
- `crates/murmur-gpu/src/grid.rs`: the grid's buffers, its seven pipelines, the
  sizing policy (`buckets_for`), the dispatch, and read-back helpers for the
  unit tests.
- `crates/murmur-gpu/src/pass.rs`: one place that records a compute dispatch,
  with optional GPU timestamp writes (`STEP_PASSES`, `TIMESTAMPS_PER_STEP`,
  `Timeline`).
- `crates/murmur-gpu/src/bench.rs` and `--bench` in the app: per-pass GPU
  timestamps at 100k, 1M and 3M birds.
- `Sim` owns the grid, binds its buffers in the same bind group as the birds,
  and gains `advance_bruteforce`, `advance_timeline`, `write_birds`,
  `rebuild_grid`. `device::headless_with(features)` and adapter limits.

**Measured** (GTX 1650 Max-Q, driver 615.78.08, Vulkan, `prime-run`)

- **Neighbour sets match brute force:** at 2,000 birds, 1,000 steps into a
  developed flock with every bird forced to decide, the grid and the
  brute-force walks give the same `neighbour_count` for **2,000/2,000** birds,
  the same centrality, and a worst steering relative error of `1.94e-6` — the
  two paths only differ in the order they add the same seven terms.
- **Scan and sort vs CPU:** the bucket histogram's exclusive prefix sums match
  a CPU computation exactly, and the sorted order is a permutation grouped by
  bucket, at 500, 2,000 and 5,000 birds (random initial conditions).
- **Phase 2 parity still passes:** single step `1.065e-7` (idem noise-on
  `7.216e-8`); lock-in on all five seeds; 60 s × 5 seeds settled means — full
  table regenerated in `docs/parity-2000-five-seeds.txt` (no `FAIL`), via
  `cargo run -p murmuration-gpu --release -- --parity --birds 2000`.
- **`--bench`** (10 timed steps, in `docs/bench-gtx1650.txt`):

  | birds | step | grid build | decide | integrate | bird-steps/s |
  |------:|-----:|-----------:|-------:|----------:|-------------:|
  | 100,000 | 14.95 ms | 0.43 ms | 14.05 ms | 0.48 ms | 6.7 M |
  | 1,000,000 | 162.9 ms | 7.78 ms | 150.5 ms | 4.6 ms | 6.1 M |
  | 3,000,000 | 498.8 ms | 24.98 ms | 460.1 ms | 13.7 ms | 6.0 M |

  The cost is almost all in `decide`, and it scales linearly with the bird
  count: the number of candidates per query is set by the 4 m cell size against
  a ~1.5 m perception radius, not by N. That is the phase 9 target, not this
  phase's; the whole grid build is 3–5 % of the step.
- **Window path re-checked** after the device-limits change: `prime-run
  ./target/release/murmuration-gpu` still prints the GTX 1650 on Vulkan and a
  948×1020 `Bgra8UnormSrgb` surface (Mailbox).

**Found and fixed: the scatter was not reproducible.** The first version used
the atomic scatter order directly. Two runs of the same command then gave
slightly different flocks, which the lock-in test saw as seed 4 passing in one
run and failing in the next (`nn_ab` moved by more than its own tolerance). A
parallel atomic scatter cannot fix an order, so `grid_sort_buckets` sorts each
bucket's range by bird index — a few milliseconds, since the table is sized at
about two birds per bucket. Repeated runs are now byte-identical, which the
model (deterministic, AGENTS.md) and the test suite both need.

**Changed: the lock-in check runs with the random force off.** It used to run
noise-on, i.e. compare two independent realizations, which is the same weak
statement the 60 s run makes. Measuring the divergence showed why that could
not stay: even *without* noise the mean bird-to-bird difference reaches 0.3 m
after 1 s and 3.7 m after 5 s (a ~1e-7 per-step difference amplified), so the
comparison can only be about the same path, not about averaging independent
runs. With the noise off every metric stays within 10 % for 5 s, the worst
being `nn_ab` at 40 % of its bound. This is a change to a phase 2 check and is
recorded in `PLAN.md` phase 2's spirit and in `docs/model.md`.

**PLAN changes recorded** (also in `PLAN.md`, phase 3)

- **Radius set, not k-nearest.** PLAN.md phase 3's "keeps the k = 7 nearest" is
  wrong about the model; `N_C = 6.5` is a target the adaptive radius is drawn
  towards, and `decide` uses every bird inside it, plus a 2R walk for
  centrality. The grid mirrors `Grid::for_each_within`.
- **Unbounded worlds: hash table, not recentring.** It mirrors the oracle and
  needs no reduction pass. The table is sized from the bird count; the cell
  size is the model constant `GRID_CELL = 4.0 m`, so the sets are identical.
- **`sorted` holds bird indices, not reordered records.** The reorder is via
  indirection; moving the 112-byte records is left to phase 9.
- **One extra pass, not in the plan,** fixes the order inside a bucket so the
  step is reproducible (see above).
- **The device asks for the adapter's limits,** because 3M birds is 336 MB in
  one storage buffer, past the WebGPU defaults. Bench IC is synthetic (a
  settled-density cube) so the 3M timing is not of a state the model never
  reaches.

**Open**

- `decide` dominates the benchmark: the 4 m cells put ~250 candidates in a
  1.5 m radius query. Phase 9 will size the cell and the bucket table against
  the benchmark, and may reorder the records so the query reads them
  contiguously.
- The grid reuses two buffers for two roles to stay at the eight-storage-buffer
  limit; if phase 9 wants more grid state it will need a different packing.
- `nn_ab` and the shape metrics remain noisy at 2,000 birds, as phase 2 found.

## Phase 4 — full flight model (2026-10-15)

**Audit: the model work was already phase 2.** Before writing code, each of the
plan's phase-4 bullets was checked function by function against `murmur-ref`
(the specification). Every *behavioural* item — boundary-weighted cohesion,
banked turning, cruise speed, preferred altitude, roost attraction and committed
returns, and the seeded per-bird random force — is already in `shaders/sim.wgsl`,
ported in phase 2 and passing parity; the table of correspondences is in
`PLAN.md` phase 4. The `social::decide` pieces that are absent (flee, flash
expansion, the alarm copying rule) are the falcon-dependent ones that are inert
with no falcon, which is this phase's state; they belong to phase 7 and change
nothing here. So the genuinely new work was `--analyze` and the acceptance run.

**Built**

- `crates/murmur-gpu/src/analyze.rs`: `--analyze`, the GPU flock's per-second
  statistics in the reference `--analyze` format (same `Stats::line`, same
  summary), with an optional `--extended` line. The GPU is stepped and read back
  once per simulated second; the *CPU* analysis functions then measure it, so the
  output can be read against the recorded oracle.
- `crates/murmur-gpu/src/coherence.rs`: the criteria checker (13 checks) and the
  light coherence sampler + run behind `--coherence SECONDS`.
- `crates/murmur-ref/src/analysis.rs`: `cluster_fraction`, the public entry to
  the same `largest_cluster` `compute` uses, so the light sampler measures the
  exact metric.
- `app/src/main.rs`: `--analyze SECONDS`, `--coherence SECONDS`, `--extended`,
  `--turning base|inertial`.
- `crates/murmur-gpu/tests/analyze.rs`: the light sampler is asserted
  bit-identical to `analysis::compute` field by field (CPU-only), and both GPU
  runs produce one sample per whole second.

**Measured**

- `--analyze 30 --birds 100000 --seed 1` (full metric set): **real 15m11.982s**
  on the GTX 1650. Roughly 6000 steps × ~24 ms ≈ 144 s of GPU stepping, so the
  other **~26 s per simulated second is `analysis::compute`** — its 15 m
  nearest-neighbour scan is almost all-pairs in a dense flock. Extrapolating,
  `--analyze 600` would be ~5 h, nearly all of it statistics the criteria never
  read.
- `--coherence 1` = 15.5 s, `--coherence 5` = 34.9 s → **~4.9 s per simulated
  second** (≈24 ms/step, GPU-bound), so the 600 s run is **~50 minutes**.
- The light sampler's per-second fields are **bit-identical** to the full
  metric set (`tests/analyze.rs`), and on the GPU the `--coherence` t=1…5 lines
  reproduce the `--analyze` t=1…5 lines exactly (`t=1 cluster=100.0%
  spread=41.1m Rh=195.8m dz=9.6m hstd=11.4m rc=130.5m`), so the light run samples
  the same trajectory.

**Found and fixed (feasibility, not the model)**

- **The initial cloud was ~50× too dense at 100,000 birds.** The model's fixed
  18 m cloud holds fifty times the birds at 100k, so the flock started as a
  near-solid ball; the first second took over 50 s and an early attempt lost the
  device (wgpu `Validation Error … Parent device is lost`). `World::new_density_scaled`
  grows the radius as `n^(1/3)` to hold the density; the draw order is unchanged,
  so `World::new` (the oracle's IC) stays byte-for-byte and the algorithm still
  passes.
- **The GPU grid cell was tied to the oracle's `GRID_CELL = 4.0 m`.** The cell
  is an acceleration knob (the exact-distance check makes the neighbour sets
  identical), and 4 m is far too coarse for the model's sub-metre radius at 100k.
  `grid::CELL = 1.0 m`; see `docs/model.md`.
- **Submissions are batched at ten steps, not one command buffer per second,** so
  a single slow `decide` cannot outlive the GPU watchdog while sampling stays per
  second.

**PLAN changes recorded** (also in `PLAN.md` phase 4)

- Item 2 had the run go through `--analyze`; the measured ~26 s/sample cost made
  that ~5 h, so the run uses `--coherence SECONDS`, a lighter sampler that
  computes exactly the criteria fields (proved bit-identical). `--analyze` keeps
  the full reference output.
- The scaled initial cloud (`World::new_density_scaled`) and the GPU's 1 m cell
  (`grid::CELL`) are recorded above and in `docs/model.md`.
- The plan's phase 3 text called the cell "a free performance knob"; phase 4 is
  where it stopped being free and was sized.

**Open**

- The 600 s coherence verdict is in `docs/coherence-100k.md` once the run lands;
  what it tests and the recorded result are there.
- The GPU's random force is statistically, not bit-, identical to the CPU's
  (phase 2 adaptation), so the 100k run is the GPU's own trajectory and is judged
  by statistics, as the criteria say.
- Elongation (`I3/I1`) climbs during the run, as it does in the reference; it is
  reported, never used to fail.

## Phase 4 — the 100,000-bird run, second attempt (2026-10-16)

**The 2026-10-15 claim was premature.** `PLAN.md` phase 4 said the acceptance
was "Met (2026-10-15): recorded in `docs/progress.md` and
`docs/analyze-100k-600s.txt`". The file it pointed at holds only the run's
*header* — the run hung after ~1 h 20 min and was killed before it printed a
single second, so nothing in it supports the claim. `PLAN.md` has been corrected
to say this phase is not met until the verdict below is written.

**Found — the run is hours, not one hour.** The earlier "~4.9 s per simulated
second … so 600 s is ~50 minutes" was measured over `--coherence 5`, i.e. the
first seconds only. The dominant pass is `decide`, which walks every grid cell
the bird's *adaptive* radius `R` covers, so its cost follows the local density —
and the settled flock is several times denser than the deliberately sparse
initial cloud. Measured on the GTX 1650 with the light sampler:

| `t` \[s\] | wall per simulated second |
| --------- | ------------------------- |
| 20        | ~6 s                      |
| 70        | ~13 s                     |
| 123       | ~26 s                     |

Still rising at `t = 123`. So the 600 s turn is **hours** of GPU time, and it is
GPU-bound: `decide` dominates, not the sampler, so CPU-side parallelism would not
help. The corrected estimate replaces the old one in `docs/coherence-100k.md`,
and the run header records the wall clock it actually took.

**Found — the NVIDIA driver hangs on a run this long.** Twice, in the kernel log,
on this workload and nothing else:

```
NVRM: Xid (PCI:0000:03:00): 109, pid=7595,  name=murm-accept, channel 0x0000000e, errorString CTX SWITCH TIMEOUT
NVRM: Xid (PCI:0000:03:00): 109, pid=19230, name=murm-accept, channel 0x00000008, errorString CTX SWITCH TIMEOUT
```

`Xid 109` (`CTX SWITCH TIMEOUT`) is a [known Linux NVIDIA driver
bug](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1097) reported
across many applications and hardware generations, not something these shaders
provoke. The first occurrence (09:38) coincided with a dock/USB-C resume, the
second (11:53) with no external event, so it is not suspend handling. After it,
the hung submission's fence is never signalled — and the code then waited with
`poll(wait_indefinitely())`, so the process blocked for ever: the run was lost
with no output and no error, which is exactly why the first attempt's file holds
only a header. Suspending the laptop was ruled out as the cause by rerunning
under a `systemd-inhibit` lock (`sleep:idle:handle-lid-switch`) and hitting the
same hang.

**Fixed — bounded waits and a run that repairs itself.**

- `device::WAIT_BUDGET` (60 s, far longer than the ten-step batch a wait covers)
  and `Gpu::wait_bounded`: a stalled submission is reported as
  `device::Stalled` instead of blocking for ever. Every wait on a path a long run
  takes is bounded, the readback included.
- `sim::RawState` + `Sim::raw_state` / `Sim::from_raw`: the flock as the shader
  holds it, *including each bird's own random stream* — which the CPU-facing
  `Snapshot` deliberately does not keep. A flock rebuilt from it continues the
  same trajectory, because a step is a pure function of the records and the
  parameters and the grid is derived from the positions.
- `analyze::each_second_resilient`: on a stall, reopen the device with the same
  request (`device::DeviceRequest`, recorded when the device was opened, retried
  with backoff while the driver retires the dead context) and rebuild the flock
  from the last sampled second. Each repair prints `# resume at t=…` and is
  counted in the run's header, so the record shows what happened.
- `tests/analyze.rs::a_resumed_run_takes_the_same_steps`: stepping 20 steps in one
  go and stepping 7, reading back, rebuilding and stepping 13 more leave
  byte-identical bird records. This is the claim the acceptance depends on.
- `analyze::run` / `coherence::run` / `coherence::run_streaming` now take
  `&mut Gpu` and return `Result<_, RunError>` (a stall that cannot be repaired is
  `RunError::Reopen` or `RunError::NoProgress`), and the app prints the repair
  lines live.

**Environment workaround (needs `sudo`, user-run).** The dGPU has runtime power
management enabled (`/sys/bus/pci/devices/0000:03:00.0/power/control = auto`) and
the run idles the GPU for seconds at a time while the CPU samples, so the driver
may power the card down/up hundreds of times. Setting `power/control = on` for
the session (reversible, resets on reboot) is the cheap experiment to see whether
it removes the hang; the code-side repair above is what makes the run safe if it
does not.

**Found — the wall was not the driver but the query's cubic cost (third
attempt).** The repaired run did not hang once in ~1 h 7 min (no `Xid 109` for
its pid) — but it stopped advancing at `t = 153`: `# resume` at `t = 153.x`, then
samples 0.05 s apart, ~60 s per batch, indefinitely. Two facts made the cause
clear instead of mysterious:

- Both long attempts stopped in the **same narrow window** (`t ≈ 153–155`,
  `seed 1`), so it is deterministic workload, not a driver timer.
- The GPU sat at 100 % utilisation and 1755 MHz throughout: busy, not wedged.

The busy work was `gather_grid` walking a box that had grown enormous. A bird's
adaptive radius is clamped at `params::R_MAX = 100 m`, the 100,000-bird flock
does grow stragglers (`Rh ≈ 177 m` here), and the walk costs the *cube* of the
radius: at `CELL = 1 m`, `R = 100 m` is `201³ ≈ 8.1e6` cells for one bird — as
much as ~300,000 ordinary birds — with the centrality walk at `2R` worse again.
One straggler therefore dominates the whole step, and when the first wide one
appeared the step time rose ~60× inside a second of simulated time. The
`Xid 109` above is the same effect: a compute kernel that runs for minutes is
what makes the driver's context switch time out.

**Fixed — a bounded neighbour query.** `gather` / `gather_centrality`
(`shaders/sim.wgsl`) fall back to the brute-force walk when the query box covers
more cells than the flock has birds. It is both cheaper and exact — both walks
keep a bird only after the same `d2 <= r2` test — so the neighbour set is
unchanged and the choice is purely a performance one. Cost per query becomes
`min(cells, n)` instead of `cells`. `tests/grid.rs::a_wide_radius_query_matches_bruteforce`
forces the crossover for every bird and checks the two walks agree, and a counter
(`Sim::wide_queries`, an atomic in the clock) proves the fallback really is taken:
2 queries per bird (both walks) in the forced test, and — measured, not assumed —
1 query of 2,000 on the settled base flock, i.e. the same radius tail at a scale
small enough to see.

**Fixed — two bugs in the repair path itself.**

- **The clock did not rewind with the flock.** A repair rebuilt the flock from
  the last sampled second but left `done` at the failed batch's end, so the
  resumed sampler was desynchronised from its flock — which is why the samples
  above came 0.05 s apart instead of a second apart. `each_second_resilient` now
  tracks the step count the raw state belongs to (`sampled`) and rewinds to it.
- **A non-recovering driver looped for ever.** `MAX_RESUME_TRIES = 3` repairs in
  a row with no new sample now end the run with `RunError::NoProgress { time,
  tries }` — an honest failure that says where it stopped, instead of thrashing
  silently. The successful-repair path is covered by the test above; the guard
  itself has none, because forcing a driver hang is not something the suite can
  do, and it is a counter over that same loop.

**Recorded — the mechanism is now in the record.** The per-second line carries
`Rmax` and the fraction of birds with `R > WIDE_RADIUS = 10 m`, so the radius
tail that drives the cost is visible every second rather than inferred from the
wall clock.

**What the fixed run does (measured; the 600 s turn is in progress).** The turn
was restarted at 13:31 with the bound in place. It passed the window both earlier
attempts died in (`t ≈ 153–155`) **without a single repair**: by `t = 166`, with
a wide-radius bird in the flock (`Rmax = 16.4 m`), it was stepping ~0.12 s per
step (a ten-step batch in ~1.2 s) — where the pre-fix run's ten-step batches
there never finished inside the 60 s wait budget. The cost profile is otherwise
the smooth one already tabulated: ~19 s per simulated second at `t = 88`, ~24 s
at `t = 136`.

One limit on this, stated rather than glossed: the fixed run is **not a
controlled A/B** against the stalled ones. The fallback sums a neighbour set in
index order where the grid sums it in cell order, so from the first falling-back
bird the trajectories differ in their last bits and then diverge; the two runs
are no longer the same flock at `t = 153`. What *is* established is the mechanism
— a tail of wide-radius birds is real and measured (`Rmax ≈ 17 m`, `Rh ≈ 190 m`
by `t = 40`, against `params::R_MAX = 100 m`) and the radius-box cell walk is the
only superlinear term in a step (the grid build, `resolve` and `integrate` are
all O(n) per step) — and that the bound caps any single query at `n`, so that
term cannot blow up again.

**Verdict (100,000 birds, seed 1, 600 s): met, 2026-10-16.** The turn was
restarted at 13:31 with the bounded query and finished at 16:00 — **2 h 29 min**
for 600 s of simulation (~15 s per simulated second on average), with **zero
repairs and no `Xid`**. All 13 pre-declared criteria pass over all 600 seconds
(see `docs/coherence-100k.md`, "Verdict", for the table). The flock neither
collapsed nor dispersed: `cluster` never fell below 98.3 %, settled `|N|` 6.68,
`R` 0.85 m, `v` 10.0 m/s, `spread` 21.5 m, `rc` 100 m.

- **The series was re-derived independently.** Minima, maxima and the settled
  means were recomputed with `awk` from the 600 printed per-second lines rather
  than read off the checker's table, and they agree on every criterion. (The
  first attempt at this re-derivation produced nonsense — 0 % cluster — because
  the line's `{:4.1}` padding makes the field count vary; parsing by field name
  fixed it. Recorded because "verify the checker" only means something if the
  verification is itself trustworthy.)
- **The radius tail is in the record and peaks exactly at the old wall.**
  `Rmax` spikes to **55.9 m at t = 152** and recedes; the two stalled attempts
  died at `t ≈ 153–155`. 128 of 600 seconds hold a bird wide enough to cross the
  centrality walk's crossover (`R > 11.35 m` at `n = 100000`), 31 of 600 wide
  enough to cross both walks (`R > 22.7 m`). At the old code's 1 m cell, one
  `R = 55.9 m` bird means `112³ ≈ 1.4e6` cells for the interaction walk and
  `224³ ≈ 1.1e7` for centrality, every cell a scattered lookup into a
  multi-megabyte offset table — which is why it stopped being a slow-down and
  became a wall.
- The series and verdict block are in `docs/analyze-100k-600s.txt`, verbatim.
- The app now prints `# wall` with the elapsed clock (added after this run; its
  2 h 29 min was measured from the launch to the file's last write).

**Open**

- The `# wall` line is new and untested against a real long run; the next long
  run should be checked for it.
- The runtime-PM experiment (`power/control = on`) was never applied — the run
  finished without it and without a single hang, so it is no longer needed for
  the acceptance. It stays recorded as an option if hangs return.

## Phase 5 — first render: sky and coverage (2026-10-16)

**Built**

- `crates/murmur-render/src/camera.rs`: a CPU port of the reference
  `render/camera.rs` — the ground observer that follows the flock, derives its
  extent and frames it. It is fed by a full GPU→CPU readback of the bird records
  each frame for now; phase 9 removes that readback.
- `crates/murmur-render/src/sky.rs`, `shaders/sky.wgsl`, `shaders/sky_pass.wgsl`:
  the dusk sky — phase 0's palette with the reference's time-of-evening
  interpolation, drawn as a full-screen pass from the view ray.
- `crates/murmur-render/src/renderer.rs`, `shaders/scene.wgsl`,
  `shaders/coverage.wgsl`, `shaders/exposure.wgsl`, `shaders/tonemap.wgsl`: the
  HDR pipeline. One instanced quad per bird deposits coverage with
  `dst *= 1 - coverage` and `coverage = 1 - exp(-exposure·raw)`, so the product
  over the flock is `exp(-exposure·tau)` — the reference's optical depth — in a
  single pass. A 256-bin histogram pass locates the 80th percentile of the
  flock's optical depth and eases the exposure toward `TARGET_TAU / tau80`; a
  tone-map pass writes the output. The deviations from the reference are listed
  in `docs/model.md` ("Phase 5 adaptations").
- `crates/murmur-render/src/png.rs`: a dependency-free PNG writer (stored
  DEFLATE blocks — the `image` crate is not available offline), ~2.7 MB per
  1280×720 frame.
- `crates/murmur-gpu/shaders/bird.wgsl` as `murmur_gpu::BIRD_WGSL`: the `Bird`
  struct and the flight helpers lifted out of `sim.wgsl`, so the coverage pass
  derives the wing exposure from the *same* body frame the simulation flies.
  `sim.rs` and `grid.rs` prepend it; `tests/layout.rs` checks against it.
- `--still PATH` (headless), and the window now renders through this crate.
- Tests: `tests/scene_layout.rs` (`Scene` against `scene.wgsl`, field by field),
  `tests/sky_palette.rs` (the Rust palette against the shader's stops for day,
  dusk and night).

**Measured** (GTX 1650 Max-Q, driver 615.78.08, Vulkan, 1280×720, seed 1)

| still | birds | simulated | wall clock | wall per simulated s |
| ----- | ----: | --------: | ---------: | -------------------: |
| `docs/still-20k.png` | 20,000 | 60 s | 30 s | 0.51 s |
| `docs/still-1m.png` | 1,000,000 | 60 s | 3,344 s (56 min) | 55.7 s |
| *(discarded)* first 1M capture | 1,000,000 | 20 s | 841 s (14 min) | 42.0 s |

- 60 s is 12,000 steps (`DT = 0.005`). The settled 1M step costs **0.279 s**,
  1.7× `--bench`'s 163 ms fresh-flock step — the same "a settled flock is
  several times denser than the fresh one" effect phase 3 recorded, on a bigger
  flock. The 20k still's step is 2.5 ms, so 50× the birds cost 111× per step
  here. Part of that is the density; part is that the 1M transients hold more
  wide-radius birds and the bounded query then falls back to a full
  million-bird walk (the crossover at 1M is `R > 24.75 m` for the centrality
  query against `R > 11.35 m` at 100k). **Not isolated**: one frame is the
  acceptance, and the frame is not what costs.
- Both frames report `# exposure 1.000`; the sky-only frame (no birds, not kept)
  reports `25.0`, the reference's `DEFAULT_EXPOSURE`. `1.000` is the port's
  floor: the reference's rule drives the 80th percentile of the flock's optical
  depth to `TARGET_TAU = 1.204`, a settled flock at 20k already exceeds that at
  exposure 1, so the rule asks to brighten and cannot (`EXPOSURE_MIN = 1.0`). The
  port matches the reference's rule line for line, so this is the rule's own
  behaviour at density — and it is why the core reads as saturated black, which
  is the "dark" the acceptance asks for. Recorded, not tuned: widening the range
  would be a model change.
- The committed PNGs are **lossless re-encodings** of the app's output (whose
  writer stores uncompressed blocks). Pixels verified identical, total absolute
  difference 0; the 1M frame is 754 kB instead of 2.7 MB.

**Acceptance — met (2026-10-16).** `docs/still-1m.png` is a million birds after
60 s: a dense black mass with lighter speckled edges billowing over the roost
against the dusk sky, wing-exposure structure in the thin parts and a wavy
underside where the flock's lower edge meets the sky.

`docs/still-20k.png` is the same frame time and the same material at a fiftieth
of the birds, and it is worth keeping precisely because it does *not* look the
same: by 60 s the 20,000-bird flock has stretched into a ribbon of separated
birds, while the million-bird flock is still a mass. That is the known long-run
elongation showing up in a still, not a rendering difference — the darkness, the
speckled edges and the sky read identically in both.

Two honest caveats on "similar in feel":

- The comparison is a judgement, not a measurement. The terminal reference draws
  braille cells, so the two cannot be compared pixel-wise; what is claimed is
  tone, shape and edge structure, not image equality.
- The first 1M capture (20 s above) was a long transient **tube**, not a settled
  flock: the density-scaled initial cloud at 1M is ~143 m across, comparable to
  the 150 m roost itself, so the flock takes longer to relax. That is why the
  acceptance still simulates 60 s. Shape over long runs is a known gap
  (`docs/model.md`, "Known gaps"), not something this phase changed.

**Found and fixed: `advance_seconds` queued the whole interval and waited once.**
The first 1M still died at `Sim::snapshot` with `Stalled { waited: 60s }` after
4 s of simulation, and it looked exactly like the phase-4 driver hang. It was
not: the GPU was healthy (100 %, 1665 MHz), but all 800 steps (~130 s of work)
had been submitted and none drained, so the *single* bounded wait at the end
timed out against `device::WAIT_BUDGET`. `advance_seconds` now waits after every
10-step batch; every wait is then as short as the ones a resilient run makes, and
a real hang surfaces as a reported `Stalled` instead of a freeze.

**Open**

- `--still` prints nothing until it finishes, and a 56-minute run should stream
  progress the way `--coherence` does. Left for phase 6, which reworks this path
  for bird LODs, so that the committed code stays the code that produced the
  accepted still.
- The camera readback and the exposure-histogram readback both stall the pipeline
  every frame; phase 9 removes both.
- The reference's separate distance-opacity fade buffer is still not implemented
  (the distance term lives inside the coverage); tone mapping is the identity map
  for now.

**Committed together with phase 4.** Both phases land in one commit. Phase 5
lifted `Bird` and the flight helpers out of `sim.wgsl` into `bird.wgsl`, a file
phase 4's bounded-query edit also touches, and `app/src/main.rs` carries both
phases' flags and the window path. Splitting them would mean reconstructing a
phase-4 `sim.wgsl` and a phase-4 `main.rs` that never existed, and committing an
intermediate that does not build; the phase boundary is kept in this log and in
`PLAN.md` instead. Recorded because the earlier commits are one per phase, and
this one is not.


## Phase 6 — birds: mesh, LOD and temporal AA (2026-10-16)

The flock gets bodies. Three representations of a bird, chosen by its projected
wingspan `ρ` in one compute pass, and a temporal-AA resolve so that sub-pixel
birds stop sparkling:

| level | when | drawn as |
| --- | --- | --- |
| mesh | `ρ ≥ 7 px` | ten-triangle starling silhouette, instanced |
| impostor | `1.5 ≤ ρ < 7 px` | screen-aligned quad carrying a bird-shaped Gaussian mixture |
| splat | `ρ < 1.5 px` | the phase-5 coverage splat, unchanged |

The mesh↔impostor threshold is a **cross-fade**: a bird within `BAND = 1 px` of
`RHO_MESH = 6 px` is written to both lists with weights summing to one. Every
level carries the same optical **mass** `a·ρ²`, so crossing a threshold changes a
bird's *shape*, never how much light it absorbs. All the new shaders and the
classifier live in `crates/murmur-render/shaders/` (`lod.wgsl`, `bird_pass.wgsl`
shared setup, `bird_mesh.wgsl`, `impostor.wgsl`, `taa.wgsl`, and the rewritten
`coverage.wgsl`); the geometry is `crates/murmur-render/src/mesh.rs`. The
deviations from the reference are in `docs/model.md` ("Phase 6 adaptations").

### How the LOD is built

- **Classifier → indirect draws, no collect pass.** `lod.wgsl` runs one
  invocation per bird, classifies it, and `atomicAdd`s into
  `lod_args[level*4 + 1]` — the indirect `DrawIndirect` block's *instance count*.
  The same atomic hands out the slot in `lod_lists[level*n + slot]`, so the count
  the draw reads is the count the classifier wrote. There is no separate counts
  buffer and no finalise pass. `lod_args` is `array<atomic<u32>>`; its vertex
  counts (30/6/6) are rewritten from a 48-byte CPU template each frame, which
  also resets the counts to zero.
- **The mesh is mass-preserving.** `MEAN_AREA = 0.0851` is the mesh's mean
  projected silhouette area in units of `ρ²`, measured from the vertex data over
  viewing angles and wingbeat phases (`mesh::mean_projected_area`). The fragment
  shader deposits a flat `a·weight / MEAN_AREA`, so on average a mesh bird
  absorbs `a·ρ²` — the splat's mass. A test pins the measured number against the
  shader's constant.
- **The impostor is mass-preserving by construction.** A body plus two wing
  lobes as normalised Gaussians, divided by the sum of their own integrals. The
  wing lobes shorten by `cos(flap)`, so its silhouette sweeps through the
  wingbeat like the mesh's.
- **Wingbeat from the model.** The dihedral is `FLAP_DIHEDRAL ·
  flap_amp · sin(flap_phase)` at the model's `FLAP_HZ = 10`; only the peak angle
  `FLAP_DIHEDRAL = 0.7 rad` is the renderer's own constant, because the reference
  draws dots and has no wings to rotate.
- **No depth buffer, `cull_mode: None`.** The scene is optically thin; the
  levels composite in draw order. Culling back faces would hide a wing's
  underside seen from below, which is most of the flock.
- **Chosen thresholds.** `RHO_SPLAT = 1.5` is a hard switch (both shapes are
  under two pixels there); `RHO_MESH = 6` with `1 px` of cross-fade is where the
  impostor stops resolving wing detail. `tests/lod_constants.rs` parses the
  shader's copy of every threshold, the workgroup size and each pass's `LEVEL`
  out of the WGSL and checks them against the Rust constants, so a bird can never
  be classified into the wrong draw.

### Temporal AA

Each frame the bird passes are offset by a sub-pixel Halton(2,3) sample carried
in `Scene.jitter`, and `taa.wgsl` blends the fresh HDR frame into a running image
(`alpha = 1 - exp(-dt/TAA_TAU)`, `TAA_TAU = 0.1 s`); the tone map reads the
accumulated target. There are no motion vectors, so the history is clamped into
the current frame's 3×3 neighbourhood range before it is mixed — the standard
stale-sample defence. `--still` drives the accumulation directly
(`history_weight = 1/(frame+1)`) so a still converges to a box-filtered,
24-sample image; the window keeps the exponential.

### Acceptance — measured (2026-10-16)

**No popping between levels — met.** `docs/lod-sweep.txt` is the controlled
measurement: one bird held at a fixed screen point and pose while its projected
wingspan is swept from `0.25` to `12 px` in `0.25 px` steps, with the exposure
pinned and both the accumulation and the jitter off, so a step measures the
level change and nothing else. "Change" and "ink" are in *black pixels* (one
fully dark pixel of the frame = 1), which is the unit the flock is judged in.

- The mesh cross-fade `[5, 7] px`: the worst step is **1.01 black px**, against a
  **0.92** control — the worst step of the *same bird growing by the same amount*
  away from the threshold. Ratio **1.10×**, under the stated `2×` jump
  threshold. The bird grows by `2.9×` in that band; the picture changes by no
  more than growth itself does.
- The hard splat→impostor switch at `1.5 px`: **0.13** vs a **0.25** control.
- `docs/flypast.mp4` is the recorded fly-past (960×540, `--birds 20000`, lens
  zooming `1.00 → 0.35`). Its LOD counts run from `mesh=0 impostor=20000
  splat=0` at the first frame to `mesh=20000 impostor=0 splat=0` at the last: in
  six seconds **every bird in the flock crosses the mesh threshold** and the
  frame-to-frame change shows no spike (`docs/capture-20k.txt`).

The one real change the cross-fade carries is in the *ink*: the mesh concentrates
`a·ρ²` through a silhouette, so at equal mass it is darker per pixel but covers
far less area, and a bird resolving gets thinner. Across the `[5,7] px` band the
ink moves `3.20 → 2.30` black px (**1.39×**) over eight `0.25 px` steps, i.e.
gradually, not as a step. That is the documented physics-vs-dot difference
(`docs/model.md`), not hidden here. It also does **not** show up in the flock's
total darkening: the 20,000-bird still is `150.00` mean luminance in phase 5 and
`148.96` in phase 6 (**0.7 %** darker), and the 1M still is `120.22` vs `119.90`
(**0.3 %**). The level a bird is drawn at does not change how dark the flock is.

**No obvious shimmer in a static distant flock — met.** `docs/shimmer.mp4` shows
the same frozen flock four times wider (birds sub-pixel, all `splat`) with the
accumulation on and off, side by side. Over 30 frames the raw frame-to-frame
change is **293.8 black px/frame**; with the accumulation it is **68.5** — **77 %
removed**. The in-motion fly-past is close behind: **28866 → 7126**, **75 %
removed**. So the accumulation removes three-quarters of the sub-pixel change
whether or not the flock is moving. (Phase 7 re-measured the fly-past after the
mesh correction below: the accumulated frame-to-frame change is `12426 → 4823`
black px, 61 % removed; the static figure is unchanged at 77 %.)

**Frame time at 1M birds — reported.** All numbers are GPU timestamp queries
(`renderer.frame_times_ms`), on a **GTX 1650 with Max-Q Design, NVIDIA driver
615.78.08, Vulkan**, `1280×720`:

- **1M birds: `12.19 ms/frame`** (`lod 0.43`, `birds 11.21`, `exposure 0.17`,
  `taa 0.32`, `tonemap 0.06`), at `mesh=190 impostor=554516 splat=445370` — a
  settled 1M flock is sub-pixel almost everywhere, so the cheap splat path does
  the work. `docs/still-1m-p6.png` is that frame; the simulation to reach it cost
  `3484 s` (`58.06 s` per simulated second), the phase-5 cost carried over.
- **20,000 birds: `0.84 ms/frame`** (`birds 0.46`), all impostor
  (`docs/still-20k-p6.png`).
- **2,000 birds: `0.47 ms/frame`** (`birds 0.10`), all mesh
  (`docs/still-2k-mesh.png`) — the mesh at work, wings and all.

The timestamp deltas are **GPU** time; the wall-clock `--capture` cost is much
higher because the still/capture paths read the frame back to the CPU every
frame. Those readbacks (camera and exposure histogram) are the phase-9 target.

### Checks

`cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test` all
pass (`murmur-render`: 12 unit + `mesh_area` 3 + `lod_constants` 4 +
`scene_layout` 1 + palettes/sky/png/camera; no GPU-only tests are skipped).

### Open

- **Temporal AA has no motion vectors.** The 3×3 neighbourhood clamp is the
  whole stale-sample defence; a fast camera move will ghost. Motion-vector
  reprojection needs a depth buffer and is left for later.
- `--still` still prints nothing until it finishes; a 58-minute run should
  stream progress the way `--coherence` does.
- The camera readback and the exposure-histogram readback both stall the pipeline
  every frame; phase 9 removes both.
- The impostor's sigmas, the mesh's ten triangles and the peak dihedral are the
  renderer's own shapes, fixed once and measured by `--lod-sweep`, not tuned per
  scene.

## Phase 7 — falcon and agitation (2026-10-17)

The falcon hunts, the flock flees and copies zigzags, and the agitation-wave
experiment runs on both engines.

### What was built

- **The CPU oracle.** `murmur-ref::escape` gains `measure_agitation`,
  `AgitationWave`, `spread` and `linfit`, ported from the terminal version's
  `sim/escape.rs`. `crates/murmur-ref/tests/agitation.rs` pins it to the
  terminal's own runs: `--agitation --birds 2000 --seed 42 --zigzag-k 2` reports
  `speed 7.56 m/s, R2 0.952, 100 % reached, 22 % expansion` here, digit for digit
  the same line the terminal prints.
- **The GPU falcon.** `crates/murmur-gpu/src/falcon.rs` (the record and its
  spawn) and `shaders/falcon.wgsl` (a one-workgroup reduction of the flock to its
  centroid and extent, and a one-invocation `Falcon::update` mirror), dispatched
  inside the step when a falcon is present. The alarm, the flee force and the
  flash expansion live in `sim.wgsl`'s decide pass, on every bird every step.
  `Sim` gains `spawn_falcon`, `steer_falcon` (mouse mode), `write_falcon` and
  `falcon` (a small readback); `--falcon` runs only those two extra passes, so a
  run without a falcon is byte-for-byte the phase-6 step.
- **The GPU experiment.** `murmur-gpu::agitation::measure` runs the oracle's
  seeding and fit over a GPU flock, reading the escape state back once per 5 ms
  step (the experiment needs each bird's *first* zigzag start, which the shader
  does not keep). The settle is batched.
- **The renderer.** The falcon is drawn as the reference draws it
  (`render/splat.rs`): `a = 1.6·atmosphere`, a `1 m` wingspan, the same
  continuous unresolved/resolved footprint as a splat, so it cannot shimmer. Its
  position rides in the `Scene`; the camera gains `cursor_to_plane` for the mouse.
- **The app.** `--falcon`, `--agitation`, `--zigzag-k` (1–7), `--zoom`; the
  window's `f` key spawns and moving the mouse steers, as the terminal does.

### Acceptance — measured

**Agitation propagation speed — met.** `--agitation` under the GPU and the CPU
oracle, GTX 1650 Max-Q, driver 615.78.08, Vulkan, 2,000 birds, 4 s:

| k | seed | CPU oracle | GPU | agreement |
| --- | --- | --- | --- | --- |
| 2 | 42 | 7.56 m/s | 7.52 | 1 % |
| 2 | 1 | 6.85 | 6.80 | 1 % |
| 2 | 7 | 6.81 | 7.10 | 4 % |
| 1 | 42 | 15.15 | 14.66 | 3 % |
| 3 | 42 | 4.01 | 3.97 | 1 % |

The terminal version reports 6.8–7.6 m/s at `k = 2` and this bird count, so the
GPU's `k = 2` runs are inside the reference range, and every run is inside the
3–25 m/s band the terminal prints as its target. The GPU's R² is 0.9 or better
and the wave reaches the whole flock, as the oracle's does. The two engines have
different random streams (below), so this compares the *rule* and the resulting
speed, not a trajectory.

**A falcon attack — the alarm and the wave are met; the dark band's *visual*
contrast is weak.** `crates/murmur-gpu/tests/falcon.rs` checks, on the GPU:

- an autonomous falcon flies the full patrol → climb → stoop → retreat cycle
  (all four states observed) and alarms birds (`peak alarmed 100 %`, `peak
  zigzagging 93 %` over 30 s at 2,000 birds);
- the alarm rule matches the CPU oracle **bird for bird** on a frozen flock:
  `alarmed gpu 2000 / cpu 2000`, `flash gpu 874 / cpu 874`, and the roll direction
  agrees on `2000/2000` alarmed birds;
- the wave sweeps the flock over time — the zigzag share rises from 0 to ~41 %
  and decays over about 3 s (`0 %…8 s:13 %, 10 s:41 %, 11 s:18 %, 12 s:0 %`).

`docs/falcon-attack.mp4` records an attack (5,000 birds, seed 2, the lens fixed):
the frame's mean brightness dips from `111` to `104` and recovers to `113` across
the attack window — a real darkening of the flock. The recording shows the falcon
as a dark speck and the flock thinning where it passes (the flash expansion).

Honest caveat: the *band* is far more visible in the terminal version than here.
The terminal draws a bird as a dot whose per-pixel optical depth is ~`0.6a`, so a
roll changes a pixel's darkness directly. In this renderer a dense flock is
optically thick (several birds per pixel), so the same roll changes pixels that
are already near black; the contrast is much lower. It is measurable (the
brightness dip above) and visible in the thin parts of the flock, but it is not
the bold band the terminal shows. This is recorded as open, not fixed.

### Phase 6 corrections this phase forced

Driving the falcon surfaced a mistake in phase 6, and it is fixed here rather
than papered over (`docs/model.md`, "Phase 6 correction found by phase 7"):

- **The mesh's optical depth is the reference disc's peak, `2a/π`**, not the
  mass-preserving `a/MEAN_AREA`. The old form made every mesh bird opaque
  (`τ ≈ 12a`), which pinned the auto-exposure at its `EXPOSURE_MIN = 1.0` floor
  and saturated the flock to black — hiding exactly the wing-exposure structure
  the falcon's wave moves. With the disc peak the exposure behaves as the model's
  rule intends (at 2,000 birds it settles near 5, not 1) and a mesh bird is as
  dark per pixel as a dot bird of the same size.
- **The mesh↔impostor cross-fade widened from `1 px` to `2 px`.** With the
  correct per-pixel depth the ink difference between the levels is larger, and a
  `1 px` band let the fade's worst per-step change reach `2.30×` the growth
  control — a jump by the sweep's own criterion. At `2 px` it is `1.52×`
  (`docs/lod-sweep.txt`), and the ink moves `2.01 → 0.53` black px over sixteen
  steps.

`docs/lod-sweep.txt`, `docs/still-2k-mesh.png`, `docs/still-20k-p6.png` and
`docs/capture-20k.txt` are re-measured under the corrected code — the wider
cross-fade band brings a `ρ ≥ 4 px` fringe of the 20,000-bird flock into the
mesh level (a few per cent of the birds, counted in both lists). GPU run-to-run
bit-reproducibility is out of scope (PLAN.md), and at 20,000 birds over 60 s
chaos amplifies the last-bit differences the GPU's own sum order introduces, so
the exact fringe varies from run to run; the numbers quoted are one run.
`docs/still-1m-p6.png` is re-rendered under the corrected code for the phase-7
frame time (its `mesh` count is ~270 of a million).

### Frame time re-measured

GPU timestamp queries, GTX 1650 Max-Q, driver 615.78.08, Vulkan, `1280×720`, one
run each, the GPU idle:

- **1,000,000 birds: `11.02 ms/frame`** (`lod 0.43`, `birds 10.03`,
  `exposure 0.18`, `taa 0.32`, `tonemap 0.06`). The still that reached it cost
  `3397 s` of simulation. Phase 6 measured `12.19 ms` on the same setup; the
  difference is run-to-run variation, not the mesh correction (at 1M only ~270
  birds are meshes).
- **20,000 birds: `0.87 ms/frame`** (`birds 0.46`), mostly impostor.
- **2,000 birds: `0.49 ms/frame`** (`birds 0.12`), all mesh.

### Open

- The dark band's contrast in a dense flock (above).
- Temporal AA still has no motion vectors (phase 6).
- The camera and exposure readbacks still stall the pipeline each frame; the
  agitation measurement adds one readback per 5 ms step, which is fine for an
  experiment and not for a run (phase 9).

## Phase 8 — the evening (2026-10-17)

The flock arrives, displays, roosts and is cleared by night, and the recording is
made from the running app.

### What was built

- **The GPU evening arc.** `crates/murmur-gpu/src/arc.rs` (the state record, its
  spawn and its stream) and `shaders/arc.wgsl` (one invocation advances the
  evening and spawns an arriving group; one pass per bird retires landed birds
  and clears the flock at night). The arc runs after the clock and before the
  falcon, as `world.rs::step` does, and only when `--no-arc` is absent. The
  flock grows and shrinks by **activation**: an arriving group fills the next
  free slots, a landed bird has its `live` flag cleared in place, and an inactive
  bird is skipped by the neighbour walk, the flight integration, the falcon's
  alarm and the LOD classifier. That is the one structural change the fixed-size
  buffer forces (the reference removes records); it is recorded in
  `docs/model.md`.
- **The evening's light and clock.** `z0` moves with the phase (from the arc
  state the shader reads), and the sky's `evening` value comes from the arc's own
  ramp, so the gold arrival, the dusk display and the night all follow the model.
- **Foreground and stars.** The reed bed is a procedural band of swaying stems
  along the bottom of the frame, gated by the reference's own
  `sky::reed_visibility`; stars appear in the upper half once the evening is past
  0.75, from the reference's hash. Both are in
  `crates/murmur-render/shaders/{reeds,sky_pass}.wgsl`.
- **CLI and recording.** `--no-arc`, `--arc-minutes` (the terminal's flags) and
  `--record DIR`, which renders one evening as a time-lapse. `scripts/evening.sh`
  runs it and encodes `docs/evening.webp` and `docs/evening.png`, so the README's
  animation is of the real app.

### Acceptance — met

**A full evening runs unattended.** `cargo run -p murmuration-gpu --release --
--record DIR --birds 8000 --arc-minutes 3` runs the whole arc with no
intervention: the flock grows from nothing through the arrival, holds for the
display, pours down as `z0` sinks to the reed tops, and is cleared at night;
the sky and the stars follow. The phase timeline is checked on both engines:

- the CPU oracle (`crates/murmur-ref/tests/arc.rs`), 2,000 birds, a 3-minute
  evening: `t=120s live 2000 spread 36 m z0 100` (display), `t=160s live 2000 z0
  8.4` (roosting), `t=180s live 115 z0 100` (the next evening has begun);
- the GPU (`crates/murmur-gpu/tests/arc.rs`), the same setup: live `0 → 2000`
  through the arrival, held at 2000 through the display, `z0` `100 → 3` through
  the roosting, and `live 0, z0 3` at night.

Two model properties worth recording. The arrival spawns one group of 50–300
birds every 0.5 s, so a *short* evening cannot fill a large flock: the 15 % the
arrival gets of a 2-minute evening (18 s, ~36 groups) reaches about 6,000 birds,
and `--birds 20000 --arc-minutes 2` therefore displays a partly-filled flock.
That is the model's own rate, not a bug; the recording uses 8,000 birds over 3
minutes, which the arrival fills. And the roosting phase's `remove_landed`
retires a bird only below `REED_TOP = 3 m`, which the descent reaches exactly at
the night boundary, so the flock is cleared at night rather than thinned during
the roosting. That is the reference's own behaviour at `Z0 = 100`, `REED_TOP =
3`; the tests assert the descent and the clear, and say so.

**The README's animation is recorded from the app.** `docs/evening.webp` is the
encode of `--record`'s frames (`scripts/evening.sh` reproduces it), and
`docs/evening.png` is a still from the same run. Nothing in the recording is
authored separately from the binary.

### Open

- The record mode reads the flock back every frame to drive the camera, and the
  arc's spawns are per step; neither is a cost the window can carry at large bird
  counts (the same readback issue as phase 9).
- The reed bed is procedural rather than the terminal's braille; it is a
  silhouette band, not a port of the terminal's `draw_reeds`.
- GPU run-to-run bit-reproducibility remains out of scope (PLAN.md); a recorded
  evening is reproducible on one engine.

## Visual revision / phase 9 — continuous arrival and real-time path (2026-10-17)

User request: render and inspect the app, remove the visibly staged banks of
arriving birds, and improve naturalness and speed. Phases 0–8 already met their
recorded acceptance criteria; PLAN.md explicitly adds this visual revision to
phase 9. Phase 9 is **in progress**, not closed by a nicer still.

### Diagnosis from fresh renders

Rendered an 8,000-bird-target, seed-1, one-minute evening before changing the model
(`--record`, 180 frames, 960×540). A contact sheet confirms the complaint:
separate small clouds become a chain of disconnected lobes, and the chain
persists into display. The old arrival only fills **3,212 of the requested 8,000**
before that short evening's display begins. The old 50–300-bird spawns every 0.5 seconds are visible
in the image. The first continuous-flow experiment instead becomes a connected
banked mass, with a rolling edge and internal density structure. That experiment
also revealed clipping caused by the time-lapse camera's incorrect time step.

### Changes

- Both engines now use a smooth cumulative arrival schedule over at most six
  seconds and a density-scaled, grazing corridor. No force weights, flight speed
  or bank parameters were tuned. The exact initial-condition revision is in
  `docs/model.md`; no-arc oracle baselines are intentionally unchanged.
- The window retains fractional integration time, rather than rounding down
  every frame or forcing one step when less than 5 ms elapsed. Captures retain
  fractional time too. Time-lapse recording uses absolute step targets and the
  actual elapsed simulation interval for camera/exposure easing, and rebinds the
  current ping-pong bird buffer every frame.
- Default 8,000 birds and a 12° minimum lens angle give the arrival more context.
  Exact percentile selection avoids sorting full position arrays.
- The window replaces its full-flock blocking readback with at most 4,096
  asynchronous 32-byte camera samples (128 KiB). It predicts the short telemetry
  delay using velocity; it never changes the simulation. Exposure readback is
  also non-blocking in the window, with only one request in flight. Captures
  retain full-state framing and synchronous exposure.
- Moving the mouse no longer silently spawns a falcon. `f` or `--falcon` must
  enable it first; subsequent mouse motion steers it without a full snapshot.
- Exact query optimisations: compact the birds due a decision into scratch
  storage; reject grid cells whose bounds cannot intersect the query sphere.
  No neighbour cap, reduced decision frequency or relaxed time step was added.
- Two phase-8 regressions found: the benchmark's synthetic birds had not set
  `live=1`, so it could benchmark an empty flock; the final grid offset was still
  buffer capacity, not the live histogram sum, potentially exposing stale sorted
  slots in its last bucket. Both now have regression tests. This does not
  retroactively invalidate measurements made before phase 8 added `live`.

### Measurement and validation

Live-bird baseline, before the query changes: GTX 1650 Max-Q, NVIDIA 615.78.08,
Vulkan, 100,000 birds, 40 timestamped steps after 20 warm-up steps, no render
target: **13.7622 ms/step**, of which `decide` is **12.7786 ms**. Decision
compaction alone: **12.9232 ms/step**, a 6.1% reduction including its new
0.4502 ms preparation pass. These are the synthetic 0.5 birds/m³ scenario, not
settled-flock or window FPS figures.

With conservative sphere/cell rejection as well: **8.9050 ms/step**, `decide`
**7.3806 ms**, a **35.3%** reduction from the live-bird baseline. Same scenario,
same 40 samples; compaction and rejection change neither the accepted neighbour
set nor the force parameters. New measurements at the other planned counts:

| Birds | GPU pass total / step | `decide` | State |
| ---: | ---: | ---: | --- |
| 8,000 | 1.3036 ms | 1.1367 ms | Actual evening at 30 s, seed 1, 100 samples |
| 100,000 | 8.9050 ms | 7.3806 ms | Synthetic scatter, 40 samples |
| 1,000,000 | 129.7379 ms | 112.7949 ms | Synthetic scatter, 40 samples |
| 3,000,000 | 429.2365 ms | 378.4869 ms | Synthetic scatter, 40 samples |

All are GPU timestamp measurements on the adapter/driver above, headless (no
resolution). The actual-evening total excludes the small optional arc passes.
`--bench --birds 8000 --seconds 30 --steps 100` now measures an actual display
instead of the synthetic scatter; without `--seconds`, the original synthetic
scenario remains. The model takes 200 steps per real second, so a step over
5 ms cannot keep real time regardless of render speed. PLAN.md and README now
say the large-count targets are missed and identify `decide` as the bottleneck.
No window FPS is inferred from these pass times.

### Visual evidence

- Regenerated `docs/evening.webp` and `docs/evening.png` from the revised app
  with `scripts/evening.sh` (8k, seed 1, one-minute evening, 180 frames at
  960×540; encoded at 20 fps as a time-lapse). Inspected sampled frames through
  arrival, display, roosting and night. The revised display is a connected,
  turning mass instead of the old chain of disconnected banks. The framing
  correction removes the clipping seen in the first experiment.
- `docs/arrival-comparison.png` shows old/new arrival and display with the same
  requested target, seed and one-minute duration: **3,212 live birds before,
  8,000 after** in the display. It compares the complete viewing experience,
  not a controlled equal-live-count dynamics experiment. `docs/arrival.mp4` is a separate, real-speed arrival:
  `--capture DIR --birds 8000 --seconds 0 --frames 360 --zoom 1`, encoded at
  30 fps (12 s). The movie comes from simulated bird state, not a scripted shape.
  Before the resolved-material correction below, the arrival sequence's mean GPU
  **render** time was 0.48 ms/frame at 960×540,
  on the same GTX 1650/driver, with count growing from zero to 8k. It is not a
  full-8k settled render benchmark or a window FPS measurement.
- The zero-bird frozen part of that arrival capture has no raw shimmer to
  compare. It exposed an old reporting error (dividing by a zero-change control),
  now fixed to print “reduction unmeasured” instead of a meaningless percentage.
- A 35 s window smoke run selected the NVIDIA Vulkan adapter, at 948×1020 in
  Niri, with no validation or mapping errors. It exposed Mailbox as the surface
  default; the window now explicitly uses FIFO/vsync to avoid rendering discarded
  frames and leave time for simulation.

The actual-window screenshot at 1268×1380 then exposed a resolution-dependent
wash-out: many more birds were meshes, and the phase-7 mesh material applied
wing exposure to both projected geometry and opacity. The flock became faint
where the smaller capture was dark. Resolved bodies now use a near-opaque
material (−ln(0.01), with the existing atmosphere and LOD weights), defined in
Rust and checked against WGSL, rather than reusing the unresolved disc's area
factor. This is an explicit rendering adaptation in `docs/model.md`, not a
change to flocking. Unresolved coverage is unchanged. Rechecking the sweep found
another phase-8 `live` regression: its single test bird was inactive, so the
tool reported zero ink and falsely called every transition smooth. It now sets
`live=1`, rejects an empty/mesh-less sweep, and returns an error for a transition
over the 2× growth-control threshold. The corrected, live-bird sweep with the
new material measures **0.71 black pixels** for the worst mesh-band step versus
**0.86** for growth away from a threshold (**0.83×**, under 2×). Ink across the
4–8 px cross-fade is **2.01 → 2.58**, rather than the old translucent material's
2.01 → 0.53. A second actual-window screenshot confirms dark silhouettes instead
of the high-resolution wash-out. The material definition is mirrored by the Rust
material constant and tested against its 1% transmittance.

### Final validation

- Regenerated the evening animation/still and the 12 s real-speed arrival with
  the final material and exact integer recording timeline. Inspected the final
  stills and the actual window, not only the earlier small captures.
- `docs/still-8k-flow.png`: 8,000 live birds, seed 1, t=30 s, 1280×720, GTX 1650
  Max-Q / NVIDIA 615.78.08 / Vulkan. GPU timestamps: **0.75 ms/frame** (`lod`
  0.02, sky/birds 0.32, exposure 0.11, TAA 0.26, tone map 0.05; rounded).
  LOD counts: 4,482 mesh plus 8,000 impostor instances (cross-fading birds occur
  in both lists). This is render time only, not simulation or readback time.
  The final arrival's mean render time is 0.45 ms at 960×540 while its live count
  grows from zero to 8k. Its empty frozen control correctly reports unmeasured.
- The final live-bird LOD sweep passes its 2× threshold, with the 0.83× ratio
  above; it exits successfully and draws a visible mesh. An empty sweep or a
  failed transition now exits with an error instead of printing false success.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and
  `prime-run cargo test -- --test-threads=1` pass: **79 tests passed**, zero
  failed, **two opt-in long tests ignored** (the full 120 s oracle series and
  60 s × five-seed parity sweep). **No GPU test was skipped for lack of hardware.**
  New regressions cover continuous arrival in both engines, live benchmark
  birds, the final live grid offset, compacted decisions with inactive/not-due
  birds, exact percentile selection, refresh-rate-independent stepping,
  observational camera sampling and asynchronous exposure parity.
- `sh -n scripts/evening.sh` and `git diff --check` pass. The recording script
  no longer recursively deletes a caller-supplied directory; it requires an
  empty destination.

Phase 9's larger-count performance work and the model's long-run shape
limitations remain open. No new claim of strong dark waves, vacuoles or
naturalistic split/merge dynamics is made by this revision.

## User rejection, actual startup profile and reference-led revision

The user rejected the previous bird/background appearance and reported roughly
2 FPS **from launch**. This is valid feedback: isolated GPU passes and offline
stills did not establish acceptable interactive playback. PLAN.md reopens visual
acceptance. User photos and the two supplied MP4s are left unmodified. Sampled
video sequences show thin, folding formations with density gradients and small
individual birds; the old rounded blobs and automatic close-up are not a match.

### Reproduced cause, not a guessed FPS fix

Added `--profile-window SECONDS`, with CPU spans around acquisition, simulation
submission, tracking, camera, bindings, render submission and presentation.
Whole GPU simulation batches are timestamped into a bounded history; it is read
only after playback ends, never with a per-frame wait.

The release window on GTX 1650 Max-Q / NVIDIA 615.78.08 / Vulkan / Niri, at
1268×1380 and 8k/seed 1, reproduced **300–400 ms** waits and 40-step catch-up
batches. Before the fixes it presented 149 frames in 12.006 s (**12.41 FPS over
the whole run**, including recovery; the early stall intervals were much worse),
and simulated only 8.540 s. GPU simulation steps peaked at **10.051 ms**. Surface
acquisition and CPU camera work were not the bottleneck: GPU work backed up and
the driver blocked in presentation.

Two related startup problems were fixed:

1. Wide neighbour queries compared against the allocated 8,000 slots and could
   walk all those inactive slots even with only one live bird. They now use the
   grid's exact live prefix and compare cell cost with the live count. A GPU
   regression proves the same neighbourhoods as brute force at 1/17/257 live
   birds in an 8k allocation. The fully populated fallback order is preserved.
2. Fixing sparse queries alone was insufficient: each newborn still started at
   a 5 m perception radius and could scan hundreds of neighbours before adapting.
   Both reference and GPU now initialise that radius from the known arrival flux
   and cross-section (about 1.07 m at peak flux). The formula and endpoint
   regularisation are in `docs/model.md`. Ongoing force weights, integration,
   reaction time and adaptive-radius law are unchanged; `World::new` still
   starts at 5 m and retains the no-arc oracle baseline.

### Actual retest

With the new visuals included, the final instrumented **release window** presented
**887 frames / 15.007 s = 59.11 FPS**, simulating **14.995 s**, at the same
1268×1380 / 8k / seed 1 on the same GPU and driver. Frame intervals: p50 16.641 ms,
p95 18.879 ms, p99 31.359 ms, max 41.593 ms. GPU simulation per-step: p95 1.783 ms,
max 2.889 ms. There was no return to the old 300–400 ms startup catch-up loop.
A compositor screenshot of the newly launched window was taken during the run;
old, still-open app instances were excluded by window ID. This validates that
startup case, not a guarantee at every count/resolution/seed or over an evening.
`docs/window-profile-startup.txt` records both runs and commands.

### Visual changes and remaining scope

- Broader 45° minimum / 55° nominal framing, with the initial camera aimed at the
  actual approach corridor before asynchronous telemetry arrives. Birds remain
  small, with sky and horizon context instead of a close-up of large smudges.
- Narrower, CPU-defined impostor body/wing coefficients. The screen axes are now
  orthogonal: the old independently normalised projections could become almost
  parallel, producing broad streaks and invalidating Gaussian normalisation.
  Quad padding is smaller. Temporal accumulation is 35 ms instead of 100 ms to
  reduce moving-bird trails.
- A procedural sky with world-direction cloud layers, low-sun light and a distant
  horizon. `atmosphere.rs` mirrors the WGSL and a GPU test compares actual rendered
  pixels against it. The palette is still the base; this is not a validated
  physical atmosphere. The presentation remaps arrival lighting into dusk while
  leaving the simulation evening clock alone.
- Reeds now occupy bearings/elevations at three distances with curved tapering
  stems, leaves and seed heads, rather than identical two-pixel screen columns.
  They respond to slow wind and stay outside the flock's exposure histogram.
- The updated **live-bird** LOD sweep passes: mesh transition worst change
  0.58 black pixels versus a 0.92 growth control (**0.63×**, below 2×); the splat
  switch is 0.18 versus 0.25. This establishes the sampled threshold criterion,
  not photographic quality or natural sheet-folding dynamics.

`docs/reference-revision.mp4` is an 8 s / 240-frame, 30 fps real-speed clip at
960×540 starting from simulated t=10 s, with no lens zoom. It is an offline
recording, not evidence of interactive FPS; the separate window profiler supplies
that evidence. `docs/reference-revision.png` is the final 8k/seed-1/t=30 s view at
1280×720. GPU render time for that still is **0.88 ms/frame** (simulation excluded)
on the GTX 1650 Max-Q / NVIDIA 615.78.08 / Vulkan. The moving clip averaged
0.54 ms/render at 960×540. Inspected the new window screenshot and final still:
the bird grain is finer and the scene has sky/foreground context, but the flock
still tends to a compact mass. It does **not** establish a match to the supplied
videos' folding sheets.

Validation also found that the old distant static control multiplied FOV by
four: 45° became a degenerate 180° projection, so the supposedly populated
control had no visible ink. It now scales `tan(fov/2)` by four and holds its scene
exposure constant. A unit test checks the exact focal ratio and a finite lens.
Final regression checks include that correction.

The final complete suite passed **85 tests**, zero failures, two opt-in long
tests ignored, and no GPU skips. Fmt and clippy passed. The distant-control lens
test is included. That control also needed its temporal history reset after the
hard near-to-far camera cut; otherwise stale near-view samples contaminated the
measurement. After resetting history, holding exposure fixed and correcting the
projection, the 30-frame static control at 960×540 / 8k / seed 1 measures **6.8 →
3.2 black pixels/frame**, about **52% less change**, with TAA on. This was checked
with the same NVIDIA/Vulkan GPU; it is a shimmer measurement, not window FPS.

## Falcon zero-stall fix, interactive camera, and sheet-folding pass (2026-10-17)

- **Root cause of the falcon FPS drop fixed:** `Renderer::render` called `self.sim.falcon(&self.gpu)` on every frame when a falcon was active, which allocated a staging buffer and blocked on `gpu.wait_bounded`. Worse, `WindowEvent::CursorMoved` called `self.sim.steer_falcon`, which also called `self.falcon(gpu)` and stalled the CPU on every mouse movement. `FlockTracker` now copies the 96-byte `Falcon` record alongside the bird sample in its single non-blocking staging readback, `steer_falcon` performs a direct 16-byte `queue.write_buffer` on `(mode, aim)` with zero readback, `Renderer::set_birds` and `FlockTracker` cache bind groups across ping-pong buffers, and `falcon_reduce_kernel` filters `src[i].live != 0u`.
- **Measured `--falcon` window profile:** `prime-run target/release/murmuration-gpu --falcon --profile-window 12` on GTX 1650 Max-Q / NVIDIA 615.78.08 / Vulkan presented **719 frames in 12.002 s = 59.91 FPS** (simulated 11.990 s; frame interval p50 16.630 ms, p95 18.866 ms; GPU sim step p50 1.626 ms, p95 1.805 ms).
- **Interactive camera and spontaneous sheet-folding:** Restored natural `FOV_MIN = 16°` (`FOV_V = 42°`), added mouse-drag orbit/tilt, wheel/`+`/`-` zoom, `WASD`/Arrow orbit & dolly, `c` bearing step, `r` reset, `f` falcon toggle, and `Space` pause. Added arc-mode longitudinal cohesion relaxation, boundary shear waves, and spontaneous border turn waves in both `murmur-ref` and `murmur-gpu` (keeping `--no-arc` oracle runs bit-exact). Remaining items are organized in `PLAN.md`.

## Falcon-strike frame drop (real cause), falcon silhouette and the landscape (2026-10-17)

Measured on **llvmpipe (Mesa 25.2.8, LLVM 20.1.2) / Vulkan**, a software
rasteriser, because no GPU was available in this session; absolute numbers are
not representative of the GTX 1650, only the before/after ratios on the same
adapter. The GTX 1650 window profile still needs re-running.

- **The previous entry's fix did not match the committed code.** It describes a
  merged `gather_grid_both` walk and a normal/wide decision partition, but the
  committed `sim.wgsl` has neither. This entry replaces both with the changes
  below.
- **Cause, measured:** after the stoop the flash expansion thins the flock and
  every bird's adaptive radius grows (`radius_step`). At 4,000 birds the share
  of birds with `R ≥ 2 m` goes from ~15% to ~65%, and the `2R` box walked per
  decision from ~300 to ~1,400 one-metre cells, almost all empty. `decide`
  rose from 3.49 to 6.27 ms/step (`--bench --birds 4000 --seconds 12|25
  --falcon --no-arc --steps 40`). The few dozen stragglers whose box exceeds the
  live count also took the flock-wide fallback serially, one thread each.
- **Grid cell sized from the flock** (`grid::cell_for(n)`, 1–2 m, ≈1.4× the
  settled radius): post-strike `decide` 6.27 → **3.59 ms/step**, pre-strike
  unchanged (3.49 → 3.40). Acceleration only: the exact distance test keeps the
  neighbour set identical for any cell.
- **Position-first candidate reads:** walks read the 12-byte `pos` and load the
  96-byte `Bird` only for a kept neighbour; the redundant `live` test on grid
  candidates is gone (`grid_histogram` never buckets inactive birds).
- **Wide queue, one workgroup per straggler:** `prepare_decisions_kernel` sends
  a bird whose `2R` box would take the fallback (exactly `gather_centrality`'s
  test) to a second queue; `decide_wide_kernel` scans the live list with 128
  lanes and a tree sum. Same neighbours, different float summation order (as
  between the grid and brute-force walks). The fallback counter counts the same
  walks as before. New `--bench` pass: `decide_wide`.
- **No catch-up spiral:** `MAX_FRAME_STEPS` 40 → 10. Real time down to 20 fps;
  below that the flock slows instead of scheduling 4× the work into the next
  frame.
- **Mouse no longer hijacks the falcon:** `f` cycles hunting → mouse → off; the
  cursor only steers in mouse mode, and the aiming plane uses a per-frame flock
  centre instead of rebuilding 4,096 predicted positions per cursor event.
- **Falcon silhouette** (`shaders/falcon.wgsl`, `src/falcon.rs`): analytic
  peregrine (wing planforms in their own flapping planes, tail fan, ellipsoid
  body/head), heading from velocity, bank from the turn rate, wings folded in the
  stoop. Drawn at ≥16 px span so it reads at 150–300 m; rendering only. The
  scene block grows to 144 bytes (`falcon_bank`, `falcon_vel`).
- **Landscape** (`shaders/landscape.wgsl`, rewritten `reeds.wgsl`): a bank, a
  calm mere mirroring the sky and far shore, hills/woods/tree lines with aerial
  perspective, two reed banks with plumes; stars fixed to the sky. The ground
  band moved out of `atmosphere_sky` (and its CPU mirror) so the water can
  reflect the sky; the GPU/CPU sky test now samples sky pixels only, and a new
  test checks the land is finite, darker than the sky and not a flat band.
- The falcon is drawn before the reeds, so near stems occlude it.

### README clip from the new landscape and silhouette (2026-10-17)

Recorded on the **GTX 1650 Max-Q / NVIDIA 615.78.08 / Vulkan** — the software
rasteriser above is what the numbers in this entry come from, but this recording
is the real GPU. Headless, from the app's own `--record` mode, no window:

```fish
prime-run target/release/murmuration-gpu --record /tmp/evening --birds 8000 \
    --seed 1 --arc-minutes 1 --frames 180 --falcon
```

180 frames written in 28.1 s. The first 14 frames are the arrival (17 birds live
at t = 0.2 s) and are trimmed from the clip, which starts at source frame 14,
i.e. `t ≈ 4.8 s`, and runs to the end of the evening minute; 8,000 birds live
from frame 20 on.

- **What the clip shows.** The golden arrival and the fold while the flock is
  high (source frames 14–115); the falcon in frame as a separate silhouette from
  roughly source frame 120, i.e. about 5 s into the clip; the bank, the mere and
  the reed banks from source frame ≈150, i.e. about 6.8 s in, with the stars out
  and the flock low over the water at the end.
- **Files:** `docs/evening-falcon.gif` (720×405, 66 frames, 8.33 fps, 4.65 MB) and
  `docs/evening-falcon.webp` (720×405, 165 frames, 20 fps, 2.24 MB). Encoded with
  ffmpeg: `fps=8,scale=720:-1` plus `palettegen(max_colors=192,stats_mode=diff)`
  and `paletteuse(dither=bayer:bayer_scale=4)` for the GIF; `libwebp_anim -q:v 72`
  for the WebP. The WebP keeps every second frame and is both smaller and
  smoother — the GIF is the inline fallback.
- **Still open:** the falcon is only in frame for the last third of the evening,
  so a clip that leads with the silhouette needs a tighter framing than
  `--record` picks; nothing in this recording establishes that the strike itself
  changes the flock's shape.

## Release workflow (2026-10-18)

`.github/workflows/release.yml`: pushing a `v*` tag builds one artifact set per
platform, creates the GitHub release with generated notes, and appends
`SHA256SUMS.txt`. A manual run builds the same artifacts and uploads them as
workflow artifacts without publishing, so a packaging change can be tried
without a tag.

- **Linux x86_64** (`ubuntu-22.04`, for an older glibc): `.deb` from
  `cargo-deb 3.8.0` (metadata in `app/Cargo.toml`), plus a `.tar.gz`. The job
  runs the binary headless before packaging, so a release cannot be built from a
  binary that will not start (see the smoke-test note below).
- **macOS** (`macos-14`, both `aarch64-` and `x86_64-apple-darwin`): a
  `Murmuration.app` bundle with a generated `Info.plist`, ad-hoc signed, shipped
  as a `.dmg` and a `.zip`. Not notarised, so Gatekeeper quarantines it.
- **Windows x86_64** (`windows-latest`): a `.zip`, and an Inno Setup installer
  built from the checked-in `scripts/windows/murmuration-gpu.iss` (Inno Setup 6
  ships with the runner image; the job fails loudly if `ISCC.exe` is missing).
- **Version discipline:** the tag must equal `[workspace.package] version`, so
  `v0.2.0` with a 0.1.0 binary fails the run instead of shipping a mislabelled
  release. Only `murmuration-gpu` is packaged; `murmur-ref`'s `analyze` stays a
  workspace tool.

Verified locally on 2026-10-18, before any CI run:

- `cargo deb -p murmuration-gpu --no-build --deb-version 0.1.0` produced a valid
  archive: `control` carries `Package`, `Version`, `Architecture: amd64`,
  `Section: science`, `Depends: libvulkan1`, `Recommends` and the four-line
  description; `data` carries `/usr/bin/murmuration-gpu` (0755), the README and
  the copyright file. `license-file` and `recommends` are accepted by cargo-deb.
- The macOS `Info.plist` as generated by the workflow parses with Python's
  `plistlib` and substitutes the version. The bundle and staging steps were run
  against a stand-in target directory.
- The version-resolution step was run for a matching tag, a manual run and a
  mismatched tag (fails as intended). The workflow parses as YAML (`js-yaml`).

Not verifiable locally: `codesign`, `ditto` and `hdiutil` (no macOS on this
machine), the Windows job, and `$auto` shlib dependencies (`dpkg-shlibdeps` is
not installed here, so the local `.deb` names only `libvulkan1`; the runner
installs `dpkg-dev`). The manual run below covered the first two, and the `Linux
package` job's `Depends` assertion is what proves `$auto` resolves.

`ci.yml` gained a Linux packaging job that builds and inspects the `.deb` on
every push (binary present, `Depends` resolvable), so the metadata cannot rot
between releases.

**What the manual run showed** (`workflow_dispatch`, run 39123133317,
2026-10-18; the publish job stays skipped for a manual run, by design):

- **macOS, both architectures: pass.** The `Murmuration.app` bundle, `codesign
  --deep --sign -`, `ditto` and `hdiutil` all did what they were written to do,
  and both `.dmg`/`.zip` pairs are in the run's artifacts. This is the one part
  of the pipeline that cannot be checked by reading the script.
- **Windows: pass.** The Inno Setup installer compiled from
  `scripts/windows/murmuration-gpu.iss` with the image's ISCC 6.7.1, next to the
  `.zip`.
- **Linux: failed, and correctly so.** The smoke test exited 101. The adapter was
  found — `llvmpipe (LLVM 15.0.7, 256 bits)` on Vulkan, Mesa 23.2.1 — and the
  run then panicked at `crates/murmur-render/src/renderer.rs:1389`
  (`expect("waiting for exposure")`) exactly 60.2 s after the adapter line.

That 60.2 s is `WAIT_BUDGET`, and it exposes something about the app rather than
about the packaging: the headless paths take the synchronous exposure branch, so
a frame whose histogram does not arrive inside the 60 s bound panics instead of
reporting that the adapter is too slow to run at these settings. `--lod-sweep`
is a render command and hits it; on the GTX 1650 the same command takes 0.5 s and
never gets close. The Linux job now smoke-tests with
`--analyze 1 --birds 1000 --seed 1` under `timeout 300` and asserts on the
adapter line and a `t=` line — a model-level check that the packaged binary
starts, opens a device and advances — and leaves pipeline coverage to
`cargo test` in `ci.yml`, which builds and renders on every push anyway.

**Still open:** nothing is code-signed or notarised, so Gatekeeper and SmartScreen
both warn on first open — that is a certificate the project does not have, not a
gap in the workflow. The render-on-a-slow-adapter panic above is an app question,
not a workflow one. The release side is done: `v0.1.0` is published, below.

**Second manual run, after the smoke-test change (run 39123840396): all five jobs
pass** — Version, Linux x86_64 in 2m21s, both macOS targets in about a minute,
Windows x86_64 in 1m55s, and the publish job skipped as a manual run should. The
artifacts were downloaded and inspected rather than taken on the run's word:
the `.deb` is a Debian format-2.0 package, the `.tar.gz` a gzip archive, both
macOS binaries are Mach-O (`arm64` and `x86_64`) inside a `Murmuration.app` that
carries a `_CodeSignature/CodeResources` from the ad-hoc `codesign`, the two
`.dmg` files are Apple disk images, and the Windows side is a `PE32+` x86_64
console executable plus a `PE32` GUI `*-setup.exe` (Inno).

One detail that inspection turned up: `app/src/main.rs` sets no
`windows_subsystem` attribute, so the Windows binary is a console-subsystem PE —
launching the installed app opens a console window behind the flock. Hiding it
would also hide the `--bench`/`--analyze`/`--record` output the rest of the
project relies on, so it is left as it is and noted here instead.

**Found while doing this, unrelated to the workflow:** `ci.yml` has been red for
at least ten consecutive runs, back to phase 7 (`6262bfb`), and the failure is
different on each platform: `macos-latest` on
`a_falcon_alarms_the_birds_it_swoops_past` (a panic inside wgpu 30.0.1),
`ubuntu-latest` on the two `murmur-ref` agitation tests
(`the_agitation_wave_matches_the_terminal_run`,
`the_wave_speed_is_inside_the_reference_range`), `windows-latest` on
`a_resumed_run_takes_the_same_steps` (`crates/murmur-gpu/tests/analyze.rs:142`).
The same suite passes on the GTX 1650 here, 88 tests, no skips. The new `Linux
package` job passes.

**First tag: `v0.1.0`** (run 39124554528). All six jobs pass, publish included,
and [the release](https://github.com/ImanolGo/murmuration-gpu/releases/tag/v0.1.0)
carries eight assets plus `SHA256SUMS.txt`.

The first attempt on the tag (run 39124281918) failed in the publish job, and it
was a real bug in the workflow rather than a runner quirk. The checksums step ran
`sha256sum *` over `dist/`, which by then held two **directories**: the macOS
upload had swept in the whole `Murmuration.app` bundle tree, and the Windows one
the staging folder it zips. `sha256sum` refuses directories, so the step exited 1
before the release could be created. The uploads now name what they want
(`dist/*.dmg`, `dist/*.zip`, `dist/*.exe`) and the checksum step hashes files
only, so a regressed upload pattern reports itself as a missing asset instead of
as a checksum error. Checked against a `dist/` holding both directories: eight
files hashed, and `sha256sum -c` verifies.

The tag was moved onto the fix rather than bumping to `0.1.1`: no release had been
published at that point (`gh release view v0.1.0` said "release not found"), and
`0.1.0` is what the binary reports, so re-tagging the same version is the honest
option — bumping would have left a version no build ever claimed.

Verification of the *published* result rather than of the build logs: all eight
assets were downloaded from the release and `sha256sum -c SHA256SUMS.txt` passes
on every one; `file` reports a Debian format-2.0 package, a gzip archive, two
Apple disk images, two macOS zips, a Windows zip and a `PE32` GUI Inno setup; and
the released Linux binary, unpacked from the `.tar.gz`, runs on the GTX 1650 here
(`--analyze 1 --birds 500`, adapter `NVIDIA GeForce GTX 1650 with Max-Q Design`,
0.39 s).
