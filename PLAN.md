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

**Phase 9 measurement update (2026-10-17): the estimates above are not met.**
Timestamped simulation pass totals on the named GTX 1650 Max-Q / NVIDIA
615.78.08 / Vulkan are 8.905 ms/step at 100k, 129.738 ms at 1M and 429.237 ms
at 3M (40 samples each, synthetic live-bird scatter at 0.5 birds/m³). All three
are dominated by `decide`, and the fixed 5 ms model requires **200 steps/s**,
not one simulation step per presented frame. These are headless measurements,
with no render resolution; they do not imply presentation FPS. A real 8k
evening at t=30 s measures 1.304 ms/step (100 samples, arc passes excluded).
The interactive default is therefore 8k, and the README reports the measured
numbers rather than promising real-time million-bird simulation. Profiling and
larger-count optimisation remain open.

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

## Phase 2 — GPU simulation, brute force (done)

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

**Update (2026-10-15) — tolerances, chaos, and the accumulator.** Building
the `--parity` harness showed that a single 2,000-bird flock is chaotic in its
*global shape* but self-averaging in its *local structure*, so "±10 % on each
metric" is not one target:

- The local, self-averaging statistics (`|N|`, nearest-neighbour distance,
  cluster fraction, alignment fraction, radius, spread, extent, height, speed,
  bank, centroid radius) agree within ±10 % (usually far below) on every seed.
- The global shape metrics (`I2/I1`, `I3/I1`, `flat`) do **not** self-average
  over one minute at 2,000 birds: the CPU's own value swings by an order of
  magnitude from seed to seed (here `I2/I1` ranged 3.2–30.4). A ±10 % bound on
  them is therefore not a meaningful match criterion yet; they are compared at
  ±50 % and left to tighten on phase 4's larger, longer runs.
- To keep a *tight* comparison, phase 2 adds a short-horizon **lock-in** check:
  both engines stepped together must agree on *every* metric (at ±10 %) for the
  first seconds, before the trajectories diverge. That is what proves the
  shader mirrors the dynamics; the 60 s run only proves the statistics match.

The noise is also a documented adaptation, not a target: the GPU uses a
counter-based PCG stream per bird instead of the CPU's `ChaCha8` (whose stream
position cannot be extracted), so bit-exact noise parity is out of scope, as
recorded in the stack notes. Short-horizon parity is exact only with the noise
switched off on both engines, through a default-preserving `w_xi` seam.

The "fixed-step loop, accumulator, step capped at 1/120 s" line is split:
`Sim::advance` runs the fixed 5 ms step now; the wall-clock accumulator and the
1/120 s presentation cap belong to the real-time render loop and land with the
first render (phase 5–6), where wall-clock time enters.

**Met (2026-10-15):** at 2,000 birds the shader reproduces the reference to
`1.1e-7` relative per bird in one noise-free step (target `1e-4`); both engines
lock in within 10 % on every metric for the first 5 s; and the 60 s statistics
agree across five seeds (within 10 %, or the relaxed 50 % for the chaotic shape
metrics) with the full table recorded in `docs/parity-2000-five-seeds.txt`.
Details in `docs/progress.md`.

## Phase 3 — Spatial grid and topological neighbours (done)

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

**Update (2026-10-15) — the neighbour set is radius-based, not k-nearest.**
This phase's second bullet said "keeps the k = 7 nearest in a small register
array". The model does not do that. `N_C = 6.5` is the *target* count the
adaptive radius `radius_step` is driven towards; `decide` then uses *every*
bird inside that radius `r` (and a second walk over `2r` for centrality). A
hard k = 7 cap would change `n_all`, the separation and alignment sums, the
centrality, and the radius update itself. AGENTS.md makes `murmur-ref` the
specification, so phase 3 keeps the radius set and the "k = 7" wording is
dropped. The GPU grid therefore reproduces `Grid::for_each_within`, not a
k-nearest structure.

**Update (2026-10-15) — unbounded worlds and the sorted array.** The hash
table was chosen over per-step recentring: it mirrors the CPU oracle's
`neighbours.rs` exactly, needs no reduction or second grid pass, and keeps
working when birds roam far from the roost. The table is `next_power_of_two(2N)`
clamped to `[2^16, 2^22]`. The table size never changes which neighbours are
found (every query also checks the exact cell), so it is a free performance
knob; the cell size is the model constant `GRID_CELL = 4.0 m`, shared with the
oracle so the sets are identical. The scatter writes bird *indices* into a
`sorted` array rather than physically reordering the 112-byte bird records: the
reorder the plan asks for is provided through that indirection, and moving the
records themselves is left to phase 9 when the record shrinks. One extra pass,
not in the plan, sorts each bucket's range by bird index: a parallel scatter's
atomic order is not reproducible, and the model (and the tests) need the GPU to
be. The grid is two of its buffers playing two roles each, to stay inside the
WebGPU limit of eight storage buffers per stage.

**Met (2026-10-15):** at 2,000 birds the grid's neighbour sets match the
brute-force walk for 2,000/2,000 birds (worst steering relative error 1.9e-6);
the scan and the bucket sort match a CPU computation on random flocks at 500,
2,000 and 5,000 birds; phase 2's parity still passes (single step `1.065e-7`,
lock-in, and all five seeds settled). The lock-in check now runs with the
random force off: with independent noise the two engines are different
realizations and it was only comparing statistics, and the discovery that even
a noise-free run diverges (0.3 m after 1 s) is what made that the right choice.
`--bench` reports per-pass GPU timestamps at 100k (14.9 ms/step), 1M (163 ms)
and 3M (499 ms) birds on the GTX 1650; the table is in
`docs/bench-gtx1650.txt` and the cost is dominated by `decide`, whose candidate
count comes from the 4 m cells rather than from the bird count. Details in
`docs/progress.md`.

## Phase 4 — Full flight model

- All remaining behaviour from the reference: boundary-weighted cohesion,
  banked turning from requested roll, cruise speed, preferred altitude, roost
  attraction and committed returns, random perturbation with a seeded
  per-bird generator (for example PCG keyed by `id` and step).
- `--analyze` in the app, matching the reference output format.

**Accept:** parity passes with all behaviours on. At 100k birds the flock
keeps a coherent display over the roost for 10 simulated minutes without
collapsing or dispersing, judged by the statistics.

**Update (2026-10-15) — the audit: the model work was already phase 2.** Before
writing code this phase, each bullet above was checked function by function
against `murmur-ref` (the specification). Every behavioural item is already in
`shaders/sim.wgsl`, ported in phase 2 and passing parity, so this phase adds no
model behaviour:

| bullet | reference | shader | done in |
| ------ | --------- | ------ | ------- |
| boundary-weighted cohesion | `social.rs` `coh.normalize_or_zero() * (W_COH * centrality)` | `resolve` same line | phase 2 |
| banked turning from requested roll | `flight.rs::step_bird` (`atan(a_lat/g)`, lag, rate) | `step_bird` | phase 2 |
| cruise speed | `social.rs` speed control + `step_bird` lift/drag/thrust | `resolve` + `step_bird` | phase 2 |
| preferred altitude | `social.rs` capped vertical roost pull | `resolve` | phase 2 |
| roost attraction, committed returns | `social.rs` `returning` block | `resolve` | phase 2 |
| seeded per-bird random force | `social.rs::random_unit` (ChaCha8) | counter PCG keyed `(seed,id)` | phase 2 (documented adaptation) |

The parts of `social::decide` that are *not* in the shader are the ones that are
inert with no falcon to run in this phase: `escape::force` (flee and flash
expansion) is absent, and the copying rule is present but never fires because no
bird is alarmed; the alarm update in `world::step` likewise needs a falcon. They
belong to phase 7 and change nothing here — on the CPU they return `Vec3::ZERO`
/ do nothing when `falcon` is `None`, which is exactly the state this phase
runs. So the only genuinely new work is:

1. `--analyze` in the app, printing the same per-second line and summary as
   `cargo run -p murmur-ref --bin analyze --release -- --birds N --seed S
   --seconds T [--extended]`. The GPU's noise is not bit-identical to the
   CPU's, so at one seed the output is *statistically* comparable, not
   identical (the same caveat `--parity` records).
2. The 100,000-bird, 10-simulated-minute coherence run. Its pass/fail criteria
   are fixed in advance in `docs/coherence-100k.md` (the acceptance is
   subjective, so it is written down before the data) and checked by
   `murmur_gpu::coherence`.

**Update (2026-10-15) — making the 100,000-bird run feasible.** The acceptance
run at 100,000 birds exposed three plan/implementation mismatches, all recorded
in `docs/model.md` and `docs/progress.md`:

- The model's fixed 18 m initial cloud holds ~50× more birds at 100,000 than the
  2,000-bird cloud it was designed for, so the first second took over 50 s and
  tripped the GPU watchdog. `World::new_density_scaled` grows the radius as
  `n^(1/3)` to hold the density; `World::new` (the oracle's own initial
  condition) is unchanged.
- Phase 3 tied the GPU grid cell to the oracle's `GRID_CELL = 4.0 m`. The cell
  is an acceleration knob (the exact-distance check makes the neighbour sets
  identical for any cell), and 4 m is far too coarse for the model's sub-metre
  radius at 100,000 birds. `grid::CELL = 1.0 m` (the plan's phase 3 text already
  calls the cell "a free performance knob").
- The plan's item 2 had the run go through `--analyze`, but the full reference
  metric set costs ~26 s per simulated second at 100,000 birds (its 15 m
  nearest-neighbour scan is nearly all-pairs in a dense flock), i.e. ~5 h for
  600 s, and almost all of it is statistics the criteria never read. So the run
  uses `--coherence SECONDS`, a lighter sampler that computes exactly the
  criteria fields (asserted bit-identical to `analysis::compute` by
  `tests/analyze.rs`); `--analyze` keeps the full reference output.

**Met (2026-10-16).** The criterion is the line at the top of this phase: *at
100,000 birds the flock keeps a coherent display over the roost for 10 simulated
minutes without collapsing or dispersing, judged by the statistics.* The run

```
target/release/murmuration-gpu --coherence 600 --birds 100000 --seed 1
```

**passes all 13 pre-declared criteria over all 600 seconds**, in one
uninterrupted run: 2 h 29 min of wall clock, zero device repairs, no `Xid`.
Thresholds were fixed before the run (`docs/coherence-100k.md`) and not touched;
the verdict was re-derived independently from the printed series. The series is
`docs/analyze-100k-600s.txt`, the argument is `docs/progress.md` (the third
attempt) and `docs/coherence-100k.md` ("Verdict").

The line here used to read "Met (2026-10-15): recorded in `docs/progress.md` and
`docs/analyze-100k-600s.txt`", which was **wrong** — that file held only the
run's header, because the run hung and was killed before it printed a second.
Getting to a real verdict took three attempts and found three problems, all now
recorded and fixed:

- The run is **hours, not the hour** the earlier estimate gave, because `decide`
  is density-bound and the settled flock is several times denser than the
  initial cloud (`docs/coherence-100k.md`, "How long it really takes").
- The neighbour query's cost is **cubic in the adaptive radius** while the model
  lets a straggler's radius reach `R_MAX = 100 m`. The accepted run's own record
  shows the tail spiking to 55.9 m at `t = 152` — the window both earlier runs
  died in — where one bird costs as much as ~300,000 ordinary ones. The query is
  now bounded (`gather` in `sim.wgsl`: when the box outgrows the flock, walk the
  flock, same exact neighbour set, pinned by `tests/grid.rs`). This is an
  acceleration change, never a model change.
- The NVIDIA driver **hangs** on a run this long (`Xid 109`, a known Linux
  driver bug), which killed the first two attempts. Every wait is now bounded and
  the run reopens the device and resumes from the last sample, with the resumed
  state proved byte-identical to the uninterrupted one
  (`tests/analyze.rs::a_resumed_run_takes_the_same_steps`). The accepted run did
  not need a single repair.

The acceptance therefore rests on the third item's repair path being correct
rather than on having been lucky, but the run that produced the verdict was
clean: the repairs are a net that was not used.

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

- Phase 5 lives in `crates/murmur-render` (`camera`, `sky`, `renderer`, `png`,
  shaders `scene.wgsl`, `sky_pass.wgsl`, `coverage.wgsl`, `exposure.wgsl`,
  `tonemap.wgsl`) and is driven by the app's `--still PATH` (headless) and the
  window. The camera is a CPU port of the reference `render/camera.rs` fed by a
  readback for now (PLAN.md phase 9 removes the readback); the deviations from
  the reference renderer are listed in `docs/model.md` ("Phase 5 adaptations").
- Note the same cost shape as phase 4 applies to a 1M-bird still: a *settled*
  1M flock is much denser than `--bench`'s fresh one, so the still needs the
  seconds it simulates to be chosen from a measurement, not from the bench's
  fresh-flock step time.

**Met (2026-10-16).** `docs/still-1m.png` — 1,000,000 birds, seed 1, 60 s of
simulation — is a dark, dense flock with lighter speckled edges against the dusk
sky; `docs/still-20k.png` is the same frame time and material at a fiftieth of
the birds, and reads differently in shape for a reason that belongs to the model
rather than the renderer: the 20,000-bird flock has stretched into a ribbon by
60 s (the long-run elongation in "Known gaps"), where the million-bird flock is
still a billowing mass. The frame
cost **3,344 s (56 min)**, 55.7 s per simulated second: exactly the "capture
stills" point above — the still is a simulation, not a screenshot. Both frames sit
at the exposure rule's floor (`EXPOSURE_MIN = 1.0`), so the core saturates to
black; that is the reference's own rule at this density (`docs/progress.md`,
phase 5), recorded rather than tuned.

Getting there changed two things, both in `docs/progress.md`:

- The first capture, at 20 s, was a transient *tube*, not a settled flock — the
  density-scaled initial cloud at 1M is ~143 m across, near the 150 m roost — so
  the acceptance still simulates 60 s.
- The first attempt at 1M exposed a real bug: `Sim::advance_seconds` queued the
  whole interval and then made one bounded wait, so a healthy GPU reported
  `Stalled { waited: 60s }`. It now drains after every 10-step batch.

The acceptance word "similar" is a judgement, not a measurement: the terminal
reference draws braille cells, so tone, shape and edge structure are what is
claimed, not image equality. Long-run elongation stays a known gap.

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

**Met (2026-10-16), all three criteria measured.** Phase 6 lives in
`crates/murmur-render` (`mesh.rs`, shaders `lod.wgsl`, `bird_pass.wgsl`,
`bird_mesh.wgsl`, `impostor.wgsl`, `taa.wgsl`, the rewritten `coverage.wgsl`,
`scene.wgsl`) and is driven by the app's `--capture DIR` and `--lod-sweep`
modes; the deviations from the reference renderer are in `docs/model.md`
("Phase 6 adaptations") and the measurements in `docs/progress.md` (phase 6).

- **No popping.** `docs/lod-sweep.txt` sweeps one bird's projected wingspan from
  `0.25` to `12 px` in `0.25 px` steps with the accumulation and jitter off, so
  each step isolates the level change. The mesh cross-fade's worst step is
  `1.01` black px against a `0.92` control — the same bird growing the same
  amount away from a threshold — i.e. **1.10×**, under the stated `2×` jump
  threshold; the `1.5 px` splat switch is `0.13` vs `0.25`. `docs/flypast.mp4`
  records the acceptance fly-past (`--birds 20000`, lens zooming `1.00 → 0.35`):
  across it the LOD counts go `mesh=0 impostor=20000 splat=0` →
  `mesh=20000 impostor=0 splat=0`, so every bird crosses the mesh threshold
  within the recording, and the frame-to-frame change shows no spike. The one
  change the cross-fade carries is the bird *thinning* as it resolves (the mesh
  is a silhouette, the splat an optically-thin disc): the ink moves `3.20 → 2.30`
  black px over the band, gradually over eight steps (`docs/model.md`), and the
  flock's total darkening is unchanged to better than 1 % (20k still: 150.00 →
  148.96; 1M still: 120.22 → 119.90).
- **No obvious shimmer.** `docs/shimmer.mp4` is a frozen, four-times-wider (so
  sub-pixel) flock with the temporal accumulation on and off: the raw
  frame-to-frame change is `293.8` black px/frame, accumulated `68.5` — **77 %
  removed**. The moving fly-past is `75 %` (phase 7 re-measured it at 61 % after the mesh
  correction below; the static figure is unchanged). `--still` also converges the
  accumulation over 24 jittered samples, which is why `docs/still-2k-mesh.png`
  is clean.
- **Frame time at 1M.** GPU timestamps (`renderer.frame_times_ms`), GTX 1650
  Max-Q, driver 615.78.08, Vulkan, `1280×720`: **`12.19 ms/frame`** at 1,000,000
  birds (`birds 11.21 ms` of it; `mesh=190 impostor=554516 splat=445370` — a
  settled 1M flock is sub-pixel almost everywhere). 20,000 birds: `0.84 ms`;
  2,000 birds (all mesh): `0.47 ms`. The still that reached the 1M frame cost
  `3484 s` of simulation (the phase-5 cost), not a render cost.

**Update (2026-10-17):** phase 7 found that the mesh's optical depth was ~12×
the reference disc's peak, which pinned the auto-exposure at its floor and
saturated the flock. The mesh now deposits the disc peak `2a/π` and the
cross-fade is `2 px` wide, so the cross-fade's worst step is `1.52×` the growth
control (still under the `2×` jump threshold) and the ink moves `2.01 → 0.53`
black px over sixteen steps (`docs/model.md`, "Phase 6 correction found by phase
7"). `docs/lod-sweep.txt`, `docs/still-2k-mesh.png`, `docs/still-20k-p6.png` and
`docs/capture-20k.txt` are re-measured; the wider band brings a `ρ ≥ 4 px` fringe
of the 20k flock into the mesh level. GPU run-to-run bit-reproducibility is out
of scope (below), so the exact fringe varies between runs.

## Phase 7 — Falcon and agitation

- Falcon with patrol, climb, dive and retreat, autonomous or mouse-steered.
- Flee forces, close-threat expansion and copied zigzags with the
  `zigzag-k` threshold. Copying reads the previous state buffer, so it
  propagates one neighbour per step, as in the model.
- `--agitation` experiment matching the reference output.

**Accept:** a falcon attack produces visible dark waves; agitation
propagation speed is within the range reported by the reference at the same
bird count.

**Met (2026-10-17), with one honest caveat.** Phase 7 adds the falcon
(`crates/murmur-gpu/src/falcon.rs`, `shaders/falcon.wgsl`, the alarm and flee in
`sim.wgsl`), the CPU agitation oracle (`murmur-ref::escape::measure_agitation`)
and the GPU experiment (`murmur-gpu::agitation`), the falcon's footprint in
`crates/murmur-render`, and the app's `--falcon`, `--agitation`, `--zigzag-k`,
`--zoom` and window mouse steering. Measurements are in `docs/progress.md`
(phase 7); the deviations are in `docs/model.md` ("Phase 7 adaptations").

- **Agitation speed.** `--agitation` at 2,000 birds: the GPU and the CPU oracle
  agree to 1–4 % (`k=2`: 7.52 / 6.80 / 7.10 m/s for seeds 42 / 1 / 7 against the
  oracle's 7.56 / 6.85 / 6.81; `k=1`: 14.66 vs 15.15; `k=3`: 3.97 vs 4.01). The
  terminal version reports 6.8–7.6 m/s at this count and `k=2`, so the GPU is
  inside that range, and every run is inside the 3–25 m/s the terminal prints.
- **A falcon attack.** The attack is real and measured: the alarm matches the
  oracle bird for bird (2000/2000 alarmed, 874/874 flash-expanded, the roll
  direction agreeing 2000/2000), the falcon flies its full patrol → climb →
  stoop → retreat cycle, the zigzag share sweeps the flock (0 → 41 % → 0 over
  ~3 s), and `docs/falcon-attack.mp4` records a 5,000-bird attack whose frame
  brightness dips ~6 % and recovers. **Caveat:** the *dark band* is much less
  visible here than in the terminal version, because a dense flock is optically
  thick and the roll only changes near-black pixels; the wave is measurable and
  visible in the flock's thin parts, and the flash expansion is clearly visible,
  but it is not the bold band the terminal shows. Recorded as open rather than
  claimed.
- **A phase 6 correction.** Driving the falcon showed the mesh's optical depth
  was ~12× the reference disc's peak, pinning the auto-exposure at its floor and
  saturating the flock. The mesh now deposits `2a/π` (the disc peak) and the
  cross-fade widened to `2 px`, so the exposure behaves as the model's rule
  intends and the wing-exposure structure survives (`docs/model.md`).

## Phase 8 — The evening

- Arrival, display, roosting and night phases, as in the terminal version.
- Reeds in the foreground with wind, stars at night.
- `--seed`, `--birds`, `--falcon`, `--no-arc`, `--arc-minutes`, matching the
  terminal CLI where they make sense.
- Recording script writing a GIF/WebP and a still into `docs/`, captured
  from the running binary at a fixed simulation step.

**Accept:** a full evening runs unattended; the README's GIF is recorded
from the real app.

**Met (2026-10-17).** Phase 8 adds the evening arc on the GPU
(`crates/murmur-gpu/src/arc.rs`, `shaders/arc.wgsl`), the `live` flag that lets a
fixed-size flock grow and shrink, the foreground reed bed and the night stars
(`crates/murmur-render/shaders/{reeds,sky_pass}.wgsl`), the terminal's
`--no-arc` and `--arc-minutes`, and `--record` with `scripts/evening.sh`.
Deviations are in `docs/model.md` ("Phase 8 adaptations"), the timeline in
`docs/progress.md` (phase 8).

- **A full evening runs unattended.** `--record DIR --birds 8000
  --arc-minutes 3` runs the whole arc without intervention: the flock grows from
  nothing through the arrival, holds for the display, pours down as the roost
  level sinks to the reed tops, and is cleared at night, with the sky and the
  stars following. Both engines are checked (`crates/{murmur-ref,murmur-gpu}`/
  `tests/arc.rs`): live `0 → 2000`, held through the display, `z0 100 → 3`
  through the roosting, `live 0` at night. One model property is recorded rather
  than hidden: `remove_landed` retires a bird only below `REED_TOP = 3 m`, which
  the descent reaches exactly at the roosting/night boundary, so the flock is
  cleared at night rather than thinned during the roosting — the reference's own
  behaviour at `Z0 = 100`, `REED_TOP = 3`.
- **The README's animation is from the app.** `docs/evening.webp` (and the still
  `docs/evening.png`) are the encode of `--record`'s frames; `scripts/evening.sh`
  reproduces both from the binary. Nothing is authored separately.
- **One structural change, recorded.** The reference's `World` adds and removes
  birds; the GPU buffer is a fixed size, so the arc fills the next free slots and
  clears a landed bird's `live` flag in place, and the grid, the neighbour walk,
  the integration and the renderer all skip inactive birds. The flock's forces
  read only the surviving birds, so the behaviour matches; the record order does
  not.

## Phase 9 — Performance & Remaining Roadmap

### Completed in this pass (2026-10-17)
- **Zero-stall falcon path:** Removed the blocking GPU readbacks in `Renderer::render` (`sim.falcon()`) and `WindowEvent::CursorMoved` (`steer_falcon`), cached ping-pong bind groups in `Renderer` and `FlockTracker`, and updated `falcon_reduce_kernel` to only average `live != 0u` birds without shared-memory barrier aliasing. Windowed `--falcon` now holds **59.91 FPS** (12 s, 8k birds, GTX 1650 Max-Q).
- **Interactive camera controls:** Added mouse-drag orbit/pitch, scroll-wheel & `+`/`-` lens zoom, `WASD`/Arrow orbit & dolly (move near/far), `c` bearing step, `r` view reset, `f` falcon toggle, and `Space` pause.
- **Spontaneous sheet-folding & turning waves:** Added arc-mode anisotropic longitudinal cohesion relaxation, boundary shear waves, and spontaneous border turn-initiation in both `murmur-ref` and `murmur-gpu` (keeping `--no-arc` oracle tests bit-exact).

### Still to be done (Next Steps Plan)

1. **Multi-flock splitting, merging, and vacuole dynamics (Model + GPU)**
   - Allow large flocks to naturally tear into 2–3 sub-flocks during sharp turns or predator stoops and smoothly re-merge, rather than a single global roost return pulling every lobe toward `(0, 0, z0)` at the same rate.
   - Tune local topological cutoffs (`R_MAX` / border cohesion falloff) in `murmur-ref` and `sim.wgsl` so interior cavities (vacuoles) open and close cleanly during flash expansion.

2. **Full Inertial Spin Turning Model (Cavagna et al. / Attanasi et al.)**
   - Expand `--turning inertial` from a single-bird roll-rate damper into neighbour spin-coupling (propagating angular momentum conservation across topological neighbours so turning waves travel ballistically without damping out).

3. **Camera Follow Modes & Smooth Cinematic Presets**
   - Add camera mode switching (e.g., `1`: Ground Observer, `2`: Flock-Follow chase camera riding alongside the ribbon, `3`: Falcon-POV camera during stoop).
   - Add an on-screen HUD overlay (`h` key) showing live FPS, bird count, current evening phase, camera mode, and active falcon state.

4. **Large-N (100k–1M) Real-Time Scaling (Phase 9 Core)**
   - **Physical Bird Reordering:** Scatter birds directly into sorted cell order (`Bird` records in `sorted` order) so neighbour walks hit contiguous L2 cache lines instead of chasing `sorted[s]` indices.
   - **64-byte Compact `Bird` Layout:** Pack render/secondary fields into `f16` / `u16` (as sketched in the Data Layout section) to cut global memory bandwidth in half.
   - **Multi-step Grid Reuse:** Rebuild and sort the spatial hash grid every 2–4 integration steps (`10–20 ms`) instead of every `5 ms` step (since `next_decision` only fires every `DU = 50 ms`, 90 % of steps don't even query the grid if `alarm` is separated).
   - **Subgroup Scan:** Use `wgpu` subgroup operations in `grid_scan_blocks` where supported.

5. **Visual Polish & Environment**
   - Add optional motion-blur / velocity-stretched impostors during high-speed falcon stoops.
   - Add subtle ground/water reflections at the horizon (like the Brighton pier / lake reference videos) and distant tree/pier silhouettes.
   - Add runtime `+` / `-` (or `[` / `]`) flock size adjustment without restarting the window.

## Out of scope for now

- Exact global k-nearest-neighbour search.
- Bit-exact GPU determinism.
- A full implementation of the inertial spin model (only the simplified
  experiment from the reference, if time allows).
- Web/WASM builds (possible later through wgpu's WebGPU backend).
