# Model notes

How the `murmur-ref` model relates to the published StarDisplay model and to the
sibling terminal project [ImanolGo/murmuration](https://github.com/ImanolGo/murmuration).
AGENTS.md requires that every documented adaptation or deviation from the
published model is written down here. Murmuration's `DEVELOPMENT.md` is the
source for the adaptations it carries; the entries below were checked against
its code before being copied here.

## Provenance

The model is a phase 1 reimplementation of `murmuration`'s `sim/` (MIT licence),
written here rather than vendored or depended on (see `docs/progress.md`). The
same constants, names and order of operations are kept on purpose: that is what
makes the oracle comparable and what the GPU mirrors (AGENTS.md, "The model is
the specification"). `Cargo.toml` pins the same `glam` and `rand` versions the
reference uses, so the arithmetic and the random streams are identical.

**Verified:** `analyze --birds 2000 --seed 42 --seconds 120` reproduces every
per-second line of the recorded terminal baseline
(`docs/reference-analyze-2000-42.txt`) byte for byte, and the test
`tests/oracle.rs` asserts it.

## Deviations carried over from the reference model

These are the reference's adaptations to Hildenbrandt, Carere & Hemelrijk (2010)
and, where the paper is internally inconsistent, to its intent. They are part of
the model, not GPU shortcuts, and the GPU reproduces them:

- **Banking (paper Eq. 19–21).** The printed roll update is a high-gain
  integrator that pins the roll at the clamp when applied per 5 ms step. Replaced
  by a coordinated turn: bank follows `atan(a_lat/g)` with a first-order lag
  (`TAU_ROLL = 0.15 s`) and a maximum roll rate (`MAX_ROLL_RATE = 6 rad/s`).
  `--turning inertial` adds roll-rate momentum (`TAU_SPIN = 0.3 s`).
  See `flight::step_bird`.
- **One source of lateral acceleration.** Lateral steering requests bank and the
  banked lift supplies the sideways force; the lateral component is not also
  added directly. Forward and vertical control stay direct.
- **Adaptive radius (paper Eq. 2).** The printed equation shrinks `R` at
  `|N| = n_c` and is unstable for the Table 1 values. Replaced with a stable
  geometric relaxation `R ← R·(n_c/|N|)^(s/3)` (`params::radius_step`).
- **All-round, unit-direction cohesion; alignment keeps the blind sector.**
  Cohesion uses every spatial neighbour (as separation does), not only the
  forward hemisphere, and is the unit resultant scaled by centrality. This
  departs from the paper's blind sector for cohesion.
- **`W_COH = 0.65 N`, not the paper's 1 N** (0.65 balances the corrected
  separation and banked flight without compressing the flock).
- **Lift divided by `cos β`** (clamped), so a banked bird holds height; the
  paper's `|a_l| sin β` term is negligible and pumped energy in.
- **Alignment (paper Eq. 8).** Uses the mean neighbour forward and applies the
  misalignment vector unnormalised (normalising it turned a nearly-aligned
  bird's tiny misalignment into a full sideways kick).
- **Separation (paper Eq. 4).** `exp(−(d−r_h)²/2σ²)` — with the factor of 2.
- **Vertical roost force (paper Eq. 12).** Capped: `w_RoostV · clamp((z0−z)/h,
  −1, 1)` with `h = 40 m`, not an uncapped spring that squeezes the flock into a
  sheet.
- **Horizontal roost force (paper Eq. 11).** Scaled by `dist_h − R_roost`, with a
  signed inward lateral direction and a minimum effective overshoot of
  `ROOST_TURN_MARGIN = 25 m`. A bird beyond `R_roost` commits to an inward turn
  until it faces inward (`f·n < −0.95`); an outward-heading bird in the outer
  20 % can anticipate the return when two neighbours are returning.
- **Bank clamp ±80°** (`MAX_BANK`), a numerical safeguard not in the paper.
- **Initial condition.** A compact, slightly flattened, already-aligned cloud at
  the roost edge (the paper starts birds at random positions and orientations;
  that did not self-organise into one flock here). The evening arc spawns
  arriving groups instead.
- **Speed relaxation (Eq. 1/14)** is applied; speed is clamped to
  `[0.5 v0, 2 v0]`.
- **Spatial grid.** An unbounded spatial hash (2¹⁶ buckets, exact distance
  check), so birds far outside the roost can still find each other.
- **Falcon / escape.** Autonomous cruise→climb→stoop→retreat hunter; alarm radius
  30 m; flee force; flash expansion inside 8 m above 20 m/s; zigzag forced roll
  ±50° over 0.6 s copied when `zigzag_k` neighbours are in a copyable zigzag
  (a neighbour counts only once its manoeuvre is at least `Δu` old, which bounds
  the wave speed).

## Deviations specific to `murmur-ref`

- **Decisions run sequentially, not with `rayon`.** Each bird reads a snapshot
  and its own random stream, so sequential and parallel execution are identical;
  the oracle does not need the parallelism.
- **`--analyze` prints an extra extended line** with `--extended` (see
  `stats.rs`). The default per-second line is unchanged.
- **State exchange uses a `Snapshot`** (birds plus the scalar world state), not a
  bare `Vec<Bird>`: the world clock and `z0` cannot live in a per-bird struct,
  and they are needed to resume exactly. See `state.rs`.
- **Experiments are deferred.** The forced-turn (`turns.rs`) and agitation
  (`escape::measure_agitation`) experiments belong to PLAN.md phase 7 and are not
  in the phase 1 oracle.

## Deviations specific to `murmur-gpu` (phase 2)

The GPU shader (`crates/murmur-gpu/shaders/sim.wgsl`) mirrors the reference
function for function, keeping the names and the order of operations:
`flight::forward`/`side_of`/`body_up`/`step_bird`, `params::radius_step`,
`escape::zigzag_roll` and `social::decide`. It deviates only where the CPU and
GPU are genuinely different machines:

- **Brute-force neighbour search.** PLAN.md phase 2 asks for the simplest
  correct version: `decide` loops over every bird (O(N²)) instead of walking the
  CPU's spatial hash. The set of neighbours within the radius is the same; only
  the order the partial sums are accumulated in differs, which is what the
  single-step parity measurement sees (1e-7 relative). Phase 3 replaces it with
  the grid (see the phase 3 adaptations below).
- **Counter-based random stream, not `ChaCha8Rng`.** The reference draws its
  random force from each bird's `ChaCha8` stream. A `ChaCha8Rng`'s position in
  its stream cannot be extracted to hand to the GPU, and reproducing `rand`'s
  float sampling exactly in WGSL is out of scope (PLAN.md lists bit-exact GPU
  determinism as out of scope). The GPU seeds a 32-bit PCG stream per bird from
  `(seed, id)` (`bird::initial_rng`) and advances it two draws per decision,
  the same count and shape as `social::random_unit`. Consequence: the noise is
  statistically equivalent but not bit-identical, so short-horizon parity is
  exact only with the noise switched off, and long-horizon parity is
  statistical (see `parity.rs`).
- **`World::w_xi` is a seam, not a model change.** The random-force weight is a
  parameter (`Snapshot::w_xi`, default `params::W_XI`) so the CPU/GPU
  single-step check can switch the noise off on *both* engines at once and
  compare the deterministic forces to 1e-7. The default is the model value, so
  the `--analyze` oracle is unchanged (verified byte for byte).
- **The bird record is 112 bytes, not PLAN.md's 64-byte sketch.** Phase 2 is the
  correctness version; the record carries the fields the mirrored functions
  need (steering held between decisions, the decision clock, the escape state
  the falcon will use in phase 7) plus a 32-bit RNG state. `speed` and `fwd`
  from the sketch are derived in the shader exactly as the reference derives
  them, so they are not stored. The layout is guarded against the WGSL struct by
  `tests/layout.rs`; phase 9 may shrink it with a benchmark, as PLAN.md asks.

### Phase 3 adaptations

- **The neighbour search is a radius query, not k-nearest.** PLAN.md phase 3
  said the pass "keeps the k = 7 nearest in a small register array". The model
  does not cap at 7: `radius_step` drives the radius towards `N_C = 6.5`, and
  `decide` uses *every* bird inside it, plus a `2R_i` walk for centrality. A cap
  would change `n_all`, the sums, the centrality and the radius update, so the
  grid mirrors `Grid::for_each_within` (all birds within the radius) instead.
  This is recorded in PLAN.md as a change to the plan, not to the model.
- **Cells hash into a table sized from the bird count, not the oracle's fixed
  2^16.** `Grid::buckets_for` uses `next_power_of_two(2N)` clamped to
  `[2^16, 2^22]`. The bucket count only changes which bin a cell lands in; every
  query also checks the exact cell, so the neighbour set is unaffected. The cell
  size itself is *not* free: it is `params::GRID_CELL` (4.0 m), the oracle's
  value, so both engines put the same birds in the same cell. The table is a
  performance knob left for phase 9, where the benchmark will size it.
- **The scatter fills a `sorted` array of bird indices, not a reordered bird
  array.** PLAN.md phase 3 says the scatter should "reorder the bird state". The
  records stay in place and the query goes through the index array. The effect
  is the same and it avoids moving 112-byte records; physically reordering is
  left to phase 9, when the record is smaller and the benefit is measurable.
- **Each bucket's range is sorted by bird index after the scatter.** A parallel
  `atomicAdd` scatter leaves the order inside a bucket to the scheduler, so two
  runs of the same command gave slightly different flocks (measured: the
  `nn_ab` metric moved by more than its own tolerance between two runs, and the
  lock-in parity check flipped pass/fail). The model is deterministic
  (AGENTS.md), so one extra pass restores a fixed order; it costs 0.1–4 ms per
  step over the benchmark's bird counts. The neighbour set never depended on
  this order, only the last bits of the f32 sums did.
- **The lock-in parity check runs with the random force off.** This is a change
  to a phase 2 check, made in phase 3. With the noise on, the two engines draw
  from independent generators, so they are two realizations and the check only
  compares statistics — the same weak statement the 60 s run makes. Measuring
  the noise-free divergence (mean 0.3 m after 1 s, 3.7 m after 5 s) showed the
  check has to be about the same path, not about smoothing independent runs, so
  it now switches the noise off like the single-step test. Every metric then
  stays within its 10 % bound for 5 s, the worst at 40 % of its bound; the
  earlier noise-on version was marginally failing on one seed.
- **The device requests the adapter's limits, not the WebGPU defaults.** The
  3M-bird acceptance run holds 336 MB in one storage buffer, past the default
  256 MB `max_buffer_size` and 128 MB `max_storage_buffer_binding_size`. This is
  a native build on a named GPU, so the portability floor is the wrong bound;
  requesting `adapter.limits()` is always valid and is what the driver reports.
- **The benchmark starts from a synthetic settled flock.** The model's initial
  condition puts the flock in an 18 m ellipsoid; at three million birds that is
  thousands of times denser than any settled flock, and the grid's cost follows
  the local density (birds per perception radius), so a benchmark from that
  state would measure a state the model never reaches. `bench::setup` scatters
  the birds uniformly at 0.5 birds/m³ with a 1.5 m radius — about seven
  neighbours, the reference's settled value — and warms up for 20 steps. The
  numbers are therefore honest for a settled flock, and this is stated wherever
  the benchmark is reported.
- **Two grid buffers play two roles each.** `counts` is the histogram until the
  level-1 scan has read it and then the scatter cursor; `block_sums` is the
  block totals until the single `grid_scan_totals` workgroup has read them all
  into shared memory and then the block offsets. Both reuses are within one pass
  boundary or one workgroup, so neither races, and they keep the pipeline inside
  the WebGPU limit of eight storage buffers per stage. `shaders/grid.wgsl`
  documents both.

### Phase 4 adaptations

Phase 4 added no model behaviour: the audit in `PLAN.md` phase 4 checks each of
its bullets function by function and finds every one already ported in phase 2
and passing parity. (The falcon-dependent rules — flee, flash expansion and the
alarm copying — are inert with no falcon, which is this phase's state; they
belong to phase 7.) The entries here are about making the 100,000-bird
acceptance run feasible and honest, and they are recorded as changes to the
plan, not to the model.

- **The initial cloud grows with the bird count (`World::new_density_scaled`).**
  The model's initial condition is a fixed 18 m ellipsoid (`CLOUD_RADIUS`). At
  100,000 birds the same volume holds fifty times the birds, so the flock starts
  as a near-solid ball: the first simulated second took over 50 s and tripped the
  driver's GPU watchdog (device lost) on an early attempt. `new_density_scaled`
  grows the radius by `(n / 2000)^(1/3)` to hold the density of the 2,000-bird
  cloud, so the initial state is the *same kind* of state at every count. It
  draws the birds in the same order, so `World::new` — the model's own initial
  condition, the one the oracle is verified against byte for byte — is unchanged;
  the scaled cloud is used only by the large-`n` GPU run.
- **The GPU grid's cell is 1 m, not the oracle's `GRID_CELL = 4.0 m`.** Phase 3
  tied the two together; the cell is an acceleration parameter, not model state,
  because both engines discard every candidate outside the exact radius, so any
  cell size returns the same neighbour set (the phase 3 test compares them
  against the brute-force walk). At the model's settled density the interaction
  radius is under 1 m at 100,000 birds, and the 4 m cell then makes each query
  scan roughly an order of magnitude more candidates than the 1 m cell; see
  `grid::CELL`. This is the acceleration freedom the plan's phase 3 text already
  allows ("a free performance knob"), and it is a plan change, never a model
  change. It is a good choice for the settled density but a bad one for the
  radius *tail* — see the bounded query below, which is what makes the tail
  affordable.
- **The neighbour query is bounded by the flock size, not only by the radius
  box.** `gather_grid` walks every cell the radius box covers, so its cost is
  cubic in `R`; with the model's `params::R_MAX = 100 m` a single straggler at
  `R = 100 m` costs `201³ ≈ 8.1e6` cells at the 1 m cell — as much as ~300,000
  ordinary birds — and the 100,000-bird run's step time jumped two orders of
  magnitude the second the first wide-radius bird appeared (`docs/progress.md`,
  third attempt; the model does grow such birds, `Rh ≈ 177 m`). `gather` and
  `gather_centrality` (`shaders/sim.wgsl`) therefore take the brute-force walk
  whenever the box covers more cells than the flock has birds. Both walks keep a
  candidate only after the same exact `d2 <= r2` test, so the neighbour set is
  identical — the same acceleration freedom as the cell size, never a model
  change — and `tests/grid.rs::a_wide_radius_query_matches_bruteforce` forces the
  crossover and pins it. The choice is observable: `sim::Clock::wide` counts the
  queries that fell back each step (`Sim::wide_queries`).
- **`--coherence` samples a lighter statistic set than `--analyze`.** The
  criteria in `docs/coherence-100k.md` read only `|N|`, `R`, `v`, `roll`, the
  5 m cluster fraction and a few O(N) reductions of the positions, but
  `analysis::compute` also pays for a 15 m nearest-neighbour scan that is almost
  all-pairs at a dense flock: measured at 100,000 birds the full metric set costs
  ~26 s per simulated second, so the 600 s run would take ~5 h and almost all of
  it would be statistics the verdict never reads. `coherence::sample` computes
  exactly the criteria fields with the same formulas in the same order as
  `compute`, and `tests/analyze.rs` asserts they are bit-identical; the one
  non-free field, the cluster fraction, is the same `analysis::cluster_fraction`
  function `compute` calls and still runs every second. `--analyze` keeps the
  full reference output for comparability. This is also a change to the plan
  (item 2 had the run go through `--analyze`).
- **Both runs submit in small batches, not one command buffer per second.** At
  large counts a single `decide` can take seconds, so a whole second of
  dispatches in one submission can outlive the watchdog; ten steps per submission
  keeps each one short while the sampling stays per second.
- **The long runs repair themselves when the driver hangs.** The development
  laptop's NVIDIA driver hangs on a long continuous compute run (`Xid 109`,
  `CTX SWITCH TIMEOUT`; a known Linux driver bug, seen twice on the 100,000-bird
  run after ~50 min and ~1 h 20 min). Every wait is bounded
  (`device::WAIT_BUDGET`, 60 s) so the hang is reported as `device::Stalled`
  rather than blocking for ever in `poll(wait_indefinitely)`, and
  `analyze::each_second_resilient` then reopens the device and resumes from the
  last sampled second. This is *not* a model change: the resume state
  (`sim::RawState`) keeps each bird's own random stream and the grid is derived
  from the positions, so the repaired run takes exactly the steps the
  uninterrupted one would have —
  `tests/analyze.rs::a_resumed_run_takes_the_same_steps` asserts the records are
  byte-identical after a repair. It is recorded here because the acceptance's
  numbers come from a run the driver interrupted, and the record must say so.
  Two details of the repair are model-neutral but were bugs once: the run's step
  count rewinds to the last sample with the flock (otherwise the sampler drifted
  off the whole seconds), and `MAX_RESUME_TRIES = 3` fruitless repairs end the run
  with `RunError::NoProgress` instead of retrying for ever.

### Phase 5 adaptations

Phase 5 is the first render (`crates/murmur-render`), which mirrors the
reference's `render/` modules. Its deviations from them:

- **The camera is driven from a CPU copy of the flock, read back every frame.**
  `render::camera` in the reference reads the live `World`; here the same
  camera code runs on the CPU over a full GPU→CPU readback of the bird records
  each frame. The framing maths is the reference's, but the readback is a
  temporary cost PLAN.md phase 9 removes (the renderer will read the bird
  storage buffer directly). It is a performance deviation, not a visual one.
- **Darkness is multiplicative and black, where the reference uses INK.** The
  reference's optical-depth compositor adds `INK`-tinted optical depth and
  tone-maps the result. This renderer instead writes coverage into an HDR target
  with `dst *= 1 - coverage`, with per-bird `coverage = 1 - exp(-exposure·raw)`,
  so the product over the flock is `exp(-exposure·tau)` — the same optical depth
  the reference accumulates, but composited in one pass and darkened toward
  black rather than toward a colour. Tone mapping then maps that optical depth
  to the output; the identity (clamp) map is used for now.
- **Exposure adapts from a 256-bin histogram, with a blocking readback.** The
  reference's automatic exposure is mirrored in its constants
  (`DEFAULT_EXPOSURE = 25`, `TARGET_TAU = 1.204`, `EXPOSURE_TAU = 4`, exposure
  clamped to `[1, 400]`) and in its behaviour (a histogram of per-pixel optical
  depth drives the target exposure), but the histogram is summed on the GPU and
  read back on the host each frame, which stalls the pipeline. Phase 9 replaces
  the readback; the exposure it produces follows the reference's rule. That
  includes the rule's floor: at 20,000 birds and above a settled flock's
  80th-percentile raw depth already exceeds `TARGET_TAU`, so the exposure rests
  at `EXPOSURE_MIN = 1.0` and the core is as dark as the rule allows. It is the
  reference's own behaviour at density, not a port artifact (`docs/progress.md`,
  phase 5), and it is not tuned here.
- **The sky ray includes the horizontal term.** The reference's sky varies by
  row only (a function of the view's pitch); here the ray's direction is used,
  so the gradient varies with the actual view direction. The palette and the
  time-of-evening interpolation are the reference's.
- **The reference's distance-opacity fade buffer is not implemented yet.** The
  reference fades a bird's opacity with distance into a separate buffer; this
  pass has the distance term in the coverage but not the reference's separate
  fade target. Recorded so the difference is visible rather than assumed absent.

## Known gaps (reported, not hidden)

Carried over from the reference and not fixed here (fixing them would change the
model the GPU mirrors):

- The flock mills over the roost rather than crossing it; the boundary force is
  radial for a tangentially moving bird, so it cannot remove the initial angular
  momentum.
- Long runs grow elongated (`I2/I1`, `I3/I1` above the paper's ranges) and can
  split. Now visible in a still: the 60 s, 20,000-bird flock in
  `docs/still-20k.png` has become a ribbon, while the same 60 s at 1,000,000
  birds (`docs/still-1m.png`) is still a mass.
