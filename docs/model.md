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

### Phase 6 adaptations

Phase 6 gives the flock birds (PLAN.md "Phase 6 — Birds"): a low-poly mesh, an
impostor quad and the phase-5 splat, chosen by projected size in a compute pass,
plus temporal anti-aliasing. The reference has none of this — it draws braille
dots — so the geometry, the thresholds and the accumulation are the renderer's
own. What is taken from the model is the **wingbeat**: the dihedral angle is
driven by the model's `flap_phase` and `flap_amp` at the model's `FLAP_HZ = 10`
Hz, the same two numbers `render/splat.rs` uses to flicker the wing exposure.
The optical **mass** is also the model's: every level carries `a·ρ²`, so which
level draws a bird does not change how much light it absorbs in total.

- **The mesh is ten triangles.** A starling silhouette: two wing strips, a body
  and a tail. The peak dihedral `FLAP_DIHEDRAL = 0.7 rad` is this renderer's own
  constant (the reference has no wings to rotate to find it); the phase and rate
  are the model's. `docs/progress.md` (phase 6) records the geometry.
- **The mesh deposits a flat optical depth per fragment** (see the phase 7
  correction below: it is the reference disc's peak `2a/π`, not the mass
  `a·MEAN_AREA` normalisation phase 6 first shipped).
- **The impostor is a mass-preserving Gaussian mixture.** A body and two wing
  lobes as normalised Gaussians, divided by the sum of their own integrals, so
  its total optical depth is `a·ρ²` by construction. This is new: the reference
  has no shape between a dot and a resolved bird. Its sigmas are the renderer's
  own (`impostor.wgsl`). Its *peak* per-pixel depth is `~1.55a`, higher than the
  reference disc's `2a/π ≈ 0.64a`, because the same mass is packed into a
  narrower, bird-shaped footprint rather than a broad disc — a deliberate
  trade-off of exact per-pixel depth for shape. A consequence, recorded and not
  hidden: an impostor flock is dense enough that its 80th-percentile optical
  depth exceeds `TARGET_TAU`, so the auto-exposure rests at its `EXPOSURE_MIN`
  floor at 20,000 birds and above. That is the same floor the reference reaches
  at density (its dot model floors there too), but it arrives at a lower bird
  count here. The mesh and the splat are at the model's per-pixel scale; the
  impostor is not.
- **The LOD thresholds are renderer constants.** Mesh `≥ 6 px` with a `1 px`
  cross-fade band `[5, 7]`, impostor `[1.5, 6)`, splat `< 1.5`. The mesh and
  impostor thresholds are a *cross-fade*: a bird inside the band is written to
  both lists with weights summing to one, and every level carries the same mass,
  so only the bird's *shape* changes as it crosses. The thresholds are chosen
  where the two shapes differ by less than a pixel; `--lod-sweep` measures the
  result (`docs/progress.md`).
- **A bird that resolves gets thinner: ink is not preserved.** The splat spreads
  `a·ρ²` over a disc of diameter `ρ` (area `π/4·ρ² ≈ 0.785ρ²`); the mesh
  concentrates it through a silhouette of mean area `0.0851ρ²`. At the same
  per-pixel optical depth (the phase 7 correction below) the mesh therefore
  covers far less area, and its **total ink is lower** — an optically thin dot
  model darkens a wide patch, a bird's silhouette does not. This is the physics
  the dot model cannot express, and it is why the mesh is worth ten triangles.
  `--lod-sweep` measures the size of it: across the `[4,8]` px cross-fade the
  ink moves `2.01 → 0.53` black pixels (`3.78×`) spread over sixteen `0.25 px`
  steps — a gradual thinning, not a step.
- **No depth buffer, and back-face culling is off.** The scene is optically thin
  and the pass composites in call order: a depth buffer would sort a single
  instance's own triangles for no gain, and culling back faces would hide the
  underside of a wing seen from below, which is most of the flock. The draw
  order is therefore the level order, and within a level the instance order.
- **Temporal AA has no motion vectors.** Each frame jitters the bird passes by a
  sub-pixel Halton(2,3) offset (carried in `Scene.jitter`) and the resolve blends
  the fresh frame into a running image with `alpha = 1 - exp(-dt/TAA_TAU)`,
  `TAA_TAU = 0.1 s`. There is no reprojection — no depth buffer and no motion
  field — so the defence is the standard 3×3 neighbourhood clamp: the history
  sample is clamped into the range the current frame shows around that pixel,
  which rejects stale content. It is enough for the sub-pixel case it exists for
  (a frozen flock: 77 % of the frame-to-frame change removed, `docs/progress.md`)
  but it is not a reprojection, and a fast camera move will ghost. Motion
  vectors are listed as open.

### The splat look under the LOD change

Level 2 (`coverage.wgsl`) is the phase-5 pass unchanged except that it reads the
classifier's instance list and scales the mass by the cross-fade weight. A bird
larger than `RHO_SPLAT` that used to be drawn by this pass is now drawn by the
impostor or the mesh, so the phase-5 still's dense black core is now a mass of
impostor dashes and mesh silhouettes. At 20,000 birds the two stills are
near-identical to the eye (`docs/still-20k.png` vs `docs/still-20k-p6.png`),
which is the continuity this phase is for.

### Phase 7 adaptations

Phase 7 adds the falcon and the agitation wave (PLAN.md "Phase 7"). The
mechanics are the model's: `murmur-ref` carries `escape.rs` (threat detection,
the flee and flash-expansion forces, the zigzag manoeuvre and its copying rule)
and `falcon.rs` (the autonomous hunter), both mirrored from the terminal
version, and the GPU now runs them per step.

- **The falcon lives in the GPU's step.** `world.rs::step` updates the falcon,
  then alarms the birds it is near, then decides and integrates; the GPU does the
  same with two small passes (`shaders/falcon.wgsl`): one workgroup reduces the
  flock to its centroid and extent (which the falcon steers by, exactly as
  `world.rs` computes them from `self.position` every step), and one invocation
  advances the falcon. The alarm and the flee/flash force run inside the decide
  pass (`sim.wgsl`), on every bird every step, as in the reference.
- **The falcon's random stream is the one adaptation.** `Falcon::new` and the
  retreat-to-cruise transition draw from a ChaCha8 stream that cannot run in
  WGSL. The GPU draws from the same counter-based PCG as the birds, seeded from
  the run's seed, and both sides advance it with the same step, so
  `Falcon::spawn` and the shader agree on the first draw. The attack times are
  therefore a different draw from the terminal's, but from the same distribution
  (5–20 s), and the parity test compares the alarm on a fixed falcon instead.
- **The alarm reads the previous step's neighbour state, and it does not
  matter.** `world.rs` alarms the flock *before* rebuilding the grid and deciding,
  so a decision sees neighbours alarmed in the same step. The GPU alarms inside
  the decide pass and the neighbour walk reads `src`, the step's input. A
  neighbour's zigzag started this step is therefore not visible to the copy rule
  until the next step — but the rule requires a zigzag at least `DU = 0.05 s`
  old (the model's own guard against the wave hopping a neighbour per step), so
  a new start is *never* copyable anyway. The copyable set is identical.
- **`--agitation` runs on both engines.** `murmur-ref::escape::measure_agitation`
  is a faithful port of the terminal's experiment, and
  `murmur-gpu::agitation::measure` runs the same seeding and the same fit over a
  GPU flock (a per-step readback, because the experiment needs each bird's first
  zigzag start). The two agree to within a few per cent (below), and the GPU's
  speed is inside the 3–25 m/s the terminal reports at the same bird count.

### Phase 6 correction found by phase 7: the mesh's optical depth

Driving the falcon exposed a phase-6 mistake. The mesh deposited the model's
optical *mass* `a·ρ²` through its silhouette (`a / MEAN_AREA` per fragment),
which makes a mesh bird fully opaque: its per-fragment optical depth is ~12× the
reference's disc peak. The reference's auto-exposure drives the 80th percentile
of the optical depth to `TARGET_TAU = 1.204` and clamps the exposure at
`EXPOSURE_MIN = 1.0`; with a mesh flock that percentile is far above the clamp,
so the exposure pinned at its floor and every mesh bird saturated to black. Two
consequences: the flock's wing-exposure structure (which is what a falcon attack
makes visible) was lost, and the model's own exposure rule could not do its job.

The mesh now deposits the **reference disc's peak optical depth**, `2a/π`
(`mesh::DISC_PEAK`), per fragment — the per-pixel convention the model's
exposure rule and darkness assume. A mesh bird is then exactly as dark per pixel
as a dot bird of the same size, and the exposure behaves as it does for a dot
flock (at 2,000 birds the exposure settles near 5, as the rule asks, instead of
pinning at 1). Its *total* ink is lower, because a silhouette is thinner than a
disc of the same wingspan — that difference is what the mesh↔impostor
cross-fade carries, and it is measured: `--lod-sweep` moves the ink `2.01 →
0.53` black pixels across the band, over sixteen `0.25 px` steps.

The cross-fade was widened from `1 px` to `2 px` for the same reason: with the
correct per-pixel depth the ink difference between the levels is larger, and the
narrower band let the fade's per-step change reach `2.3×` the growth control.
At `2 px` it is `1.52×` (`docs/lod-sweep.txt`), back under the stated jump
threshold.

### Phase 8 adaptations

Phase 8 adds the evening (PLAN.md "Phase 8"): the arrival, display, roosting and
night phases, a foreground reed bed and stars, and a recording script. The arc
is the model's — `murmur-ref/src/arc.rs` mirrors `murmuration::sim::arc`, and the
phases, the evening brightness ramp, the approach corridor and the landing rule
are the reference's own.

- **The flock grows and shrinks by activation.** The reference's `World` adds
  and removes birds; the GPU's storage buffer is a fixed size. The arc therefore
  spawns an arriving group into the *next free slots* and retires a landed bird
  by clearing its `live` flag in place. An inactive bird is skipped by the
  neighbour walk, the flight integration, the falcon's alarm and the LOD
  classifier, so it is absent from the flock exactly as a removed record would
  be. Two consequences are recorded rather than hidden: the reference's removal
  *compacts* the array (so a removed bird's swap can reorder the flock), and its
  `len()` shrinks; here the slot count never changes and the arc's own `next`
  stands in for the live count. Neither difference is observable in the model's
  behaviour — the flock's forces only ever read the surviving birds — but the
  order of the records is not the reference's.
- **The arc runs on the GPU, one step at a time.** `shaders/arc.wgsl` advances
  the evening clock, sets the roost level for the phase and spawns groups; a
  second pass retires landed birds and clears the flock at night. The evening's
  time is `clock.time % duration`, so the arc needs no clock of its own. The
  shader's `resolve` reads the roost level from the arc state instead of the
  params, which is how `world.rs` lets `arc.update` move `z0` under the model.
- **One adaptation: the random stream.** The oracle draws its arrival bearings,
  corridor radii, group sizes and group shapes from ChaCha8, which cannot run in
  WGSL; the GPU draws from the same counter-based PCG as the birds, and both
  sides advance it with the same step, so a run is reproducible on one engine.
  The groups are from the same distributions, not from the same draws.
- **The landing and the night clear coincide.** `remove_landed` retires a bird
  only once it is below the reed tops (`z < REED_TOP = 3 m`), and the roosting
  phase's descent reaches `REED_TOP` exactly at the roosting/night boundary, so
  in practice the flock is cleared at night rather than thinned during the
  roosting. That is the reference's own behaviour at these constants (Z0 = 100 m,
  REED_TOP = 3 m); the port keeps it and `crates/murmur-gpu/tests/arc.rs` records
  it.
- **The reed bed is procedural, not braille.** The terminal draws its reeds as
  braille dots; here `shaders/reeds.wgsl` draws a sparse band of swaying stems in
  the bottom of the frame, gated by `sky::reed_visibility` (the reference's
  angular band, ported) so the stems appear when the camera looks low enough.
  Like the reference, they are drawn last, over the birds. They are a
  *foreground* silhouette and are not part of the flock's optical depth, so they
  do not enter the exposure histogram.
- **Stars are drawn when the evening is past 0.75.** `sky_pass.wgsl` adds a
  sparse field in the upper half of the frame, from the reference's own hash
  (about 6 in 4096 pixels), lit as the sky darkens.

## Visual revision and phase 9 work (2026-10-17)

The user requested a natural, continuous arrival instead of the visibly separate
half-second banks in phase 8. This intentionally revises the arrival **initial
condition** in both `murmur-ref::arc` and `shaders/arc.wgsl`; it does not change
the social forces, cruise speed, perception, banking or escape parameters.

- Arrival fills over `min(6 s, 0.15 × evening duration)`. The cumulative count is
  `floor(N × (3u² − 2u³))`, where `u` is elapsed arrival time divided by that
  span, clamped to `[0,1]`. Births occur each integration step, with no random
  group size or periodic half-second pause. The display boundary completes any
  fractional last bird.
- A shared corridor is 180–200 m from the roost (formerly 280–320 m). Its heading
  is `−0.6 × radial + 0.8 × tangent`: a grazing approach rather than a beeline
  through the centre. Birds are distributed uniformly across an elliptical
  cross-section, with vertical scale 0.6 and radius
  `max(3 m, sqrt(N / (v0 × span × π × 0.5)))`. Longitudinal jitter is ±0.5 m;
  heading jitter remains ±0.05 horizontally and ±0.025 vertically. The
  cross-section depends on total flux, not the number born in one frame.
- CPU and GPU still use their documented different random streams. The arrival
  count schedule is shared, not the individual trajectories. The old terminal
  arrival recording is no longer the oracle for this revised initial condition;
  the unchanged no-arc analysis baseline remains an oracle for flight behaviour.
- The app defaults to 8,000 birds, rather than 2,000, for a fuller silhouette.
  The ground camera's minimum vertical lens angle is 12°, rather than 6°, so
  the first few arriving birds do not trigger an extreme close-up. Exact
  percentile selection replaces full sorting. Window framing uses at most 4,096
  uniformly sampled slots, copied asynchronously as 32-byte position/velocity
  records; inactive slots are excluded and the short telemetry delay is
  predicted from velocity (at most 0.25 s). Captures retain full-flock framing.
- Window auto-exposure uses the same histogram and adaptation rule but consumes
  completed readbacks without waiting. At most one request is outstanding;
  captures keep synchronous exposure. This introduces a small observation delay,
  not a change to the exposure target or the flock model.
- The real-time loop retains fractional 5 ms steps. It no longer discards time
  at ordinary refresh rates or forces a step at extremely high refresh rates.
  Captures use this accumulator too; time-lapse samples use absolute step times
  and update the camera with their actual simulation interval. Mouse movement
  steers a falcon only after `f` or `--falcon`, not merely when watching.
  Presentation uses FIFO/vsync rather than the surface's preferred Mailbox mode,
  avoiding GPU work on frames that would be discarded between refreshes.
- Exact simulation accelerations: decisions due on a step are compacted into
  the grid's now-unused bucket scratch buffer before the neighbour pass. Queries
  still read the previous flock; independent per-bird random streams and neighbour
  accumulation order are unchanged. Cells whose AABB cannot touch the query
  sphere are skipped conservatively, before hashing. The final grid offset is
  computed from the actual live histogram, not the inactive buffer capacity.

### Resolved-bird material correction found in the actual window

The 1268×1380 window exposed a problem that 960×540 captures hid: more birds
resolve to meshes at high pixel density, and phase 7's `2a/π` mesh material makes
them nearly transparent. A resolved wing's projected area already changes with
bank and wingbeat. Multiplying that silhouette's opacity by the unresolved
disc's wing-exposure factor counts wing exposure a second time and makes the
flock fade as it resolves.

Resolved silhouettes now use a material optical depth **`MESH_TAU = −ln(0.01)`**
(4.6051702, 99% absorption at unit atmosphere and exposure), multiplied by the
existing distance attenuation and LOD weight, not by the disc's area term. The
Rust definition in `mesh.rs` and the WGSL constant are checked together. This
supersedes the phase-7 mesh material above; unresolved impostors and splats keep
the reference's coverage law, and no simulation parameters change. The optical
histogram still sees the material actually drawn, so dense mesh views may reach
the exposure floor; opaque bird bodies are intentional. The cross-fade remains
necessary because a silhouette and a Gaussian disc have different spatial mass.
The revision rechecks the LOD sweep and records the result in `docs/progress.md`.

## Startup / photographic revision after user testing

The user reports roughly 2 FPS from launch and supplied `Brightons-Stunning-
Starling-SnapYT.App.mp4` (33.8 s, 30 fps), `The-Secrets-and-SnapYT.App.mp4`
(13.9 s, 25 fps), and three photos. The earlier headless pass timings did not
prove acceptable windowed playback. They also did not prove the desired folded
flock morphology. These references supersede the earlier subjective visual
acceptance, not the unchanged no-arc mathematical oracle.

- **Exact sparse search.** When the grid has inactive capacity, wide queries
  walk its live sorted prefix, not every allocated slot. The crossover compares
  cell count with the live count. All exact-distance tests are unchanged; the
  fully populated fallback retains its former accumulation order. A regression
  compares 1, 17 and 257 live birds in an 8,000-slot buffer with brute force.
- **Density-aware arrival perception, on both engines.** A new bird no longer
  starts at 5 m regardless of the arrival density. At arrival fraction `u`, flux
  is `6N u(1-u)/span`, and density is flux divided by `v0 × πr² × 0.6`, where
  `r` is the existing corridor cross-section radius. Perception starts at
  `cbrt(n_c / (4π/3 × max(density, 0.05)))`, clamped to `[r_h, 5 m]`. The floor
  regularises the nearly zero flux at the endpoints. At peak flux this is about
  1.07 m, rather than scanning hundreds of neighbours until the adaptive law
  settles. This revises arrival initial state only: `World::new`, the ongoing
  adaptive law, reaction interval, integration step and force weights are
  unchanged. The CPU and GPU implement the same expression.
- **Photographic framing.** Minimum vertical FOV is 45°, nominal 55°, instead of
  zooming to 12°. This keeps birds small and leaves sky/horizon context. The
  window initially aims at the known approach corridor before telemetry arrives,
  so the first bird does not cause a horizon-to-flock camera snap.
- **Bird shape and temporal resolve.** The Gaussian impostor's body/chord are
  narrowed, with CPU-defined silhouette coefficients checked against WGSL. Its
  screen axes are orthogonal: independently normalised projected 3D axes could
  become parallel, producing broad streaks and breaking Gaussian mass
  normalisation. Temporal history is shortened from 100 to 35 ms to reduce
  smearing of moving silhouettes. LOD transitions must be remeasured.
- **Sky and foreground.** `atmosphere.rs` is the CPU oracle for world-direction
  cloud layers, low-sun glow and a distant horizon. A real GPU pixel test checks
  it against `atmosphere.wgsl`. The original palette remains the base. Display
  lighting starts at evening 0.30 and maps the arc's brightness into `[0.30,1]`;
  the simulation's evening clock is unchanged. Reeds now occupy world bearings
  and elevations at three distances with tapered, bent stems, leaves and seed
  heads. They replace the repeating two-pixel screen columns, do not enter the
  flock histogram, and use the existing visibility gate. This is procedural
  scenery, not a physically validated atmosphere or vegetation model.
- **Window profiler.** `--profile-window SECONDS` runs the real surface path,
  traces CPU submission/acquisition/presentation and timestamps each entire GPU
  simulation batch. Timestamp history is mapped only after playback. FPS from
  this instrumented window is distinct from isolated headless pass timings.
- **Distant static control.** Its view now scales `tan(fov/2)`, not the angle
  itself: multiplying the new 45° minimum by four made a degenerate 180° camera
  and an invisible flock. Static comparison exposure is held fixed so exposure
  adaptation cannot be misreported as shimmer. A unit test checks the exact
  focal-length ratio and finite projection.

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
