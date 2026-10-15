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
  the grid.
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

## Known gaps (reported, not hidden)

Carried over from the reference and not fixed here (fixing them would change the
model the GPU mirrors):

- The flock mills over the roost rather than crossing it; the boundary force is
  radial for a tangentially moving bird, so it cannot remove the initial angular
  momentum.
- Long runs grow elongated (`I2/I1`, `I3/I1` above the paper's ranges) and can
  split.
