# Phase 4 coherence run — criteria (declared before the run)

The phase 4 acceptance says: *at 100,000 birds the flock keeps a coherent
display over the roost for 10 simulated minutes without collapsing or
dispersing, judged by the statistics.* This file fixes what "coherent",
"collapsing" and "dispersing" mean **before** the run is looked at, so the
verdict cannot be read back from the result.

The CPU oracle cannot simulate 100,000 birds for 10 minutes in reasonable time
(its grid is per-bird linked lists and the analysis is single-threaded), so this
check is GPU-only statistics. The run is driven by the app's `--coherence
SECONDS`, whose per-second line carries the reference fields the criteria read,
and the thresholds are checked by
[`murmur_gpu::coherence`](../crates/murmur-gpu/src/coherence.rs).

## Run

```
target/release/murmuration-gpu --coherence 600 --birds 100000 --seed 1
```

`600 s / DT = 120000` steps. The run uses the light coherence sampler, not
`--analyze`: the full reference metric set costs ~26 s per simulated second at
this size (its 15 m nearest-neighbour scan is almost all-pairs in a dense
flock), so `--analyze 600` would spend ~5 h on statistics these criteria never
read. The sampler computes exactly the fields below — the test suite proves it
bit-identical to `analysis::compute` — and `--analyze` keeps the full reference
line for comparing against the recorded 2,000-bird oracle.

**How long it really takes (corrected 2026-10-16, mid-run).** The first estimate
here said "~6 s per simulated second … so 600 s is about an hour". That was
measured on the first seconds only, and it is wrong for the run as a whole. The
dominant pass is `decide`, which walks every grid cell the bird's *adaptive*
radius `R` covers; the initial cloud is deliberately sparse and the settled flock
is several times denser, so the cost grows with the density the flock settles at.
Measured on the GTX 1650 with the light sampler: ~6 s per simulated second at
`t = 20`, ~13 s at `t = 70`, ~26 s at `t = 120`, and still rising. At that rate
600 s is **hours, not one hour**, and it is GPU-bound: the sampler is not the
bottleneck, `decide` is. The honest estimate therefore replaces the old one. The
run's own output carries the wall clock it took as a `# wall` line (added
2026-10-16, after the accepted run, whose 2 h 29 min was measured from the launch
to the file's last write).

**Why it did not merely take long: the query cost is cubic in the radius
(found 2026-10-16, after the second attempt stalled).** The estimates above
describe a smooth slow-down, but neither long attempt behaved that way: both
stopped in the same narrow window (`t ≈ 153–155`, `seed 1`) and then repeated
`# resume` repairs without advancing. A wall at a fixed simulated time is a
workload effect, not a driver timer — and the cause is the neighbour query:

- A bird's adaptive radius `R` is clamped at `params::R_MAX = 100 m`
  (`radius_step`), and the 100,000-bird flock does grow stragglers far from the
  roost (`Rh` runs to ~177 m).
- `gather_grid` walks *every cell the radius box covers*, so its cost grows with
  the cube of the radius. At the grid's `CELL = 1 m`, `R = 100 m` is
  `201³ ≈ 8.1e6` cells for **one** bird — as much as ~300,000 ordinary birds,
  which cost ~27 cells each — and the centrality walk at `2R` is worse again. A
  single straggler therefore dominates the whole step, and the model does produce
  them, so a step can go from ~0.1 s to beyond the 60 s wait budget as the tail
  grows — which is what the stalled attempts look like (their ten-step batches
  stopped finishing at `t ≈ 153`). The attribution is an inference, not a
  controlled comparison: the stalled runs predate the `Rmax` column, and the
  fixed run is a slightly different trajectory (the fallback sums a neighbour set
  in a different order). What is measured is the tail (`Rmax ≈ 17 m`,
  `Rh ≈ 190 m`) and that the cell walk is the step's only superlinear term.

The fix is in `gather` (`crates/murmur-gpu/shaders/sim.wgsl`): **when the box is
larger than the flock, walk the flock instead.** Both walks keep a bird only
after the same exact distance test, so the neighbour set is unchanged — the
choice is purely a performance one. `tests/grid.rs` forces the crossover and
checks the two walks agree; on the settled 2,000-bird flock 1 query of 2,000
already crosses it, so the fallback is real at small scale too. This bounds any
one query at `min(cells, n)` and is what makes the 100,000-bird run possible at
all.

The per-second line also reports the two quantities that drove this — `Rmax` and
the fraction of birds wider than `WIDE_RADIUS = 10 m` (`coherence.rs`) — so the
mechanism is visible in the record rather than inferred from the slow-down. In
the accepted run it is loud: `Rmax` spikes to **55.9 m at t = 152** (then
recedes), the same window the two stalled attempts died in, and 128 of 600
seconds hold a bird wide enough to cross the centrality walk's crossover
(`R > 11.35 m`), 31 of 600 wide enough to cross both (`R > 22.7 m`).

## Keeping the run alive

A run of this length needs two things the first attempts did not have, both
environment facts rather than model choices:

- **The driver hangs, and a hang must not be fatal.** The development laptop's
  NVIDIA driver hangs on a long continuous compute run — `Xid 109`
  (`CTX SWITCH TIMEOUT`) in the kernel log, a known Linux driver bug reported
  across many applications, not something these shaders provoke. Twice, after
  ~50 min and ~1 h 20 min of GPU work, the submission's fence stopped being
  signalled. The old code waited with `poll(wait_indefinitely())`, so the process
  blocked for ever and the run was lost; and even a bounded wait alone would
  only turn the loss into a crash.
- **Therefore the run repairs itself.** Every wait is bounded
  (`device::WAIT_BUDGET`, 60 s — far longer than the ten-step batch it covers),
  so a hang is reported as `device::Stalled` instead of blocking. The stepping
  loop (`analyze::each_second_resilient`) then keeps the flock's raw state at
  the last sampled second (`sim::RawState`, which — unlike the CPU-facing
  `Snapshot` — keeps each bird's own random stream), opens a fresh device with
  the same request (`device::DeviceRequest`, retried briefly while the driver
  retires the dead context) and rebuilds the flock from that state.

  This does not change the trajectory. The step is a pure function of the bird
  records plus the parameters, and the grid is derived from the positions, so a
  flock rebuilt from the last sample takes exactly the steps the uninterrupted
  run would have taken; `tests/analyze.rs::a_resumed_run_takes_the_same_steps`
  asserts the records are byte-identical after a repair. Each repair is printed
  as a `# resume …` line and counted in the run's header, so the record shows
  what happened rather than hiding it.
- **A repair also rewinds the clock.** The flock is rebuilt from the last
  sampled second, so the run's step count rewinds to that second with it. The
  first version rebuilt the flock but left the counter at the failed batch's end,
  which desynchronised the sampler from the flock (the resumed run then printed
  samples 0.1 s apart instead of whole seconds) — hence the `sampled` step count
  in `each_second_resilient`.
- **A driver that does not come back stops the run instead of looping.** If
  `MAX_RESUME_TRIES = 3` repairs in a row produce no new sample, the run returns
  `RunError::NoProgress` with the simulated time it reached. Retrying for ever
  would have thrashed silently for hours, which is worse than an honest failure
  (the flock is resumable from that second).

The criteria below are untouched by any of this: the run still samples every
whole second from 1 to 600, and the same thresholds are checked.

## How the thresholds were chosen

They come from the reference's settled values at 2,000 birds
(`docs/reference-analyze-2000-42.txt`) and from the model constants, not from
the 100,000-bird run:

- `|N|` is driven toward `N_C = 6.5` by `radius_step`; the oracle settles at
  6.4–6.9, so `[4, 12]` is a generous "still interacting" band.
- `R` settles near 1.4–1.7 m in the oracle; `[0.5, 6]` allows the larger radius
  a bigger flock may settle at without admitting a collapse.
- `v` is held near `V0 = 10 m/s`; `[7, 13]` is well inside `[V_MIN, V_MAX]`.
- `hstd`: the vertical roost pull ramps over `H_ROOST_V = 40 m`, so 40 m is the
  natural upper bound; a "sheet" is a collapse toward zero, hence the 0.3 m
  floor. `dz` is the same pull and 20 m is twice the oracle's settled 1–2 m.
- `Rh`: the roost boundary is `R_ROOST = 150 m`; the oracle runs 130–175 m while
  it mills. `2.5 × R_ROOST = 375 m` is "still over the roost" rather than
  "loose in the sky".
- `rc`, the centroid's horizontal radius, is the same story capped at 200 m.
- `cluster ≥ 90 %` every second is one flock; `≥ 98 %` settled means it is
  genuinely one connected mass, not a barely-touching pair of clouds.
- `clamp ≤ 5 %` is the paper's own "bank clamp < 2 %" target with room for a
  bigger flock; a flock pinned at ±80° is not flying coherently.

## Criteria

Every second, `t = 1 … 600`:

| # | measured | must be | meaning |
| - | -------- | ------- | ------- |
| 1 | `cluster` | `≥ 90 %` | one flock, no split |
| 2 | `spread` | `≤ 250 m` | not dispersed |
| 3 | `Rh` | `≤ 375 m` | still over the roost |
| 4 | `dz` | `≤ 20 m` | altitude held |
| 5 | `hstd` | `0.3 … 40 m` | a volume, not a sheet, not a column |
| 6 | `clamp` | `≤ 5 %` | bank not saturated |

Settled means over the last half, `t = 300 … 600`:

| # | measured | must be | meaning |
| - | -------- | ------- | ------- |
| 7 | `cluster` | `≥ 98 %` | one connected mass |
| 8 | `\|N\|` | `4 … 12` | still interacting |
| 9 | `R` | `0.5 … 6 m` | perception radius sane |
| 10 | `v` | `7 … 13 m/s` | cruising |
| 11 | `rc` | `≤ 200 m` | centred over the roost |
| 12 | `spread` | `≥ 5 m` | not collapsed to a point |

The initial transient (`t < ~60 s`) relaxes from the model's compact initial
cloud; the per-second series documents it. Only `cluster`, `spread`, `Rh`, `dz`,
`hstd` and `clamp` are required every second because those are the
does-not-blow-up guards; `|N|` and `R` legitimately start far from settled.

## Reported, never used to pass or fail

The reference's two known gaps (see `docs/model.md`) are expected here too, so
they are reported by `--analyze` and **not** part of the criteria:

- **Elongation.** `I2/I1` and `I3/I1` climb over long runs (the terminal project
  reports `I3/I1 ≈ 190` after a while and thousands on very long runs).
- **Milling.** A persistent sign of `tang` means the flock orbits the roost
  instead of crossing it.

If the flock splits or disperses, criteria 1/2/7 catch it; that is the failure
this run is looking for.

## Verdict — met (2026-10-16)

```
target/release/murmuration-gpu --coherence 600 --birds 100000 --seed 1
```

**All 13 criteria pass, 600 of 600 seconds, in one uninterrupted run.** The
series and the verdict block are in `docs/analyze-100k-600s.txt`; the thresholds
above were not touched before, during or after the run.

| | measured | required |
| - | -------- | -------- |
| `cluster` every second (min) | 98.3 % | ≥ 90 % |
| `spread` every second (max) | 41.1 m | ≤ 250 m |
| `Rh` every second (max) | 215.9 m | ≤ 375 m |
| `dz` every second (max) | 9.6 m | ≤ 20 m |
| `hstd` every second | 3.8 – 11.4 m | 0.3 – 40 m |
| `clamp` every second (max) | 0.0 % | ≤ 5 % |
| `cluster` settled (t = 300…600) | 98.52 % | ≥ 98 % |
| `\|N\|` settled | 6.68 | 4 – 12 |
| `R` settled | 0.85 m | 0.5 – 6 m |
| `v` settled | 10.0 m/s | 7 – 13 m/s |
| `rc` settled | 100.3 m | ≤ 200 m |
| `spread` settled | 21.5 m | ≥ 5 m |

The verdict was re-derived independently from the 600 printed lines (minima,
maxima and settled means recomputed with `awk` from the series, not read from the
checker's table) and agrees on every criterion. The flock neither collapsed nor
dispersed: it stayed one connected mass (`cluster` never below 98.3 %), held its
cruising speed at `V0`, kept its altitude, and milled over the roost inside the
boundary.

Run facts, for the record:

- **Zero repairs.** The driver did not hang once, so the resilience machinery in
  "Keeping the run alive" was not exercised in the accepted run; it remains the
  safety net that makes a run of this length safe to launch. (It *was* exercised
  during development, and `tests/analyze.rs` covers resuming.)
- **2 h 29 min of wall clock** (13:31 → 16:00) for 600 s of simulation, ~15 s per
  simulated second on average, on the GTX 1650 with the light sampler.
- **No `Xid`** in the kernel log for this run.
- `Rmax` reached **55.9 m** of the model's `R_MAX = 100 m`; the tail is real and
  the run survives it because the query is bounded (see the cubic-cost section
  above).
