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

## Phase 1 — reference model as oracle (blocked on a decision)

**The check PLAN.md asks for: is the simulation usable as a library,
without the terminal code?**

Answer: the simulation is exposed and pure, but it cannot be built without
the terminal code. Details (murmuration 0.2.0; the local checkout is the
published source plus two README-only commits, and crates.io confirms the
published crate declares `features: {}`):

- Yes: `murmuration::sim` is a public module (`World`, `params`, `social`,
  `flight`, `neighbours`, `escape`, `falcon`, `arc`, `turns`, `analysis`),
  it never references `crate::cli` or `crate::render` (verified by search),
  and it is deterministic for a seed. `main.rs` itself drives it through
  exactly this API for `--analyze`.
- No: `ratatui`, `crossterm`, `clap` and `signal-hook` are unconditional
  dependencies, so depending on the crate compiles the whole TUI stack.
  `sim::flight::Turning` even derives `clap::ValueEnum` — the sim itself
  references a CLI crate. All modules are `#[doc(hidden)]` with a "no
  stability guarantee" note, and the published crate has no features.

**Baseline recorded (the acceptance oracle)**

`murmuration --analyze 120 --birds 2000 --seed 42` (release, this machine):
16.3 s wall for 120 simulated seconds. Full per-second output:
`docs/reference-analyze-2000-42.txt`. Averaged over t = 30–120 s:

| statistic | value |
| --- | --- |
| mean topological neighbours \|N\| | 6.62 |
| mean nearest-neighbour distance | 0.77 m |
| horizontal flock-plane alignment | 0.90 |
| nearest-neighbour ahead/behind fraction | 0.095 |
| height standard deviation | 2.06 m |
| largest single-link cluster (5 m) | ≥ 99.9 % |
| mean speed | 10.0 m/s |
| inertia ratios I2/I1, I3/I1 | 3.43, 32.4 |

These match the "current measurements" table in murmuration's
DEVELOPMENT.md. The phase 1 acceptance run must reproduce them.

**Proposal (not applied): the smallest change to murmuration**

Gate the terminal side behind a default `tui` feature. About 15 changed
lines; the default build is unchanged, and the simulation builds alone
with `default-features = false`:

```toml
# Cargo.toml
[features]
# The terminal UI, its CLI and its signal handling. The simulation
# (sim::) needs none of these; depend on murmuration with
# default-features = false to use only the model.
default = ["tui"]
tui = ["dep:ratatui", "dep:crossterm", "dep:clap", "dep:signal-hook"]

[dependencies]
ratatui = { version = "0.30", optional = true }
crossterm = { version = "0.29", optional = true }
clap = { version = "4", features = ["derive"], optional = true }
signal-hook = { version = "0.3", optional = true }
# rand, rand_chacha, glam, rayon stay non-optional: sim needs them.

[[bin]]
name = "murmuration"
required-features = ["tui"] # --no-default-features builds the lib only
```

```rust
// src/lib.rs
pub mod sim; // always built

#[cfg(feature = "tui")]
pub mod cli;
#[cfg(feature = "tui")]
pub mod render;
```

```rust
// src/sim/flight.rs — the only place sim references clap
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "tui", derive(clap::ValueEnum))]
pub enum Turning {
```

```yaml
# .github/workflows/ci.yml — one extra line so the lib-only build cannot rot
- run: cargo test -p murmuration --lib --no-default-features
```

Nothing else changes: the binary, the `xtask` and the render tests all run
with the default features, exactly as today. If this lands, `murmur-ref`
depends on the crate — from a checkout
(`murmuration = { path = "../murmuration", default-features = false }`)
or from crates.io once released — and reuses `sim` and `sim::analysis`
wholesale, which keeps one source of truth for the model.

Optional cosmetic follow-up, left to the maintainer: un-hide `pub mod sim`
and soften the "no stability guarantee" note, since it would now be a
deliberate library surface.

**Blocked on**

The maintainer's answer: apply the feature gate upstream and depend on the
crate, or vendor `sim/` into `murmur-ref` with attribution and strip the
terminal code. Until then the phase 1 work (statistics wiring,
`export`/`import`, the headless acceptance run) is on hold, per instruction;
nothing has been vendored.
