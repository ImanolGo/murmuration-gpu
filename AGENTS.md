# AGENTS.md — rules for working on murmuration-gpu

Read this and [PLAN.md](PLAN.md) before writing code.

## How to work

- Follow PLAN.md phase by phase. Do not start a phase until the previous
  one's acceptance criteria pass.
- Keep a short log at the end of each phase in `docs/progress.md`: what was
  done, what was measured, what is still open.
- When the plan and reality disagree (an API changed, a target is
  unrealistic), write down the change and the reason in `docs/progress.md`
  and update PLAN.md. Do not quietly drift.

## The model is the specification

- The CPU model in `murmur-ref` is the source of truth for behaviour. GPU
  code mirrors its functions, names and order of operations.
- Never tune a GPU parameter to make it look better without the same change
  in the reference, and a note explaining it.
- Every documented adaptation or deviation from the published model goes into
  `docs/model.md`, as the terminal version does in `DEVELOPMENT.md`.

## Honesty about results

- Report performance only from GPU timestamp queries or a profiler, with the
  GPU name, driver, bird count and resolution.
- Don't claim a behaviour (dark waves, splits, vacuoles) is reproduced until
  it has been seen in a recording or measured by a statistic.
- If a test is skipped because no GPU is available, say so in the output.

## Code standards

- Rust 1.95+. `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and
  `cargo test` must pass before a phase is closed.
- `Params` and `Bird` are defined once in Rust and checked against WGSL by a
  test (size, alignment, field offsets).
- Shaders live in `crates/*/shaders/*.wgsl`, one purpose per file, with a
  header comment saying which reference function each one mirrors.
- No `unsafe` unless a crate's API requires it, with a comment why.
- Use the latest stable `wgpu` and `winit`; check their docs and changelogs
  for the current API instead of relying on memory.

## Running on the development machine

```
prime-run cargo run --release                     # windowed
cargo test                                        # headless, GPU tests if available
cargo run --release -- --bench --birds 1000000    # timings
cargo run --release -- --parity --birds 2000 --seed 42
```

If the window opens on the integrated GPU, the app prints the adapter name
at startup. Check it before reporting any measurement.
