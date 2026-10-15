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
