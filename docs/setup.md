# Setup

How to get murmuration-gpu running on the development machine: Arch Linux,
hybrid graphics (Intel UHD + NVIDIA GTX 1650 Mobile), Wayland or X11.

## Rust

Rust **1.95** or newer, from [rustup](https://rustup.rs). The workspace is a
plain `cargo` workspace:

```sh
cargo build --release
cargo test
```

## Vulkan

Simulation and rendering share one wgpu device on the Vulkan backend (wgpu
also supports Metal and DX12, but this project targets and is measured on
Vulkan).

What wgpu sees is decided by the Vulkan loader and the installed ICDs. On
this machine they are in `/usr/share/vulkan/icd.d/`:

- `nvidia_icd.json` — the proprietary driver (`nvidia-utils`)
- `intel_icd.json`, `intel_hasvk_icd.json` — Mesa

To list what the loader exposes, install the tools and check:

```sh
sudo pacman -S vulkan-tools
vulkaninfo --summary        # both GPUs, under GPU0 / GPU1
```

A software adapter (useful for headless CI) comes from `mesa-vulkan-drivers`
(llvmpipe). Without any ICD, `cargo test` still passes: the GPU test skips
with a `SKIP: no suitable GPU adapter was found` message.

## Hybrid graphics: run on the NVIDIA GPU

The laptop has two Vulkan devices. Everything windowed must run on the
discrete GPU:

```sh
prime-run cargo run --release
```

`prime-run` (package `nvidia-prime`) sets `__NV_PRIME_RENDER_OFFLOAD=1` and
`__GLX_VENDOR_LIBRARY_NAME=nvidia`. Without it the window can land on the
Intel GPU.

The app prints the adapter it actually got at startup:

```
murmuration-gpu: adapter "NVIDIA GeForce GTX 1650 with Max-Q Design" on Vulkan (DiscreteGpu), driver "NVIDIA" 615.78.08
```

**Check that line before trusting any measurement** (AGENTS.md). The device
request asks for `PowerPreference::HighPerformance`, which on this machine
selects the GTX 1650 even without `prime-run`; the printed line is still the
authority, especially on other compositors and driver versions.

Useful overrides while debugging:

```sh
WGPU_BACKEND=vulkan prime-run cargo run --release   # force the backend
WGPU_POWER_PREF=low-power cargo run --release       # ask for the iGPU instead
```

## Windowing

winit talks to Wayland and X11/XWayland. The development machine runs a
Wayland session (`WAYLAND_DISPLAY` set, `DISPLAY` for XWayland); both paths
work. If a Wayland-specific problem appears, `WAYLAND_DISPLAY= ` (empty)
forces winit onto XWayland.
