# murmuration-gpu

Starlings at dusk, on your GPU: a research-based murmuration of up to millions
of birds, simulated and rendered in real time. Written in Rust with wgpu.

murmuration-gpu is the big-sky sibling of
[murmuration](https://github.com/ImanolGo/murmuration). The terminal version
draws a few thousand starlings in braille. This version keeps the same flocking
model and moves it to GPU compute shaders. Each bird is a small flapping,
banking body, and the flock is drawn as a dark, grainy silhouette against an
evening sky, the way a real murmuration looks from the ground.

The shapes come from individual birds following local rules. There is no
prerecorded animation, no choreographed outline and no leader.

> **Status:** early development. See [PLAN.md](PLAN.md) for the roadmap and
> [Limitations](#limitations) for what this is and is not.

## Goals

- **The real model, not generic boids.** Topological attention (about seven
  neighbours through an adaptive radius), boundary-weighted cohesion, banked
  flight, cruise-speed control, roost attraction and predator responses,
  following StarDisplay (Hildenbrandt, Carere & Hemelrijk, 2010) and the
  starling field studies.
- **Scale.** From tens of thousands of birds up to millions, with the
  neighbour search on the GPU through a sorted uniform grid.
- **Birds, not particles.** Instanced low-poly birds with wingbeats and
  banking near the camera, flapping impostors further out, and sub-pixel
  coverage in the distance.
- **The look of the photographs.** Overlapping birds darken multiplicatively,
  wing exposure drives the travelling dark waves, and a physically inspired
  dusk sky sits behind them.
- **Checked against the reference.** The CPU model from `murmuration` is
  kept as an oracle, and the GPU simulation is compared against it
  statistically.

## Requirements

- Rust **1.95** or newer.
- A GPU with Vulkan, Metal or DX12 support. Development targets Linux and
  Vulkan on a modest laptop GPU (GTX 1650 Mobile, 4 GB).

On hybrid-graphics laptops (NVIDIA Optimus), run on the discrete GPU:

```
prime-run cargo run -p murmuration-gpu --release
```

Unlike a CUDA and OpenGL setup, simulation and rendering share one wgpu
device, so there is no interop step to pair across GPUs.

## Getting started

```
cargo run -p murmuration-gpu --release                          # a dusk flock with the default bird count
cargo run -p murmuration-gpu --release -- --birds 1000000       # one million birds
cargo run -p murmuration-gpu --release -- --falcon              # add an autonomous falcon
cargo run -p murmuration-gpu --release -- --seed 42             # a repeatable start
```

Headless tools, without opening a window:

```
cargo run -p murmuration-gpu --release -- --bench --birds 1000000          # GPU timings per pass
cargo run -p murmuration-gpu --release -- --parity --birds 2000 --seed 42  # GPU vs CPU reference statistics
cargo run -p murmuration-gpu --release -- --analyze 120 --birds 50000      # per-second flock statistics
```

Planned controls:

| Input          | Action                                  |
| -------------- | --------------------------------------- |
| `q` / `Esc`    | Quit                                    |
| `Space`        | Pause or resume                         |
| Mouse drag     | Orbit the observer around the roost     |
| `f`            | Spawn an autonomous falcon              |
| Move the mouse | Steer the falcon                        |
| `+` / `-`      | Change the flock size                   |
| `h`            | Show or hide the overlay and timings    |

## How it works

Each frame runs a fixed-step simulation, then a render pass, on one GPU:

```
┌──────────────────────────────── GPU ────────────────────────────────┐
│                                                                     │
│  simulate (fixed step, compute)                                     │
│    1. hash birds into a uniform grid                                │
│    2. counting sort by cell, reorder bird state for locality        │
│    3. per bird: scan nearby cells within its adaptive radius,       │
│       keep the ~7 nearest, accumulate social forces                 │
│    4. banked flight, speed control, altitude, roost, predator       │
│    5. integrate into the next state buffer                          │
│                                                                     │
│  render                                                             │
│    6. dusk sky                                                      │
│    7. birds by projected size: meshes, impostors, coverage points   │
│    8. multiplicative darkness accumulation, no sorting needed       │
│    9. temporal anti-aliasing, tone map                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The bird state buffer that the simulation writes is the same buffer the
render pass reads. Nothing is copied back to the CPU during normal running.

### Workspace

```
crates/
  murmur-ref/      CPU reference model and flock statistics (the oracle)
  murmur-gpu/      wgpu compute: grid, sort, neighbours, forces, integration
  murmur-render/   sky, bird LOD, darkness accumulation, TAA, tone mapping
app/               the murmuration-gpu binary: window, camera, input, CLI
```

## Limitations

This is a research-inspired visual simulation, not a validated predictor of
wild flocks. The same caveats as the terminal version apply, plus:

- Nearest-neighbour search is approximate: each bird searches within its own
  adaptive radius rather than doing an exact global k-nearest search. This
  matches how the model works but may differ in edge cases.
- GPU results are not bit-identical to the CPU reference. A seed gives a
  repeatable start, and parity is checked with flock statistics rather than
  frame by frame.
- Millions of birds is beyond the size of most real murmurations. Large
  counts are an artistic choice.
- Performance figures are only reported when measured with GPU timestamps on
  named hardware.

## References

See the [references in murmuration](https://github.com/ImanolGo/murmuration#references).
The main ones for this project:

- Hildenbrandt, Carere & Hemelrijk (2010). Self-organized aerial displays of
  thousands of starlings: a model. *Behavioral Ecology* 21(6).
- Ballerini et al. (2008). Interaction ruling animal collective behavior
  depends on topological rather than metric distance. *PNAS* 105(4).
- Hemelrijk, van Zuidam & Hildenbrandt (2015). What underlies waves of
  agitation in starling flocks. *Behav. Ecol. Sociobiol.* 69(5).
- Costanzo, Hildenbrandt & Hemelrijk (2022). Causes of variation of darkness
  in flocks of starlings, a computational model. *Swarm Intelligence* 16.
- Green, S. (2010). Particle simulation using CUDA (uniform grid with
  sorting). NVIDIA whitepaper.

## License

MIT
