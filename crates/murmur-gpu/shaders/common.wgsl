// Structs and helpers shared by the two shader modules of the simulation.
//
// `sim.wgsl` (the per-bird force step and flight integration) and `grid.wgsl`
// (the spatial grid that replaces the brute-force neighbour search, PLAN.md
// phase 3) are both compiled with this file and `bird.wgsl` prepended, so
// `Params`, `Bird` and the cell/bucket hash have exactly one definition. The
// Rust side mirrors `Params` and `Bird` in `params.rs` and `bird.rs` and checks
// them field for field in `tests/layout.rs`. The grid buffer bindings are
// declared in the two modules that use them, not here.

struct Params {
    dt: f32, du: f32, time: f32, turning: u32,
    v0: f32, mass: f32, g: f32, l0: f32,
    d0: f32, tau: f32, tau_roll: f32, max_roll_rate: f32,
    tau_spin: f32, max_bank: f32, v_min: f32, v_max: f32,
    n_c: f32, r_max: f32, radius_lerp: f32, r_h: f32,
    sigma: f32, w_sep: f32, w_align: f32, w_coh: f32,
    blind_cos: f32, w_xi: f32, r_roost: f32, w_roost_h: f32,
    roost_turn_margin: f32, w_roost_v: f32, h_roost_v: f32, z0: f32,
    flap_hz: f32, flap_amplitude: f32, r_threat: f32, w_flee: f32,
    r_flash: f32, v_flash: f32, w_flash: f32, flash_duration: f32,
    zigzag_max_roll: f32, zigzag_duration: f32, zigzag_cooldown: f32, reed_top: f32,
    n: u32, zigzag_k: u32, seed: u32, buckets: u32,
    inv_cell: f32, falcon_cruise: f32, falcon_climb: f32, falcon_stoop: f32,
    falcon_attack_height: f32, falcon_attack_min: f32, falcon_attack_max: f32, pad3: u32,
}

@group(0) @binding(0) var<uniform> P: Params;

// The falcon's agent state (PLAN.md phase 7). Mirrored by
// `crates/murmur-gpu/src/falcon.rs` and checked field for field in
// `tests/layout.rs`. It lives here because both `sim.wgsl` (the alarm and flee
// forces read the falcon's position) and `falcon.wgsl` (the update advances it)
// are compiled with this file prepended; the falcon's *kernels* are in
// `falcon.wgsl`. The grid module sees the declaration and ignores it.
struct Falcon {
    pos: vec3<f32>, speed: f32,
    vel: vec3<f32>, mode: u32,
    aim: vec3<f32>, state: u32,
    dive: vec3<f32>, until: f32,
    centroid: vec3<f32>, rng: u32,
    extent: f32, present: u32, pad0: u32, pad1: u32,
}

@group(0) @binding(9) var<storage, read_write> falcon: Falcon;

// The grid cell a position falls in: `floor(p / cell)`. The CPU oracle's
// `Grid::coord` uses `params::GRID_CELL = 4.0 m`; the GPU uses its own
// `grid::CELL` (`1 m`), which is only an acceleration parameter. Every query
// keeps a candidate only after an exact distance check (`d*d <= r*r`), so the
// neighbour set is the same for any cell size (PLAN.md phase 3/4). See
// `crates/murmur-gpu/src/grid.rs::CELL`.
fn cell_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p * P.inv_cell));
}

// The hash bucket of a cell, matching `Grid::hash` in the CPU oracle. The CPU
// masks to its own bucket count; only the count differs here (the GPU sizes the
// table from the bird count), which never changes which neighbours are found
// because every query also checks the exact cell.
fn bucket_of(c: vec3<i32>) -> u32 {
    let h = bitcast<u32>(c.x) * 73856093u
        ^ bitcast<u32>(c.y) * 19349663u
        ^ bitcast<u32>(c.z) * 83492791u;
    return h & (P.buckets - 1u);
}
