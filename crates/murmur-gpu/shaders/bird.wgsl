// The GPU bird record and the flight-frame helpers, shared by the simulation
// (`sim.wgsl`), the spatial grid (`grid.wgsl`) and the renderer's coverage pass
// (`crates/murmur-render/shaders/coverage.wgsl`).
//
// This file has no bindings and no `Params`: it is prepended (in Rust, with
// `format!`) to every module that reads a bird, so `Bird`, `forward`, `side_of`
// and `body_up` have exactly one definition across the two crates. The Rust
// mirror (`crates/murmur-gpu/src/bird.rs`) is checked against `struct Bird` here
// by `crates/murmur-gpu/tests/layout.rs`.

struct Bird {
    pos: vec3<f32>, roll: f32,
    vel: vec3<f32>, spin: f32,
    steering: vec3<f32>, radius: f32,
    next_decision: f32, centrality: f32, flap_phase: f32, flap_amp: f32,
    neighbour_count: u32, returning: u32, id: u32, rng: u32,
    alarmed: u32, zigzag_active: u32, zigzag_sign: f32, zigzag_start: f32,
    cooldown_until: f32, flash_until: f32, pad0: u32, pad1: u32,
}

// --- flight.rs: the body frame --------------------------------------------

// flight::forward
fn forward(v: vec3<f32>) -> vec3<f32> {
    if dot(v, v) > 1e-12 {
        return normalize(v);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// flight::side_of
fn side_of(f: vec3<f32>) -> vec3<f32> {
    let s = cross(f, vec3<f32>(0.0, 0.0, 1.0));
    if dot(s, s) > 1e-8 {
        return normalize(s);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// flight::body_up
fn body_up(f: vec3<f32>, beta: f32) -> vec3<f32> {
    let side = side_of(f);
    let up0 = cross(side, f);
    return up0 * cos(beta) + side * sin(beta);
}
