// The evening arc (PLAN.md phase 8), mirrored from `murmur-ref/src/arc.rs`.
//
// Two kernels per step while an evening is running: `arc_kernel` (one
// invocation) advances the evening clock, sets the roost height for the phase
// and spawns an arriving group into the next free slots; `landed_kernel` (one
// invocation per bird) retires the birds that have landed during the roosting
// phase, and clears the flock at night.
//
// The flock buffer is bound here as read-write (binding 10) instead of the sim
// module's read-only `src`, because the arc must *add* birds in place; the
// buffer is otherwise the same one, so a bird spawned this step is seen by the
// grid rebuild, the decisions and the integration that follow, exactly as
// `world.rs::step` spawns before it decides.
//
// One documented adaptation: the random stream. The oracle draws from ChaCha8,
// which cannot run in WGSL; this uses the same counter-based PCG as the birds,
// seeded from the run's seed (`arc.rs` seeds it identically).

// This file is appended to the simulation module, so `P`, `clock` and `arc` are
// already declared; only the flock is bound differently here, read-write instead
// of the sim's read-only `src`, because the arc adds and retires birds.
@group(0) @binding(10) var<storage, read_write> flock: array<Bird>;

// The phase boundaries and the arc's own constants (`arc.rs`).
const ARRIVAL_END: f32 = 0.15;
const DISPLAY_END: f32 = 0.70;
const ROOSTING_END: f32 = 0.90;

// The next uniform in [0, 1) from the arc's counter-based stream, matching
// `next_uniform` in `crates/murmur-gpu/src/arc.rs`.
fn next_uniform() -> f32 {
    let r = arc.rng * 747796405u + 2891336453u;
    var w = ((r >> ((r >> 28u) + 4u)) ^ r) * 277803737u;
    w = (w >> 22u) ^ w;
    arc.rng = r;
    return f32(w) * (1.0 / 4294967296.0);
}

fn uniform_range(lo: f32, hi: f32) -> f32 {
    return lo + next_uniform() * (hi - lo);
}

// social::random_unit, from the arc's stream.
fn random_unit() -> vec3<f32> {
    let z = uniform_range(-1.0, 1.0);
    let theta = uniform_range(0.0, TAU);
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(theta), r * sin(theta), z);
}

// The initial random-stream state for slot `i`, the same mix `bird.rs` uses so
// an arriving bird's noise stream is like a seeded bird's.
fn slot_rng(i: u32) -> u32 {
    var h = P.seed ^ 0x9E3779B9u;
    h = h ^ (i * 0x85EBCA6Bu);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    return h ^ (h >> 16u);
}

// world.rs::push_bird: a fresh bird, inactive fields at their defaults.
fn new_bird(pos: vec3<f32>, vel: vec3<f32>, slot: u32, perception: f32) -> Bird {
    var b: Bird;
    b.pos = pos;
    b.roll = 0.0;
    b.vel = vel;
    b.spin = 0.0;
    b.steering = vec3<f32>(0.0);
    b.radius = perception;
    b.next_decision = clock.time + uniform_range(0.0, P.du);
    b.centrality = 0.0;
    b.flap_phase = uniform_range(0.0, TAU);
    b.flap_amp = 1.0;
    b.neighbour_count = 0u;
    b.returning = 0u;
    b.id = slot;
    b.rng = slot_rng(slot);
    b.alarmed = 0u;
    b.zigzag_active = 0u;
    b.zigzag_sign = 1.0;
    b.zigzag_start = -1e30;
    b.cooldown_until = -1e30;
    b.flash_until = -1e30;
    b.live = 1u;
    b.pad1 = 0u;
    return b;
}

// arc.rs::spawn_group: continuous flux through a shared grazing approach.
fn spawn_group(count: u32) {
    let outward = vec3<f32>(cos(arc.bearing), sin(arc.bearing), 0.0);
    let tangent = vec3<f32>(-outward.y, outward.x, 0.0);
    let heading = -outward * 0.6 + tangent * 0.8;
    let side = cross(heading, vec3<f32>(0.0, 0.0, 1.0));
    let base = outward * arc.radius + vec3<f32>(0.0, 0.0, P.z0);
    let span = min(arc.duration * ARRIVAL_END, 6.0);
    let spread = max(sqrt(f32(P.n) / (P.v0 * span * (TAU * 0.5) * 0.5)), 3.0);
    let u = clamp((clock.time - f32(arc.epoch) * arc.duration) / span, 0.0, 1.0);
    let flux = f32(P.n) * 6.0 * u * (1.0 - u) / span;
    let density = flux / (P.v0 * (TAU * 0.5) * spread * spread * 0.6);
    let perception = clamp(pow(P.n_c / ((4.0 / 3.0) * (TAU * 0.5) * max(density, 0.05)), 1.0 / 3.0), P.r_h, 5.0);
    for (var k = 0u; k < count; k = k + 1u) {
        let slot = arc.next + k;
        if slot >= P.n {
            break;
        }
        let angle = uniform_range(0.0, TAU);
        let r = spread * sqrt(next_uniform());
        let pos = base + side * (r * cos(angle))
            + vec3<f32>(0.0, 0.0, r * sin(angle) * 0.6)
            + heading * uniform_range(-0.5, 0.5);
        var dir = heading
            + vec3<f32>(
                uniform_range(-0.05, 0.05),
                uniform_range(-0.05, 0.05),
                uniform_range(-0.025, 0.025),
            );
        dir = normalize_or_zero(dir);
        flock[slot] = new_bird(pos, dir * P.v0, slot, perception);
    }
    arc.next = min(arc.next + count, P.n);
}

// arc.rs::Arc::update. A single invocation: the flock's own passes follow.
@compute @workgroup_size(1)
fn arc_kernel() {
    if arc.present == 0u {
        return;
    }
    let time = clock.time;
    let epoch = u32(time / arc.duration);
    if epoch != arc.epoch {
        // A new evening: the flock is cleared (night did that on the last step
        // of the previous evening), and the approach corridor is redrawn.
        arc.epoch = epoch;
        arc.next = 0u;
        arc.spawn_timer = 0.0;
        arc.bearing = uniform_range(0.0, TAU);
        arc.radius = uniform_range(180.0, 200.0);
    }
    let f = time / arc.duration - f32(epoch);

    if f < ARRIVAL_END {
        arc.z0 = P.z0;
        let span = min(arc.duration * ARRIVAL_END, 6.0);
        let u = clamp((time - f32(epoch) * arc.duration) / span, 0.0, 1.0);
        let wanted = min(u32(f32(P.n) * u * u * (3.0 - 2.0 * u)), P.n);
        if wanted > arc.next {
            spawn_group(wanted - arc.next);
        }
    } else if f < DISPLAY_END {
        // Display: the flock holds the roost height.
        arc.z0 = P.z0;
        if arc.next < P.n {
            spawn_group(P.n - arc.next);
        }
    } else if f < ROOSTING_END {
        // Roosting: the roost level sinks to the reed tops as the flock lands.
        let g = clamp((f - DISPLAY_END) / (ROOSTING_END - DISPLAY_END), 0.0, 1.0);
        arc.z0 = P.z0 + (P.reed_top - P.z0) * g;
    } else {
        arc.z0 = P.reed_top;
    }
}

// world.rs::remove_landed, and `clear` at night: the bird is retired in place
// rather than removed, so the slots stay where they are.
@compute @workgroup_size(256)
fn landed_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n || arc.present == 0u {
        return;
    }
    if flock[i].live == 0u {
        return;
    }
    let f = clock.time / arc.duration - f32(arc.epoch);
    if f >= ROOSTING_END {
        flock[i].live = 0u;
        return;
    }
    if f >= DISPLAY_END {
        let p = flock[i].pos;
        let horizontal = sqrt(p.x * p.x + p.y * p.y);
        if p.z < P.reed_top && horizontal < P.r_roost {
            flock[i].live = 0u;
        }
    }
}
