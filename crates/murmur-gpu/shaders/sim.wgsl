// The per-bird force step and flight integration, mirrored one function at a
// time from the CPU oracle in `crates/murmur-ref` (AGENTS.md, "The model is the
// specification"). Each function below names the reference function it mirrors
// and keeps that function's order of operations.
//
//   forward / side_of / body_up / step_bird  <- flight.rs
//   radius_step                              <- params.rs
//   decide                                   <- social.rs::decide
//   the escape fields                        <- escape.rs
//
// Phase 2 is the simplest correct version: the neighbour search inside `decide`
// is brute force, O(N^2). Phase 3 replaces it with the spatial grid.
//
// The random force is the one documented adaptation: the CPU oracle draws it
// from each bird's ChaCha8 stream, which cannot be extracted (see
// docs/model.md). The GPU keeps a counter-based PCG stream per bird keyed by
// (seed, id). Bit-exact noise parity is out of scope (PLAN.md), so the parity
// check tolerates it and the single-step test switches it off on both engines.
//
// Two ordered compute passes per step: `decide` reads the whole flock from
// `src` and writes `dst` (a faithful simultaneous snapshot, exactly like the
// reference reading `self` once); `integrate` then advances `dst` in place.
// `tick` advances the shared clock so many steps can be encoded into one
// command buffer without a queue write per step.

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
    n: u32, zigzag_k: u32, seed: u32, pad: u32,
}

struct Bird {
    pos: vec3<f32>, roll: f32,
    vel: vec3<f32>, spin: f32,
    steering: vec3<f32>, radius: f32,
    next_decision: f32, centrality: f32, flap_phase: f32, flap_amp: f32,
    neighbour_count: u32, returning: u32, id: u32, rng: u32,
    alarmed: u32, zigzag_active: u32, zigzag_sign: f32, zigzag_start: f32,
    cooldown_until: f32, flash_until: f32, pad0: u32, pad1: u32,
}

struct Clock {
    time: f32,
    step: u32,
    pad0: u32,
    pad1: u32,
}

struct Zz {
    running: bool,
    roll: f32,
}

struct StepOut {
    pos: vec3<f32>,
    vel: vec3<f32>,
    roll: f32,
    spin: f32,
}

const TAU: f32 = 6.283185307179586;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<Bird>;
@group(0) @binding(2) var<storage, read_write> dst: array<Bird>;
@group(0) @binding(3) var<storage, read_write> clock: Clock;

// --- flight.rs -------------------------------------------------------------

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

// A local stand-in for glam's Vec3::normalize_or_zero.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if l > 0.0 {
        return v / l;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// flight::step_bird. `forced` carries the zigzag override: when `running`, the
// reference's `Some(target)`.
fn step_bird(
    p0: vec3<f32>,
    v0: vec3<f32>,
    roll0: f32,
    spin0: f32,
    steering: vec3<f32>,
    forced: Zz,
    dt: f32,
    turning: u32,
) -> StepOut {
    var p = p0;
    var v = v0;
    var roll = roll0;
    var spin = spin0;

    let f = forward(v);
    let side = side_of(f);
    let speed = max(length(v), 1e-6);

    // Coordinated turn: bank toward atan(a_lat / g), with a first-order lag and
    // a maximum roll rate.
    let a_lat = dot(steering, side) / P.mass;
    var beta = roll;
    if forced.running {
        beta = clamp(forced.roll, -P.max_bank, P.max_bank);
    } else {
        let bank_target = clamp(atan2(a_lat, P.g), -P.max_bank, P.max_bank);
        let rate = clamp((bank_target - roll) / P.tau_roll, -P.max_roll_rate, P.max_roll_rate);
        if turning == 0u {
            spin = rate;
            beta = clamp(roll + rate * dt, -P.max_bank, P.max_bank);
        } else {
            spin += (rate - spin) * (dt / P.tau_spin);
            beta = clamp(roll + spin * dt, -P.max_bank, P.max_bank);
        }
    }
    roll = beta;

    let up = body_up(f, beta);

    // Lift (Eq. 16) along the body up, divided by cos β so a banked bird holds
    // its height; drag against travel; default thrust along forward.
    let cos_beta = max(cos(beta), 0.2);
    let sr = speed / P.v0;
    let lift = sr * sr * P.l0 / cos_beta;
    let drag = sr * sr * P.d0;
    let flight_force = up * lift - f * drag + f * P.d0 - vec3<f32>(0.0, 0.0, P.mass * P.g);

    // Euler integration (Eq. 22–23). Lateral steering requests a bank only.
    let control = steering - side * dot(steering, side);
    v += (control + flight_force) / P.mass * dt;
    let sp = length(v);
    if sp > P.v_max {
        v *= P.v_max / sp;
    } else if sp < P.v_min && sp > 1e-6 {
        v *= P.v_min / sp;
    }
    p += v * dt;

    return StepOut(p, v, roll, spin);
}

// --- escape.rs -------------------------------------------------------------

// escape::zigzag_roll
fn zigzag_roll(running: bool, start: f32, sign: f32, time: f32) -> Zz {
    if !running {
        return Zz(false, 0.0);
    }
    let t = time - start;
    if t < 0.0 || t >= P.zigzag_duration {
        return Zz(false, 0.0);
    }
    let phase = TAU * t / P.zigzag_duration;
    return Zz(true, sign * P.zigzag_max_roll * sin(phase));
}

// --- params.rs -------------------------------------------------------------

// params::radius_step
fn radius_step(r: f32, n: u32) -> f32 {
    let nn = f32(max(n, 1u));
    let factor = pow(P.n_c / nn, P.radius_lerp / 3.0);
    return clamp(r * factor, P.r_h, P.r_max);
}

// --- random (documented adaptation) ----------------------------------------

// One PCG step of the per-bird stream, mirroring `initial_rng` in bird.rs. The
// CPU oracle draws its random force from ChaCha8 instead; bit-exact noise
// parity is out of scope (PLAN.md), so this is kept inline where it is used.

// --- social.rs::decide -----------------------------------------------------

fn decide(i: u32, time: f32) -> Bird {
    var b = src[i];
    let n = P.n;

    let p = b.pos;
    let r = b.radius;
    let f_i = forward(b.vel);

    var sep = vec3<f32>(0.0);
    var coh = vec3<f32>(0.0);
    var align = vec3<f32>(0.0);
    var n_all = 0u;
    var n_star = 0u;
    var n_align = 0u;
    var zigzag_neighbours = 0u;
    var zigzag_sign_sum = 0.0;
    var returning_neighbours = 0u;

    // One pass over the topological neighbourhood (Eq. 3), brute force here.
    for (var j = 0u; j < n; j = j + 1u) {
        if j == i {
            continue;
        }
        let d = src[j].pos - p;
        let dist = length(d);
        if dist > r {
            continue;
        }
        n_all += 1u;
        returning_neighbours += src[j].returning;
        let dhat = d / max(dist, 1e-6);

        // Separation over all neighbours, no blind angle (Eq. 4).
        var g = 1.0;
        if dist > P.r_h {
            g = exp(-((dist - P.r_h) * (dist - P.r_h)) / (2.0 * P.sigma * P.sigma));
        }
        sep -= dhat * g;

        // Cohesion uses spatial awareness on all sides; alignment keeps the
        // rear blind sector.
        if dist > P.r_h {
            coh += dhat;
            n_star += 1u;
        }
        if dot(f_i, dhat) >= P.blind_cos {
            align += forward(src[j].vel);
            n_align += 1u;
        }

        // Neighbours whose zigzag is old enough to have been perceived.
        let age = time - src[j].zigzag_start;
        if src[j].zigzag_active != 0u && age >= P.du {
            zigzag_neighbours += 1u;
            zigzag_sign_sum += src[j].zigzag_sign;
        }
    }

    var force = vec3<f32>(0.0);
    if n_all > 0u {
        force += sep * (P.w_sep / f32(n_all));
    }

    // Centrality over a wider radius, 2R_i (Eq. 7).
    var csum = vec3<f32>(0.0);
    var n_g = 0u;
    for (var j = 0u; j < n; j = j + 1u) {
        if j == i {
            continue;
        }
        let d = src[j].pos - p;
        let dist = length(d);
        if dist > 2.0 * r {
            continue;
        }
        csum += d / max(dist, 1e-6);
        n_g += 1u;
    }
    var centrality = 0.0;
    if n_g > 0u {
        centrality = length(csum) / f32(n_g);
    }

    // Cohesion, stronger at the border through C_i (Eq. 5).
    if n_star > 0u {
        force += normalize_or_zero(coh) * (P.w_coh * centrality);
    }

    // Alignment (Eq. 8).
    var a = vec3<f32>(0.0);
    if n_align > 0u {
        a = align / f32(n_align) - f_i;
    }
    force += a * P.w_align;

    // A committed return extends Eq. 11 (Eq. 11 and the roost boundary).
    let horiz = vec3<f32>(p.x, p.y, 0.0);
    let dist_h = length(horiz);
    let near_boundary = dist_h > P.r_roost * 0.8;
    let heading_out = dist_h > 1.0 && dot(f_i, horiz / dist_h) > -0.8;
    var returning = src[i].returning != 0u
        || dist_h > P.r_roost
        || (near_boundary && heading_out && returning_neighbours >= 2u);
    if returning && dist_h > 1.0 {
        let nrm = horiz / dist_h;
        let side = side_of(f_i);
        var inward_side = side;
        if dot(side, -nrm) < 0.0 {
            inward_side = -side;
        }
        if dot(f_i, nrm) < -0.95 {
            returning = false;
        } else {
            force += inward_side * (P.w_roost_h * max(dist_h - P.r_roost, P.roost_turn_margin));
        }
    }

    // Vertical roost pull toward the preferred height (Eq. 12), capped.
    force += vec3<f32>(0.0, 0.0, P.w_roost_v * clamp((P.z0 - p.z) / P.h_roost_v, -1.0, 1.0));

    // Speed control (Eq. 1, part of Eq. 14).
    let speed = length(b.vel);
    force += f_i * ((P.mass / P.tau) * (P.v0 - speed));

    // Random force (Eq. 13). The noise is the documented adaptation: the GPU
    // draws from the bird's own counter-based stream instead of the CPU's
    // ChaCha8, so it is statistically equivalent but not bit-identical.
    var rng = b.rng;
    rng = rng * 747796405u + 2891336453u;
    var w1 = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
    w1 = (w1 >> 22u) ^ w1;
    rng = rng * 747796405u + 2891336453u;
    var w2 = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
    w2 = (w2 >> 22u) ^ w2;
    b.rng = rng;
    let z_rand = f32(w1) * (1.0 / 4294967296.0) * 2.0 - 1.0;
    let theta = f32(w2) * (1.0 / 4294967296.0) * TAU;
    let r_rand = sqrt(max(1.0 - z_rand * z_rand, 0.0));
    force += vec3<f32>(r_rand * cos(theta), r_rand * sin(theta), z_rand) * P.w_xi;

    // Flee and flash expansion (escape::force) are inert in phase 2: there is
    // no falcon, so `falcon` is None and it contributes nothing. Phase 7 adds
    // it. The copying rule below is likewise inert without an alarmed bird.
    var start_zigzag = 0.0;
    if b.alarmed == 0u
        && b.zigzag_active == 0u
        && time >= b.cooldown_until
        && zigzag_neighbours >= P.zigzag_k
    {
        if zigzag_sign_sum > 0.0 {
            start_zigzag = 1.0;
        } else if zigzag_sign_sum < 0.0 {
            start_zigzag = -1.0;
        } else {
            b.rng = b.rng * 747796405u + 2891336453u;
            var c = ((b.rng >> ((b.rng >> 28u) + 4u)) ^ b.rng) * 277803737u;
            c = (c >> 22u) ^ c;
            if (c >> 31u) == 1u {
                start_zigzag = 1.0;
            } else {
                start_zigzag = -1.0;
            }
        }
    }

    b.steering = force;
    b.radius = radius_step(r, n_all);
    b.neighbour_count = n_all;
    b.centrality = centrality;
    b.returning = u32(returning);
    if start_zigzag != 0.0 {
        b.zigzag_start = time;
        b.zigzag_active = 1u;
        b.zigzag_sign = start_zigzag;
    }
    b.next_decision += P.du;
    return b;
}

@compute @workgroup_size(256)
fn decide_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let time = clock.time;
    if time >= src[i].next_decision {
        dst[i] = decide(i, time);
    } else {
        dst[i] = src[i];
    }
}

// --- world.rs::step (flight integration) -----------------------------------

@compute @workgroup_size(256)
fn integrate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let time = clock.time;
    var b = dst[i];

    // End finished zigzags and start their cooldown.
    if b.zigzag_active != 0u && time - b.zigzag_start >= P.zigzag_duration {
        b.zigzag_active = 0u;
        b.cooldown_until = time + P.zigzag_cooldown;
    }

    let forced = zigzag_roll(b.zigzag_active != 0u, b.zigzag_start, b.zigzag_sign, time);
    let before = b.vel;
    let out = step_bird(b.pos, b.vel, b.roll, b.spin, b.steering, forced, P.dt, P.turning);

    b.pos = out.pos;
    b.vel = out.vel;
    b.roll = out.roll;
    b.spin = out.spin;

    // Advance the wingbeat and let it strengthen with acceleration.
    let accel = length(out.vel - before) / max(P.dt, 1e-6);
    b.flap_phase = (b.flap_phase + TAU * P.flap_hz * P.dt) % TAU;
    b.flap_amp = 1.0 + clamp(accel / P.g, 0.0, 1.0);

    dst[i] = b;
}

// --- clock -----------------------------------------------------------------

// Advances the shared clock once per integration step, so the two passes see
// the same `time` and many steps fit in one command buffer.
@compute @workgroup_size(1)
fn tick_kernel() {
    clock.step += 1u;
    clock.time += P.dt;
}
