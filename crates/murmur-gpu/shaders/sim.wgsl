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
// The neighbour search inside `decide` is the spatial grid built by
// `grid.wgsl`: each bird walks the cells its adaptive radius covers and keeps
// every bird inside it, exactly the set the CPU oracle's `Grid::for_each_within`
// returns. `decide_bruteforce_kernel` is the same decision with an O(N^2) scan
// over all birds; it is kept so the phase 3 acceptance test can compare the two
// neighbour sets, and is not used by the sim loop.
//
// The random force is the one documented adaptation: the CPU oracle draws it
// from each bird's ChaCha8 stream, which cannot be extracted (see
// docs/model.md). The GPU keeps a counter-based PCG stream per bird keyed by
// (seed, id). Bit-exact noise parity is out of scope (PLAN.md), so the parity
// check tolerates it and the single-step test switches it off on both engines.
//
// One step is: `tick`, the grid build (see grid.wgsl), `decide`, which reads the
// whole flock from `src` and writes `dst` (a faithful simultaneous snapshot,
// exactly like the reference reading `self` once), then `integrate`, which
// advances `dst` in place.

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

// The neighbourhood accumulators of `social::decide`, gathered in one walk.
struct Acc {
    sep: vec3<f32>,
    coh: vec3<f32>,
    align: vec3<f32>,
    n_all: u32,
    n_star: u32,
    n_align: u32,
    zigzag_neighbours: u32,
    zigzag_sign_sum: f32,
    returning_neighbours: u32,
}

// The centrality accumulator: a wider walk, position only.
struct Cen {
    csum: vec3<f32>,
    n_g: u32,
}

const TAU: f32 = 6.283185307179586;

@group(0) @binding(1) var<storage, read> src: array<Bird>;
@group(0) @binding(2) var<storage, read_write> dst: array<Bird>;
@group(0) @binding(3) var<storage, read_write> clock: Clock;
@group(0) @binding(4) var<storage, read_write> bucket: array<u32>;
@group(0) @binding(6) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> sorted: array<u32>;

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

// --- the two neighbourhood walks -------------------------------------------
//
// `Acc` and `Cen` are gathered here, so `decide` is the same whichever walk
// filled them. The accumulation order inside `acc_add` is `social::decide`'s.

fn acc_empty() -> Acc {
    return Acc(
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0),
        0u, 0u, 0u, 0u, 0.0, 0u,
    );
}

fn acc_add(acc: Acc, other: Bird, d: vec3<f32>, dist: f32, f_i: vec3<f32>, time: f32) -> Acc {
    var a = acc;
    a.n_all += 1u;
    a.returning_neighbours += other.returning;
    let dhat = d / max(dist, 1e-6);

    // Separation over all neighbours, no blind angle (Eq. 4).
    var g = 1.0;
    if dist > P.r_h {
        g = exp(-((dist - P.r_h) * (dist - P.r_h)) / (2.0 * P.sigma * P.sigma));
    }
    a.sep -= dhat * g;

    // Cohesion uses spatial awareness on all sides; alignment keeps the rear
    // blind sector.
    if dist > P.r_h {
        a.coh += dhat;
        a.n_star += 1u;
    }
    if dot(f_i, dhat) >= P.blind_cos {
        a.align += forward(other.vel);
        a.n_align += 1u;
    }

    // Neighbours whose zigzag is old enough to have been perceived.
    let age = time - other.zigzag_start;
    if other.zigzag_active != 0u && age >= P.du {
        a.zigzag_neighbours += 1u;
        a.zigzag_sign_sum += other.zigzag_sign;
    }
    return a;
}

fn cen_empty() -> Cen {
    return Cen(vec3<f32>(0.0), 0u);
}

fn cen_add(cen: Cen, d: vec3<f32>, dist: f32) -> Cen {
    var c = cen;
    c.csum += d / max(dist, 1e-6);
    c.n_g += 1u;
    return c;
}

// Walks the cells the radius covers and keeps every bird inside it, the same
// set `Grid::for_each_within` visits. A bucket can hold birds from several cells
// (the table is smaller than the world), so each candidate is kept only when its
// own cell matches the one being queried; that also stops a bird being counted
// twice when two query cells hash to the same bucket.
fn gather_grid(p: vec3<f32>, radius: f32, i: u32, f_i: vec3<f32>, time: f32) -> Acc {
    var acc = acc_empty();
    let lo = cell_of(p - vec3<f32>(radius));
    let hi = cell_of(p + vec3<f32>(radius));
    let r2 = radius * radius;
    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
                let cell = vec3<i32>(cx, cy, cz);
                let b = bucket_of(cell);
                let end = offsets[b + 1u];
                for (var s = offsets[b]; s < end; s = s + 1u) {
                    let j = sorted[s];
                    if j == i {
                        continue;
                    }
                    let other = src[j];
                    if any(cell_of(other.pos) != cell) {
                        continue;
                    }
                    let d = other.pos - p;
                    let d2 = dot(d, d);
                    if d2 <= r2 {
                        acc = acc_add(acc, other, d, sqrt(d2), f_i, time);
                    }
                }
            }
        }
    }
    return acc;
}

// The same walk over the 2R_i box, for centrality (Eq. 7).
fn gather_grid_centrality(p: vec3<f32>, radius: f32, i: u32) -> Cen {
    var cen = cen_empty();
    let lo = cell_of(p - vec3<f32>(radius));
    let hi = cell_of(p + vec3<f32>(radius));
    let r2 = radius * radius;
    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
                let cell = vec3<i32>(cx, cy, cz);
                let b = bucket_of(cell);
                let end = offsets[b + 1u];
                for (var s = offsets[b]; s < end; s = s + 1u) {
                    let j = sorted[s];
                    if j == i {
                        continue;
                    }
                    let other = src[j];
                    if any(cell_of(other.pos) != cell) {
                        continue;
                    }
                    let d = other.pos - p;
                    let d2 = dot(d, d);
                    if d2 <= r2 {
                        cen = cen_add(cen, d, sqrt(d2));
                    }
                }
            }
        }
    }
    return cen;
}

// The phase 2 walk: every other bird, in order. Kept for the acceptance test.
fn gather_bruteforce(p: vec3<f32>, radius: f32, i: u32, f_i: vec3<f32>, time: f32) -> Acc {
    var acc = acc_empty();
    let n = P.n;
    let r2 = radius * radius;
    for (var j = 0u; j < n; j = j + 1u) {
        if j == i {
            continue;
        }
        let other = src[j];
        let d = other.pos - p;
        let d2 = dot(d, d);
        if d2 <= r2 {
            acc = acc_add(acc, other, d, sqrt(d2), f_i, time);
        }
    }
    return acc;
}

fn gather_bruteforce_centrality(p: vec3<f32>, radius: f32, i: u32) -> Cen {
    var cen = cen_empty();
    let n = P.n;
    let r2 = radius * radius;
    for (var j = 0u; j < n; j = j + 1u) {
        if j == i {
            continue;
        }
        let d = src[j].pos - p;
        let d2 = dot(d, d);
        if d2 <= r2 {
            cen = cen_add(cen, d, sqrt(d2));
        }
    }
    return cen;
}

// --- social.rs::decide -----------------------------------------------------

// The decision itself, once a walk has filled the accumulators (the body of
// `social::decide` from the cohesion force on).
fn resolve(b_in: Bird, time: f32, r: f32, f_i: vec3<f32>, acc: Acc, cen: Cen) -> Bird {
    var b = b_in;
    let p = b.pos;

    var force = vec3<f32>(0.0);
    if acc.n_all > 0u {
        force += acc.sep * (P.w_sep / f32(acc.n_all));
    }

    var centrality = 0.0;
    if cen.n_g > 0u {
        centrality = length(cen.csum) / f32(cen.n_g);
    }

    // Cohesion, stronger at the border through C_i (Eq. 5).
    if acc.n_star > 0u {
        force += normalize_or_zero(acc.coh) * (P.w_coh * centrality);
    }

    // Alignment (Eq. 8).
    var a = vec3<f32>(0.0);
    if acc.n_align > 0u {
        a = acc.align / f32(acc.n_align) - f_i;
    }
    force += a * P.w_align;

    // A committed return extends Eq. 11 (Eq. 11 and the roost boundary).
    let horiz = vec3<f32>(p.x, p.y, 0.0);
    let dist_h = length(horiz);
    let near_boundary = dist_h > P.r_roost * 0.8;
    let heading_out = dist_h > 1.0 && dot(f_i, horiz / dist_h) > -0.8;
    var returning = b.returning != 0u
        || dist_h > P.r_roost
        || (near_boundary && heading_out && acc.returning_neighbours >= 2u);
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

    // Flee and flash expansion (escape::force) are inert before phase 7: there
    // is no falcon, so `falcon` is None and it contributes nothing. The copying
    // rule below is likewise inert without an alarmed bird.
    var start_zigzag = 0.0;
    if b.alarmed == 0u
        && b.zigzag_active == 0u
        && time >= b.cooldown_until
        && acc.zigzag_neighbours >= P.zigzag_k
    {
        if acc.zigzag_sign_sum > 0.0 {
            start_zigzag = 1.0;
        } else if acc.zigzag_sign_sum < 0.0 {
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
    b.radius = radius_step(r, acc.n_all);
    b.neighbour_count = acc.n_all;
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

// The decided bird, from the grid walk.
fn decide(i: u32, time: f32) -> Bird {
    let b = src[i];
    let f_i = forward(b.vel);
    let acc = gather_grid(b.pos, b.radius, i, f_i, time);
    let cen = gather_grid_centrality(b.pos, 2.0 * b.radius, i);
    return resolve(b, time, b.radius, f_i, acc, cen);
}

// The same decision from the brute-force walk (phase 3 acceptance test only).
fn decide_bruteforce(i: u32, time: f32) -> Bird {
    let b = src[i];
    let f_i = forward(b.vel);
    let acc = gather_bruteforce(b.pos, b.radius, i, f_i, time);
    let cen = gather_bruteforce_centrality(b.pos, 2.0 * b.radius, i);
    return resolve(b, time, b.radius, f_i, acc, cen);
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

@compute @workgroup_size(256)
fn decide_bruteforce_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let time = clock.time;
    if time >= src[i].next_decision {
        dst[i] = decide_bruteforce(i, time);
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

// Advances the shared clock once per integration step, so the passes see the
// same `time` and many steps fit in one command buffer.
@compute @workgroup_size(1)
fn tick_kernel() {
    clock.step += 1u;
    clock.time += P.dt;
}
