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
@group(0) @binding(11) var<storage, read_write> arc: ArcState;
@group(0) @binding(4) var<storage, read_write> bucket: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> sorted: array<u32>;

// --- flight.rs -------------------------------------------------------------
//
// `forward`, `side_of` and `body_up` live in `bird.wgsl` (shared with the grid
// and the renderer). Only the step and the helpers specific to the simulation
// are defined here.

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
fn cell_intersects_sphere(cell: vec3<i32>, p: vec3<f32>, r2: f32) -> bool {
    let size = 1.0 / P.inv_cell;
    let lo = vec3<f32>(cell) * size;
    let nearest = clamp(p, lo, lo + vec3<f32>(size));
    let delta = nearest - p;
    // Conservative padding at cell faces: reject only cells certainly outside.
    return dot(delta, delta) <= r2 + 1e-4;
}

fn gather_grid(p: vec3<f32>, radius: f32, i: u32, f_i: vec3<f32>, time: f32) -> Acc {
    var acc = acc_empty();
    let lo = cell_of(p - vec3<f32>(radius));
    let hi = cell_of(p + vec3<f32>(radius));
    let r2 = radius * radius;
    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
                let cell = vec3<i32>(cx, cy, cz);
                if !cell_intersects_sphere(cell, p, r2) {
                    continue;
                }
                let b = bucket_of(cell);
                let end = offsets[b + 1u];
                for (var s = offsets[b]; s < end; s = s + 1u) {
                    let j = sorted[s];
                    if j == i {
                        continue;
                    }
                    // `sorted` only holds live birds (grid_histogram skips the
                    // rest), so no `live` test. Read the 12-byte position first
                    // and the 96-byte record only for a kept neighbour: most
                    // candidates in the walked cells are rejected.
                    let q = src[j].pos;
                    if any(cell_of(q) != cell) {
                        continue;
                    }
                    let d = q - p;
                    let d2 = dot(d, d);
                    if d2 <= r2 {
                        acc = acc_add(acc, src[j], d, sqrt(d2), f_i, time);
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
                if !cell_intersects_sphere(cell, p, r2) {
                    continue;
                }
                let b = bucket_of(cell);
                let end = offsets[b + 1u];
                for (var s = offsets[b]; s < end; s = s + 1u) {
                    let j = sorted[s];
                    if j == i {
                        continue;
                    }
                    let q = src[j].pos;
                    if any(cell_of(q) != cell) {
                        continue;
                    }
                    let d = q - p;
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
        if src[j].live == 0u {
            continue;
        }
        let d = src[j].pos - p;
        let d2 = dot(d, d);
        if d2 <= r2 {
            acc = acc_add(acc, src[j], d, sqrt(d2), f_i, time);
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
        if src[j].live == 0u {
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

// The same exact-radius query over the live prefix of the grid's sorted list.
// During arrival the allocated capacity can be thousands of times larger than
// the actual flock. Empty cells and inactive slots are not useful candidates.
fn gather_live(p: vec3<f32>, radius: f32, i: u32, f_i: vec3<f32>, time: f32) -> Acc {
    var acc = acc_empty();
    let r2 = radius * radius;
    for (var s = 0u; s < offsets[P.buckets]; s += 1u) {
        let j = sorted[s];
        if j == i { continue; }
        let d = src[j].pos - p;
        let d2 = dot(d, d);
        if d2 <= r2 { acc = acc_add(acc, src[j], d, sqrt(d2), f_i, time); }
    }
    return acc;
}

fn gather_live_centrality(p: vec3<f32>, radius: f32, i: u32) -> Cen {
    var cen = cen_empty();
    let r2 = radius * radius;
    for (var s = 0u; s < offsets[P.buckets]; s += 1u) {
        let j = sorted[s];
        if j == i { continue; }
        let d = src[j].pos - p;
        let d2 = dot(d, d);
        if d2 <= r2 { cen = cen_add(cen, d, sqrt(d2)); }
    }
    return cen;
}

// --- the bounded query -----------------------------------------------------

// Cells a query would walk at this radius.
fn grid_span(lo: vec3<i32>, hi: vec3<i32>) -> f32 {
    let d = vec3<f32>(hi - lo) + vec3<f32>(1.0);
    return d.x * d.y * d.z;
}

// The grid walk costs one bucket lookup per cell in the box, so its cost grows
// with the cube of the radius: at `R = R_MAX = 100 m` a single bird would walk
// `201^3 ≈ 8.1e6` cells — as much as ~300,000 ordinary birds — and the model
// does let a straggler's radius get that far (`params::radius_step` clamps at
// `R_MAX`, and the 100,000-bird flock grows birds hundreds of metres from the
// roost). Measured, that is what made the acceptance run explode at one
// particular simulated second instead of degrading smoothly.
//
// When the box is larger than the flock, walking every bird is both cheaper and
// exact: both walks keep a bird only after the same `d2 <= r2` test, so they
// find the same neighbours — `tests/grid.rs` checks the two agree. The choice
// is a performance one, never a model one, and it bounds a query's cost at
// `min(cells, n)`.
fn gather(p: vec3<f32>, radius: f32, i: u32, f_i: vec3<f32>, time: f32) -> Acc {
    let lo = cell_of(p - vec3<f32>(radius));
    let hi = cell_of(p + vec3<f32>(radius));
    let live = offsets[P.buckets];
    if grid_span(lo, hi) > f32(live) {
        atomicAdd(&clock.wide, 1u);
        if live < P.n { return gather_live(p, radius, i, f_i, time); }
        return gather_bruteforce(p, radius, i, f_i, time);
    }
    return gather_grid(p, radius, i, f_i, time);
}

// The same bound for the centrality walk, which runs at `2R`.
fn gather_centrality(p: vec3<f32>, radius: f32, i: u32) -> Cen {
    let lo = cell_of(p - vec3<f32>(radius));
    let hi = cell_of(p + vec3<f32>(radius));
    let live = offsets[P.buckets];
    if grid_span(lo, hi) > f32(live) {
        atomicAdd(&clock.wide, 1u);
        if live < P.n { return gather_live_centrality(p, radius, i); }
        return gather_bruteforce_centrality(p, radius, i);
    }
    return gather_grid_centrality(p, radius, i);
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

    // Cohesion, stronger at the border through C_i (Eq. 5). During an evening
    // display (`arc.present != 0u`), longitudinal cohesion along the flight
    // axis is relaxed and boundary birds pick up a slow spatial wave +
    // spontaneous edge turns, so the flock folds into sweeping ribbons,
    // sheets and dark turning bands even without a falcon.
    if acc.n_star > 0u {
        var coh_dir = normalize_or_zero(acc.coh);
        if arc.present != 0u {
            coh_dir -= f_i * (dot(coh_dir, f_i) * 0.68);
        }
        force += coh_dir * (P.w_coh * centrality);
    }
    if arc.present != 0u {
        let side = side_of(f_i);
        let phase_a = p.x * 0.048 + p.y * 0.038 - time * 0.72;
        let phase_b = p.x * 0.028 - p.y * 0.052 + time * 0.48;
        let wave_lat = sin(phase_a) * 0.65 + cos(phase_b) * 0.35;
        let wave_vert = sin(phase_b * 0.9 + p.z * 0.06);
        let edge_weight = max(centrality - 0.14, 0.0);
        force += side * (wave_lat * 0.36 * (0.40 + edge_weight))
            + vec3<f32>(0.0, 0.0, wave_vert * 0.12 * (0.25 + edge_weight));
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
            if arc.present != 0u {
                let sweep_vert = sin(p.x * 0.032 - p.y * 0.028 + time * 0.65);
                force += vec3<f32>(0.0, 0.0, sweep_vert * 0.11);
            }
        }
    }

    // Vertical roost pull toward the preferred height (Eq. 12), capped.
    force += vec3<f32>(0.0, 0.0, P.w_roost_v * clamp((arc.z0 - p.z) / P.h_roost_v, -1.0, 1.0));

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

    // Flee and flash expansion (escape::force). Inert without a falcon: the
    // reference returns `Vec3::ZERO` when `falcon` is `None`.
    if falcon.present != 0u {
        let away = p - falcon.pos;
        let dist = length(away);
        if dist < P.r_threat {
            let dir = away / max(dist, 1e-6);
            force += dir * (P.w_flee * (1.0 - dist / P.r_threat));
            if time < b.flash_until {
                force += dir * P.w_flash;
            }
        }
    }

    // Copy a zigzag (escape.rs, the copying rule): a non-alarmed bird starts one
    // when at least `zigzag_k` of its neighbours are in a copyable zigzag.
    // During an evening display, border birds also occasionally initiate a
    // spontaneous turn wave that propagates through the flock.
    var start_zigzag = 0.0;
    if b.alarmed == 0u
        && b.zigzag_active == 0u
        && time >= b.cooldown_until
    {
        if acc.zigzag_neighbours >= P.zigzag_k {
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
        } else if arc.present != 0u && centrality > 0.38 {
            b.rng = b.rng * 747796405u + 2891336453u;
            var c = ((b.rng >> ((b.rng >> 28u) + 4u)) ^ b.rng) * 277803737u;
            c = (c >> 22u) ^ c;
            let u_spont = f32(c) * (1.0 / 4294967296.0);
            if u_spont < 0.0011 {
                let phase = sin(p.x * 0.035 - p.y * 0.035 + time * 0.55);
                start_zigzag = select(-1.0, 1.0, phase >= 0.0);
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

// world.rs::step lines 331-363: the falcon alarms the birds within `R_THREAT`,
// rolls them away from it and flash-expands the closest, fastest pass. Every
// bird every step, whether or not it is due to decide.
fn alarm(b_in: Bird, time: f32) -> Bird {
    var b = b_in;
    if falcon.present == 0u {
        b.alarmed = 0u;
        return b;
    }
    let dist = length(b.pos - falcon.pos);
    if dist < P.r_threat {
        let side = side_of(forward(b.vel));
        let to_falcon = dot(falcon.pos - b.pos, side);
        if b.zigzag_active == 0u && time >= b.cooldown_until {
            b.zigzag_start = time;
            b.zigzag_active = 1u;
            b.zigzag_sign = select(1.0, -1.0, to_falcon > 0.0);
        }
        b.alarmed = 1u;
        if dist < P.r_flash && falcon.speed > P.v_flash {
            b.flash_until = time + P.flash_duration;
        }
    } else {
        b.alarmed = 0u;
    }
    return b;
}

// The decided bird, from the bounded neighbour walk. `b` has already been
// through `alarm`.
fn decide(i: u32, b: Bird, time: f32) -> Bird {
    let f_i = forward(b.vel);
    let acc = gather(b.pos, b.radius, i, f_i, time);
    let cen = gather_centrality(b.pos, 2.0 * b.radius, i);
    return resolve(b, time, b.radius, f_i, acc, cen);
}

// The same decision from the brute-force walk (phase 3 acceptance test only).
fn decide_bruteforce(i: u32, b: Bird, time: f32) -> Bird {
    let f_i = forward(b.vel);
    let acc = gather_bruteforce(b.pos, b.radius, i, f_i, time);
    let cen = gather_bruteforce_centrality(b.pos, 2.0 * b.radius, i);
    return resolve(b, time, b.radius, f_i, acc, cen);
}

@compute @workgroup_size(256)
fn prepare_decisions_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let time = clock.time;
    if src[i].live == 0u {
        // The evening arc's slots not yet filled: no decision, nothing to do.
        dst[i] = src[i];
        return;
    }
    let b = alarm(src[i], time);
    if time >= b.next_decision {
        // After scatter, `bucket` and `counts` are scratch space: queries only
        // read `offsets` and `sorted`. Pack the ~10% of birds due a decision so
        // whole warps do useful work instead of divergent, mostly idle lanes.
        //
        // A bird whose `2R_i` box would take the flock-wide fallback goes to a
        // second queue at the tail of `bucket`, decided by
        // `decide_wide_kernel` with a whole workgroup per bird. Done serially,
        // one such bird scans the whole flock twice in one thread, and that one
        // thread holds the entire `decide` dispatch: after a falcon strike the
        // scattered birds' radii grow and this was the frame-rate collapse.
        if is_wide(b) {
            let slot = atomicAdd(&counts[1], 1u);
            bucket[P.n - 1u - slot] = i;
        } else {
            let slot = atomicAdd(&counts[0], 1u);
            bucket[slot] = i;
        }
    }
    dst[i] = b;
}

// Whether this bird's centrality walk (`2R_i`, the larger of its two) would
// take the flock-wide fallback in `gather_centrality`. Its `R_i` walk is then
// either on the grid or wide as well; both give the same neighbour set as the
// flock-wide scan, which tests only `d2 <= r2`, so the wide kernel may scan.
fn is_wide(b: Bird) -> bool {
    let r_cen = 2.0 * b.radius;
    let lo = cell_of(b.pos - vec3<f32>(r_cen));
    let hi = cell_of(b.pos + vec3<f32>(r_cen));
    return grid_span(lo, hi) > f32(offsets[P.buckets]);
}

@compute @workgroup_size(256)
fn decide_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= atomicLoad(&counts[0]) {
        return;
    }
    let i = bucket[gid.x];
    dst[i] = decide(i, dst[i], clock.time);
}

// --- the wide queue: one workgroup per straggler ---------------------------
//
// The same neighbourhood `gather_live` + `gather_live_centrality` find, split
// across `WIDE_THREADS` lanes that each take every `WIDE_THREADS`-th live bird
// of the sorted list, then a tree sum. Only the summation order differs from
// the serial walk (float addition), as it already does between the grid and
// brute-force walks. Lanes read only the 12-byte position of each candidate.

const WIDE_THREADS: u32 = 128u;
// Workgroups `sim.rs` dispatches; each strides over the wide queue.
const WIDE_GROUPS: u32 = 64u;

var<workgroup> wide_n: u32;
var<workgroup> w_sep: array<vec4<f32>, WIDE_THREADS>;
var<workgroup> w_coh: array<vec4<f32>, WIDE_THREADS>;
var<workgroup> w_align: array<vec4<f32>, WIDE_THREADS>;
var<workgroup> w_csum: array<vec4<f32>, WIDE_THREADS>;
var<workgroup> w_n: array<vec4<u32>, WIDE_THREADS>;
var<workgroup> w_m: array<vec2<u32>, WIDE_THREADS>;

@compute @workgroup_size(128)
fn decide_wide_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    if t == 0u {
        wide_n = atomicLoad(&counts[1]);
    }
    let count = workgroupUniformLoad(&wide_n);
    let live = offsets[P.buckets];
    let time = clock.time;
    for (var k = wid.x; k < count; k += WIDE_GROUPS) {
        let i = bucket[P.n - 1u - k];
        let b = dst[i];
        let p = b.pos;
        let f_i = forward(b.vel);
        let r2 = b.radius * b.radius;
        let r_cen = 2.0 * b.radius;
        let r_cen2 = r_cen * r_cen;

        var acc = acc_empty();
        var cen = cen_empty();
        for (var s = t; s < live; s += WIDE_THREADS) {
            let j = sorted[s];
            if j == i {
                continue;
            }
            let d = src[j].pos - p;
            let d2 = dot(d, d);
            if d2 <= r_cen2 {
                let dist = sqrt(d2);
                cen = cen_add(cen, d, dist);
                if d2 <= r2 {
                    acc = acc_add(acc, src[j], d, dist, f_i, time);
                }
            }
        }
        w_sep[t] = vec4<f32>(acc.sep, acc.zigzag_sign_sum);
        w_coh[t] = vec4<f32>(acc.coh, 0.0);
        w_align[t] = vec4<f32>(acc.align, 0.0);
        w_csum[t] = vec4<f32>(cen.csum, 0.0);
        w_n[t] = vec4<u32>(acc.n_all, acc.n_star, acc.n_align, acc.zigzag_neighbours);
        w_m[t] = vec2<u32>(acc.returning_neighbours, cen.n_g);
        workgroupBarrier();
        for (var h = WIDE_THREADS / 2u; h > 0u; h = h / 2u) {
            if t < h {
                w_sep[t] += w_sep[t + h];
                w_coh[t] += w_coh[t + h];
                w_align[t] += w_align[t + h];
                w_csum[t] += w_csum[t + h];
                w_n[t] += w_n[t + h];
                w_m[t] += w_m[t + h];
            }
            workgroupBarrier();
        }
        if t == 0u {
            let total = Acc(
                w_sep[0].xyz, w_coh[0].xyz, w_align[0].xyz,
                w_n[0].x, w_n[0].y, w_n[0].z, w_n[0].w, w_sep[0].w, w_m[0].x,
            );
            let c = Cen(w_csum[0].xyz, w_m[0].y);
            dst[i] = resolve(b, time, b.radius, f_i, total, c);
            // Count the walks that took the fallback exactly as `gather` and
            // `gather_centrality` do: the `2R_i` one always, the `R_i` one when
            // its own box is wider than the live flock as well.
            let lo = cell_of(p - vec3<f32>(b.radius));
            let hi = cell_of(p + vec3<f32>(b.radius));
            atomicAdd(&clock.wide, select(1u, 2u, grid_span(lo, hi) > f32(live)));
        }
        // The next straggler reuses the shared arrays.
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn decide_bruteforce_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let time = clock.time;
    if src[i].live == 0u {
        dst[i] = src[i];
        return;
    }
    var b = alarm(src[i], time);
    if time >= b.next_decision {
        b = decide_bruteforce(i, b, time);
    }
    dst[i] = b;
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
    if b.live == 0u {
        // Not in the flock: no flight, no wingbeat.
        return;
    }

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
// same `time` and many steps fit in one command buffer. The step's wide-query
// count starts here (see `gather`).
@compute @workgroup_size(1)
fn tick_kernel() {
    clock.step += 1u;
    clock.time += P.dt;
    atomicStore(&clock.wide, 0u);
}
