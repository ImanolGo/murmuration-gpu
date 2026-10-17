// The falcon (PLAN.md phase 7), mirrored from `murmur-ref/src/falcon.rs` and the
// falcon block of `world.rs::step`.
//
// Two dispatches run per step while a falcon hunts. `falcon_reduce_kernel`
// computes the flock's centroid and extent, which the falcon steers by exactly
// as `world.rs` computes them from `self.position` every step (one workgroup
// walks the flock twice: the sum first, because the extent is measured about the
// centroid). `falcon_kernel` then advances the falcon by one step.
//
// The agent state lives in `Falcon`, declared in `common.wgsl` and mirrored by
// `crates/murmur-gpu/src/falcon.rs`. One adaptation: the random stream.
// `Falcon::new` and the retreat-to-cruise transition draw from a ChaCha8 stream
// that cannot run in WGSL, so the GPU draws from the same counter-based PCG as
// the birds, seeded from the run's seed; both sides advance it with the same
// step, so `Falcon::spawn` and the shader agree on the first draw.

const FALCON_THREADS: u32 = 256u;

var<workgroup> falcon_reduce: array<f32, FALCON_THREADS>;

// A tree sum over the workgroup.
fn falcon_sum(v: f32, t: u32) -> f32 {
    falcon_reduce[t] = v;
    workgroupBarrier();
    var s = FALCON_THREADS / 2u;
    loop {
        if s == 0u {
            break;
        }
        if t < s {
            falcon_reduce[t] += falcon_reduce[t + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    return falcon_reduce[0];
}

// A tree max over the workgroup.
fn falcon_max(v: f32, t: u32) -> f32 {
    falcon_reduce[t] = v;
    workgroupBarrier();
    var s = FALCON_THREADS / 2u;
    loop {
        if s == 0u {
            break;
        }
        if t < s {
            falcon_reduce[t] = max(falcon_reduce[t], falcon_reduce[t + s]);
        }
        workgroupBarrier();
        s = s / 2u;
    }
    return falcon_reduce[0];
}

// world.rs::step lines 319-325.
@compute @workgroup_size(256)
fn falcon_reduce_kernel(@builtin(local_invocation_id) lid: vec3<u32>) {
    if P.n == 0u {
        return;
    }
    let t = lid.x;
    var sx = 0.0;
    var sy = 0.0;
    var sz = 0.0;
    var count = 0.0;
    var i = t;
    while (i < P.n) {
        if src[i].live != 0u {
            let p = src[i].pos;
            sx += p.x;
            sy += p.y;
            sz += p.z;
            count += 1.0;
        }
        i += FALCON_THREADS;
    }
    let live_n = max(falcon_sum(count, t), 1.0);
    let c = vec3<f32>(
        falcon_sum(sx, t) / live_n,
        falcon_sum(sy, t) / live_n,
        falcon_sum(sz, t) / live_n,
    );

    var m = 0.0;
    var j = t;
    while (j < P.n) {
        if src[j].live != 0u {
            m = max(m, length(src[j].pos - c));
        }
        j += FALCON_THREADS;
    }
    // `world.rs` folds a 1.0 floor into the extent.
    let extent = max(falcon_max(m, t), 1.0);
    if t == 0u {
        falcon.centroid = c;
        falcon.extent = extent;
    }
}

// falcon.rs::Falcon::steer_toward.
fn steer_toward(point: vec3<f32>, speed: f32, dt: f32) {
    let toward = point - falcon.pos;
    var desired = vec3<f32>(0.0);
    if dot(toward, toward) > 1e-6 {
        desired = normalize(toward) * speed;
    }
    falcon.vel += (desired - falcon.vel) * (1.0 - exp(-dt / 0.4));
    falcon.pos += falcon.vel * dt;
}

// The next uniform in [0, 1) from the falcon's counter-based stream, matching
// `falcon_next_uniform` in `crates/murmur-gpu/src/falcon.rs`.
fn falcon_next_uniform() -> f32 {
    let r = falcon.rng * 747796405u + 2891336453u;
    var w = ((r >> ((r >> 28u) + 4u)) ^ r) * 277803737u;
    w = (w >> 22u) ^ w;
    falcon.rng = r;
    return f32(w) * (1.0 / 4294967296.0);
}

// falcon.rs::Falcon::update.
@compute @workgroup_size(1)
fn falcon_kernel() {
    if falcon.present == 0u {
        return;
    }
    let dt = P.dt;
    let time = clock.time;

    if falcon.mode == 1u {
        // Chases the point the user points at.
        let toward = falcon.aim - falcon.pos;
        var desired = vec3<f32>(0.0);
        if dot(toward, toward) > 1e-6 {
            desired = normalize(toward) * P.falcon_stoop;
        }
        falcon.vel += (desired - falcon.vel) * (1.0 - exp(-dt / 0.3));
        falcon.pos += falcon.vel * dt;
        falcon.speed = length(falcon.vel);
        return;
    }

    let centroid = falcon.centroid;
    let extent = falcon.extent;
    switch falcon.state {
        case 0u: {
            // Cruise: patrol around the flock, then attack.
            let patrol = centroid + vec3<f32>(extent + 80.0, 0.0, 20.0);
            steer_toward(patrol, P.falcon_cruise, dt);
            if time >= falcon.until {
                falcon.aim = centroid;
                falcon.state = 1u;
            }
        }
        case 1u: {
            // Climb: rise above the flock, then pick the dive direction.
            falcon.aim = centroid;
            let above = falcon.aim + vec3<f32>(0.0, 0.0, P.falcon_attack_height);
            steer_toward(above, P.falcon_climb, dt);
            if length(falcon.pos - above) < 8.0 {
                var dive = normalize_or_zero(falcon.aim - falcon.pos);
                if dot(dive, dive) < 0.5 {
                    dive = vec3<f32>(0.0, 0.0, -1.0);
                }
                falcon.dive = dive;
                falcon.state = 2u;
            }
        }
        case 2u: {
            // Stoop: straight at the flock until it is overshot.
            falcon.vel = falcon.dive * P.falcon_stoop;
            falcon.pos += falcon.vel * dt;
            if dot(falcon.pos - falcon.aim, falcon.dive) > 0.0 {
                falcon.state = 3u;
            }
        }
        default: {
            // Retreat: climb away, then cruise again after a random interval.
            let away = normalize_or_zero(
                falcon.pos - falcon.aim + vec3<f32>(0.0, 0.0, P.falcon_attack_height),
            );
            steer_toward(falcon.pos + away * 100.0, P.falcon_climb, dt);
            if falcon.pos.z > falcon.aim.z + P.falcon_attack_height + 20.0 {
                falcon.state = 0u;
                let u = falcon_next_uniform();
                falcon.until = time + P.falcon_attack_min + u * (P.falcon_attack_max - P.falcon_attack_min);
            }
        }
    }
    falcon.speed = length(falcon.vel);
}
