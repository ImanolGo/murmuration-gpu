// The reed bed in front of the observer (foreground extension of
// sky.rs::reed_visibility / the reference's draw_reeds). Rendering only; the
// reeds never enter the flock's optical depth.
//
// Two banks of Phragmites, ~30 m and ~12 m out, each drawn in its own metres: `x` along
// the bank (bearing × distance) and `y` as height above the ground. Each bank
// is a dense body of stems, ragged at the top, from which taller stems rise
// with drooping, feathery seed plumes and a few long leaves, all bent by one
// slow wind. Nearer banks are darker; farther ones take the haze. Every edge is
// anti-aliased against the pixel's size in metres at that distance.

struct ReedOut {
    @location(0) hdr: vec4<f32>,
    @location(1) optics: vec4<f32>,
};

@vertex
fn vs_reeds(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vid], 0.0, 1.0);
}

fn reed_hash(n: u32) -> vec4<f32> {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    let a = f32(h & 255u) / 255.0;
    let b = f32((h >> 8u) & 255u) / 255.0;
    let c = f32((h >> 16u) & 255u) / 255.0;
    let d = f32((h >> 24u) & 255u) / 255.0;
    return vec4<f32>(a, b, c, d);
}

fn reed_noise(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = fract(sin(i * 91.7) * 43758.5453);
    let b = fract(sin((i + 1.0) * 91.7) * 43758.5453);
    return mix(a, b, u);
}

// Anti-aliased coverage of a band of half-width `w` at distance `d`.
fn band(d: f32, w: f32, px: f32) -> f32 {
    return clamp((w - d) / px + 0.5, 0.0, 1.0) * min(w * 2.0 / px, 1.0);
}

// Distance from `p` to the segment `a`–`b`, and where along it (0…1).
fn seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return vec2<f32>(length(p - a - ab * t), t);
}

@fragment
fn fs_reeds(@builtin(position) coord: vec4<f32>) -> ReedOut {
    var out: ReedOut;
    out.hdr = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.optics = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (scene.reed < 0.002) {
        return out;
    }

    let ray = ray_through(coord.xy);
    let elevation = asin(clamp(ray.z, -1.0, 1.0));
    if elevation > 0.20 { return out; }
    let bearing = atan2(ray.y, ray.x);
    let aa = scene.fov / scene.viewport.y;
    let eye_h = max(scene.eye.z, 0.5);
    let gust = sin(scene.time * 0.31) * 0.5 + sin(scene.time * 0.13 + 1.3) * 0.5;

    var transmission = vec3<f32>(1.0);
    let distances = array<f32, 2>(34.0, 13.0);
    for (var layer = 0u; layer < 2u; layer += 1u) {
        let dist = distances[layer];
        // This pixel in the bank's own metres.
        let x = bearing * dist;
        let y = eye_h + tan(elevation) * dist;
        if y < 0.0 { continue; }
        let px = aa * dist;
        let seed = f32(layer) * 37.0;

        // The banks come in stands with gaps onto the water; the far one is
        // nearly continuous, the near one patchy. A stand tapers in height at
        // its ends rather than fading out.
        let freq = mix(0.035, 0.06, f32(layer));
        let lo = mix(0.12, 0.36, f32(layer));
        let stand = smoothstep(lo, lo + 0.10, reed_noise(x * freq + seed * 0.37));

        // The body: dense stems up to a rolling top, its edge combed into
        // stems rather than cut straight.
        // Pointed tips of the unresolved stems, a few centimetres apart.
        let tip_cell = x / 0.055;
        let tip_rand = reed_hash(bitcast<u32>(i32(floor(tip_cell))) * 977u + layer * 131u);
        let tip_shape = pow(max(1.0 - abs(2.0 * fract(tip_cell) - 1.0), 0.0), 0.6);
        let comb = tip_rand.x * tip_rand.x * 0.32 * tip_shape + reed_noise(x * 7.0 + seed) * 0.10;
        let top = (2.0 + 0.45 * reed_noise(x * 0.07 + seed) + comb) * mix(0.6, 1.0, stand) - 2.5 * (1.0 - stand);
        var cover = clamp((top - y) / px + 0.5, 0.0, 1.0);

        // Taller stems, about one every 0.4 m in a stand, each with a plume
        // and some with a long leaf.
        let spacing = 0.28;
        let cell = floor(x / spacing);
        for (var k = -2; k <= 2; k += 1) {
            let index = cell + f32(k);
            let r = reed_hash(bitcast<u32>(i32(index)) * 2654435761u + layer * 2246822519u);
            let here = smoothstep(lo - 0.02, lo + 0.10, reed_noise((index + 0.5) * spacing * freq + seed * 0.37));
            if r.x > here * 0.9 { continue; }
            let base_x = (index + 0.15 + 0.7 * r.y) * spacing;
            let height = 2.25 + 0.45 * reed_noise(base_x * 0.07 + seed) + 0.45 * r.z - 2.6 * (1.0 - here);
            // Wind: one gust for the bed, a phase per stem; tips lean the most.
            let sway = 0.10 + 0.22 * gust + 0.05 * sin(scene.time * (1.1 + r.w) + r.y * 20.0);
            let rise = clamp((y - 1.0) / max(height - 1.0, 0.1), 0.0, 1.0);
            let stem_x = base_x + sway * rise * rise * (height - 1.0);
            if y < height {
                cover = max(cover, band(abs(x - stem_x), mix(0.014, 0.008, rise), px));
            }
            // A long leaf leaving the stem and arching down-wind.
            if r.w > 0.55 {
                let leaf_h = height - 0.55 + 0.3 * r.w;
                let ly = clamp((leaf_h - 1.0) / max(height - 1.0, 0.1), 0.0, 1.0);
                let a = vec2<f32>(base_x + sway * ly * ly * (height - 1.0), leaf_h);
                let dir = select(-1.0, 1.0, r.y > 0.5);
                let b = a + vec2<f32>(dir * (0.20 + 0.10 * r.z) + sway * 0.25, 0.06 - 0.14 * r.x);
                let s = seg(vec2<f32>(x, y), a, b);
                cover = max(cover, band(s.x, 0.010 * (1.0 - s.y) + 0.0015, px));
            }
            // The plume: a soft, drooping panicle at the tip.
            let tip = vec2<f32>(stem_x, height);
            let droop = vec2<f32>(0.10 + sway * 0.7, -0.22);
            let s = seg(vec2<f32>(x, y), tip, tip + droop);
            let w = 0.03 * sin(3.14159 * clamp(s.y * 0.85 + 0.15, 0.0, 1.0)) + 0.004;
            let fluff = 0.82 + 0.18 * reed_noise((x + y) * 70.0 + seed);
            cover = max(cover, band(s.x, w, px) * fluff);
        }

        // The far bank takes a little haze.
        let haze = select(1.0, 0.95, layer == 0u);
        let absorption = vec3<f32>(0.995, 0.985, 0.975) * (cover * haze * scene.reed);
        transmission *= 1.0 - absorption;
    }
    out.hdr = vec4<f32>(1.0 - transmission, 1.0);
    return out;
}
