// The land under the sky: a wetland at dusk, seen from 1.7 m.
//
// Rendering only, never read by the flock. Composited by `sky_pass.wgsl` on
// top of `atmosphere_sky`, for the rays that can meet it (a few degrees around
// the horizon and below), so a camera looking up at the flock pays nothing.
//
// From the far distance in:
//
// * a low line of hills, almost lost in the haze;
// * woods and hedgerows at two distances: crowns, gaps, the odd poplar;
// * the ground itself, ray-cast onto z = 0 in world coordinates so it does not
//   slide with the camera: dark fields and marsh, and winding channels and pools
//   of open water that mirror the sky — the brightest thing below the horizon at
//   dusk, as in the photographs;
// * aerial perspective throughout: everything fades into the horizon colour
//   with distance, so there is no hard grey band.
//
// Every edge is anti-aliased against the angular pixel size.

fn land_hash1(x: f32) -> f32 {
    return fract(sin(x * 127.1) * 43758.5453);
}

fn land_noise1(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(land_hash1(i), land_hash1(i + 1.0), u);
}

// Two-dimensional value noise in world metres, reusing the sky's lattice hash.
fn land_noise2(p: vec2<f32>) -> f32 {
    return sky_noise(p);
}

fn land_fbm2(p: vec2<f32>) -> f32 {
    return land_noise2(p) * 0.5 + land_noise2(p * 2.07 + 31.0) * 0.3
        + land_noise2(p * 4.13 - 17.0) * 0.2;
}

// Bearing is periodic: sample noise on a circle so there is no seam behind the
// observer. `freq` is in features per radian.
fn ring_noise(bearing: f32, freq: f32, seed: f32) -> f32 {
    let r = freq / 6.283185307179586;
    let p = vec2<f32>(cos(bearing), sin(bearing)) * r + vec2<f32>(seed, seed * 1.7);
    return land_noise2(p);
}

// A tree line `distance` metres away, as an angular height above the horizon
// for this bearing. `cell` is the crown spacing [rad]; woods come and go in
// clumps, with fields between them.
fn tree_line(bearing: f32, cell: f32, crown: f32, seed: f32) -> f32 {
    let clump = ring_noise(bearing, 9.0, seed);
    let hedge = ring_noise(bearing, 40.0, seed + 5.0);
    let x = bearing / cell;
    let i = floor(x);
    var h = 0.0;
    for (var k = -1.0; k <= 1.0; k += 1.0) {
        let c = i + k;
        let r = land_hash1(c * 1.37 + seed);
        let r2 = land_hash1(c * 2.91 + seed * 3.1);
        // Where this crown stands, how tall and how wide.
        let centre = c + 0.2 + 0.6 * r;
        let present = smoothstep(0.42, 0.55, clump + 0.25 * (r2 - 0.5));
        var height = crown * (0.55 + 0.6 * r2) * present;
        var width = 0.65 + 0.5 * r;
        // An occasional poplar: tall and narrow.
        if r2 > 0.93 {
            height = crown * 2.3 * present;
            width = 0.28;
        }
        let u = (x - centre) / width;
        if abs(u) < 1.0 {
            h = max(h, height * sqrt(1.0 - u * u));
        }
    }
    // A low hedge joins the clumps.
    let hedge_h = crown * 0.28 * smoothstep(0.35, 0.6, hedge);
    return max(h, hedge_h);
}

// Distance fog towards the horizon colour.
fn land_fog(colour: vec3<f32>, horizon: vec3<f32>, distance: f32) -> vec3<f32> {
    let f = 1.0 - exp(-distance / 1400.0);
    return mix(colour, horizon, f * 0.92);
}

// The distant shore, far to near, over `base` at elevation `e` [rad]: hills
// (~4 km), woods (~1.6 km) and tree lines and hedges (~500 m), each fading
// into `horizon` with distance. `lift` [rad] shifts every profile, which is
// how the water mirrors them.
fn shore_silhouettes(base: vec3<f32>, e: f32, bearing: f32, horizon: vec3<f32>,
                     ambient: vec3<f32>, light: f32, aa: f32) -> vec3<f32> {
    var colour = base;
    // A layer at D metres stands on ground seen at -eye/D: nothing below that.
    let eye_h = max(scene.eye.z, 0.5);
    let hills = 0.0030 + 0.0070 * (ring_noise(bearing, 5.0, 2.0) * 0.7
        + ring_noise(bearing, 14.0, 7.0) * 0.3);
    let hill_col = land_fog(vec3<f32>(0.020, 0.024, 0.032) * light + ambient, horizon, 4000.0);
    colour = mix(colour, hill_col, clamp((hills - e) / aa + 0.5, 0.0, 1.0)
        * clamp((e + eye_h / 4000.0) / aa + 0.5, 0.0, 1.0));

    let far_trees = 0.0010 + tree_line(bearing, 0.0065, 0.0070, 11.0);
    let far_col = land_fog(vec3<f32>(0.014, 0.016, 0.014) * light + ambient * 0.8, horizon, 1600.0);
    colour = mix(colour, far_col, clamp((far_trees - e) / aa + 0.5, 0.0, 1.0)
        * clamp((e + eye_h / 1600.0) / aa + 0.5, 0.0, 1.0));

    let near_trees = tree_line(bearing, 0.016, 0.019, 23.0)
        * smoothstep(0.30, 0.55, ring_noise(bearing, 3.0, 41.0));
    let near_col = land_fog(vec3<f32>(0.010, 0.011, 0.009) * light + ambient * 0.6, horizon, 600.0);
    colour = mix(colour, near_col, clamp((near_trees - e) / aa + 0.5, 0.0, 1.0)
        * clamp((e + eye_h / 600.0) / aa + 0.5, 0.0, 1.0) * step(1e-4, near_trees));
    return colour;
}

// Composites the landscape over `sky` for the view ray `ray`.
fn landscape(ray: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {
    let elevation = asin(clamp(ray.z, -1.0, 1.0));
    // Nothing on the land reaches above ~3.5°.
    if elevation > 0.06 {
        return sky;
    }
    let aa = scene.fov / scene.viewport.y;
    let eye_h = max(scene.eye.z, 0.5);
    let bearing = atan2(ray.y, ray.x);
    let light = 1.0 - smoothstep(0.55, 1.0, scene.evening) * 0.97;

    // The horizon colour this way: the haze everything fades into.
    let flat_ray = normalize(vec3<f32>(ray.x, ray.y, 0.0));
    let horizon = atmosphere_sky(flat_ray);
    // The land is lit by the sky, not the sun, at this hour.
    let ambient = horizon * 0.05 + vec3<f32>(0.005, 0.004, 0.003) * light;

    var colour = sky;
    if ray.z < 0.0 {
        let d = eye_h / max(-ray.z, 1e-5);
        let g = scene.eye.xy + ray.xy * d;
        // How many metres one pixel spans on the ground here, along the view.
        let pixel_m = d * aa / max(-ray.z, 1e-3);
        // Where the far shore is along this bearing [m].
        let shore = 650.0 + 700.0 * ring_noise(bearing, 6.0, 3.0);

        // Land: the bank at the observer's feet and the fields beyond the
        // shore. Dark, lit only by the sky.
        let tone = land_fbm2(g * 0.02);
        // Close up, the bank is rough grass: fine streaks that fade with range.
        let grass = land_noise2(vec2<f32>(bearing * 1400.0, d * 6.0)) * (1.0 - smoothstep(0.03, 0.12, pixel_m));
        var land = mix(vec3<f32>(0.004, 0.0035, 0.003), vec3<f32>(0.010, 0.008, 0.005), tone) * light
            + ambient * mix(0.10, 0.22, tone) * (0.75 + 0.5 * grass);

        // Open water between the bank (~20 m out) and the far shore: the sky
        // and the far shore mirrored, with the faint horizontal streaks of a
        // calm surface, and Fresnel reflectance high at these grazing angles.
        let refl_ray = vec3<f32>(ray.x, ray.y, -ray.z);
        var mirrored = atmosphere_sky(refl_ray);
        mirrored = shore_silhouettes(mirrored, -elevation - 2.0 * eye_h / 900.0, bearing,
                                     horizon, ambient, light, aa);
        let fresnel = 0.58 + 0.38 * pow(1.0 - min(-ray.z * 6.0, 1.0), 3.0);
        let streak_detail = 1.0 - smoothstep(0.5, 4.0, pixel_m);
        let streak = land_noise2(vec2<f32>(bearing * 25.0, d * 1.1)) - 0.5;
        let water_col = mirrored * fresnel * (1.0 + streak * 0.22 * streak_detail);
        // The near bank's edge wanders between ~9 and ~17 m out.
        let bank_edge = 9.0 + 8.0 * ring_noise(bearing, 22.0, 5.0)
            + 0.9 * land_noise1(bearing * 2600.0) + 0.5 * land_noise1(bearing * 9000.0);
        let open = smoothstep(bank_edge - 0.5 * pixel_m, bank_edge + 0.5 * pixel_m, d)
            * (1.0 - smoothstep(shore - 0.5 * pixel_m, shore + 0.5 * pixel_m, d));

        colour = mix(land, water_col, open);
        colour = land_fog(colour, horizon, d * mix(1.0, 0.45, open));
        // Blend into the horizon over the last pixel, so it never aliases.
        colour = mix(colour, horizon, smoothstep(-aa * 1.5, 0.0, elevation));
    }
    return shore_silhouettes(colour, elevation, bearing, horizon, ambient, light, aa);
}
