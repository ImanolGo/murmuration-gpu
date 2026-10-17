// The falcon (PLAN.md phase 7), drawn as a peregrine silhouette rather than a
// Gaussian dot.
//
// Rendering only: the simulation's falcon is a point with a velocity. The body
// frame comes from that velocity and the bank `crate::falcon::FalconMotion`
// derives (`scene.falcon_vel`, `scene.falcon_bank`), with the starlings' own
// `forward` / `side_of` / `body_up`.
//
// The shape is analytic, evaluated per pixel inside one screen quad, so it stays
// crisp and anti-aliased at any size without depending on TAA:
//
// * each wing is a planform polygon in its own plane, tilted by the wingbeat's
//   dihedral, so a flapping wing really foreshortens;
// * the tail is a narrow fan in the body plane;
// * the body and head are ellipsoids, so the bird keeps its mass edge-on.
//
// Pose follows speed: cruising and climbing alternate flapping bursts with
// glides; above ~22 m/s (the stoop is 30 m/s) the wings fold back into the
// teardrop of a diving peregrine. A real 1 m falcon at 150–300 m is only a few
// pixels across, so the drawing is scaled up to a minimum projected span
// (`MIN_SPAN_PX`) to stay legible as a falcon; this changes the picture, never
// the simulation.

const FALCON_SPAN: f32 = 1.0;
const MIN_SPAN_PX: f32 = 16.0;
// Silhouette dilation [px]: keeps the narrow wings at least a pixel thick.
const DILATE_PX: f32 = 0.35;
const FALCON_TAU: f32 = 4.6051702;
const TAU_F: f32 = 6.283185307179586;

struct FalconOut {
    @builtin(position) clip: vec4<f32>,
    /// Drawing scale (≥ 1: the size floor), atmosphere.
    @location(0) @interpolate(flat) params: vec2<f32>,
};

struct Frame {
    f: vec3<f32>,
    side: vec3<f32>,
    up: vec3<f32>,
};

fn falcon_frame() -> Frame {
    var fr: Frame;
    fr.f = forward(scene.falcon_vel);
    fr.side = side_of(fr.f);
    fr.up = body_up(fr.f, scene.falcon_bank);
    return fr;
}

@vertex
fn vs_falcon(@builtin(vertex_index) vid: u32) -> FalconOut {
    var out: FalconOut;
    let rel = scene.falcon_pos - scene.eye;
    let depth = dot(rel, scene.forward);
    if scene.falcon < 0.5 || depth <= 1.0 {
        out.clip = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.params = vec2<f32>(1.0, 0.0);
        return out;
    }
    let inv = focal_px() / depth;
    let centre = vec2<f32>(
        scene.viewport.x * 0.5 + dot(rel, scene.right) * inv,
        scene.viewport.y * 0.5 - dot(rel, scene.up) * inv,
    ) + scene.jitter;
    let k = max(1.0, MIN_SPAN_PX / (FALCON_SPAN * inv));
    // Half the span plus the tail's reach, and two pixels for the edge ramp.
    let r = 0.62 * k * FALCON_SPAN * inv + 2.0;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0),
    );
    let px = centre + corners[vid] * r;
    out.clip = vec4<f32>(
        px.x / scene.viewport.x * 2.0 - 1.0,
        1.0 - px.y / scene.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.params = vec2<f32>(k, clamp(1.0 - depth / ATMOSPHERE, 0.25, 1.0));
    return out;
}

// Signed distance to a polygon (negative inside), after Inigo Quilez.
fn sd_polygon9(p: vec2<f32>, poly: array<vec2<f32>, 9>) -> f32 {
    var v = poly;
    var d = dot(p - v[0], p - v[0]);
    var s = 1.0;
    var j = 8u;
    for (var i = 0u; i < 9u; i += 1u) {
        let e = v[j] - v[i];
        let w = p - v[i];
        let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));
        let c0 = p.y >= v[i].y;
        let c1 = p.y < v[j].y;
        let c2 = e.x * w.y > e.y * w.x;
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            s = -s;
        }
        j = i;
    }
    return s * sqrt(d);
}

fn sd_polygon5(p: vec2<f32>, poly: array<vec2<f32>, 5>) -> f32 {
    var v = poly;
    var d = dot(p - v[0], p - v[0]);
    var s = 1.0;
    var j = 4u;
    for (var i = 0u; i < 5u; i += 1u) {
        let e = v[j] - v[i];
        let w = p - v[i];
        let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));
        let c0 = p.y >= v[i].y;
        let c1 = p.y < v[j].y;
        let c2 = e.x * w.y > e.y * w.x;
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            s = -s;
        }
        j = i;
    }
    return s * sqrt(d);
}

// One wing's planform \[m\]: x spanwise from the root, y along the body (head
// positive). Long, pointed, with the wrist bowed forward when gliding; folded
// back along the body in the stoop.
fn wing_planform(tuck: f32) -> array<vec2<f32>, 9> {
    let glide = array<vec2<f32>, 9>(
        vec2<f32>(-0.02, 0.085), vec2<f32>(0.12, 0.100), vec2<f32>(0.23, 0.090),
        vec2<f32>(0.38, 0.010), vec2<f32>(0.47, -0.095), vec2<f32>(0.40, -0.090),
        vec2<f32>(0.27, -0.070), vec2<f32>(0.12, -0.085), vec2<f32>(-0.02, -0.075),
    );
    let stoop = array<vec2<f32>, 9>(
        vec2<f32>(-0.02, 0.060), vec2<f32>(0.04, 0.065), vec2<f32>(0.085, 0.030),
        vec2<f32>(0.105, -0.080), vec2<f32>(0.085, -0.300), vec2<f32>(0.065, -0.255),
        vec2<f32>(0.050, -0.170), vec2<f32>(0.030, -0.100), vec2<f32>(-0.02, -0.080),
    );
    var out: array<vec2<f32>, 9>;
    for (var i = 0u; i < 9u; i += 1u) {
        out[i] = mix(glide[i], stoop[i], tuck);
    }
    return out;
}

// Ray/plane hit in the plane's own coordinates, and how many metres one pixel
// covers there. `w` is 0 when the plane is seen edge-on and should be ignored.
struct PlaneHit {
    uv: vec2<f32>,
    px_m: f32,
    w: f32,
};

fn plane_hit(ray: vec3<f32>, origin: vec3<f32>, ax: vec3<f32>, ay: vec3<f32>, focal: f32) -> PlaneHit {
    var h: PlaneHit;
    let n = normalize(cross(ax, ay));
    let denom = dot(ray, n);
    let grazing = abs(denom);
    let t = dot(origin - scene.eye, n) / select(denom, 1e-6, abs(denom) < 1e-6);
    let hit = scene.eye + ray * t - origin;
    h.uv = vec2<f32>(dot(hit, ax), dot(hit, ay));
    h.px_m = max(t, 1.0) / focal / max(grazing, 0.08);
    h.w = smoothstep(0.03, 0.10, grazing) * select(0.0, 1.0, t > 0.0);
    return h;
}

// Coverage of an ellipsoid with semi-axes `a` along (f, side, up) at `centre`.
fn ellipsoid_cover(ray: vec3<f32>, centre: vec3<f32>, fr: Frame, a: vec3<f32>, focal: f32) -> f32 {
    let o = scene.eye - centre;
    let o2 = vec3<f32>(dot(o, fr.f), dot(o, fr.side), dot(o, fr.up)) / a;
    let d2 = vec3<f32>(dot(ray, fr.f), dot(ray, fr.side), dot(ray, fr.up)) / a;
    let tc = -dot(o2, d2) / max(dot(d2, d2), 1e-12);
    let m = length(o2 + d2 * tc);
    // Normalised distance to metres with the ellipsoid's smaller cross-section,
    // then to pixels at the closest approach.
    let dist = length(-o);
    let px_m = dist / focal;
    let edge = (m - 1.0) * min(a.y, a.z) / px_m;
    return clamp(0.5 + DILATE_PX - edge, 0.0, 1.0);
}

@fragment
fn fs_falcon(in: FalconOut) -> BirdOut {
    let k = in.params.x;
    let atmosphere = in.params.y;
    let focal = focal_px();
    let ray = ray_through(in.clip.xy - scene.jitter);
    let fr = falcon_frame();
    let c = scene.falcon_pos;

    // Pose from speed: stoop tucks; otherwise flapping bursts between glides.
    let speed = length(scene.falcon_vel);
    let tuck = smoothstep(21.0, 27.0, speed);
    let burst = smoothstep(-0.2, 0.4, sin(scene.time * 0.9) + 0.35 * sin(scene.time * 2.3));
    let beat = sin(scene.time * TAU_F * 4.2);
    // Upstroke higher than the downstroke, with a slight gull-wing set.
    let dihedral = (1.0 - tuck) * (0.10 + burst * (0.55 * beat + 0.12));

    var cover = 0.0;
    let wing = wing_planform(tuck);
    for (var w = 0u; w < 2u; w += 1u) {
        let sgn = select(-1.0, 1.0, w == 0u);
        // Spanwise axis, raised by the dihedral about the body axis.
        let span = fr.side * (sgn * cos(dihedral)) + fr.up * sin(dihedral);
        let root = c + fr.side * (sgn * 0.035 * k);
        let h = plane_hit(ray, root, span, fr.f, focal);
        let d = sd_polygon9(h.uv / k, wing) * k;
        cover = max(cover, clamp(0.5 + DILATE_PX - d / h.px_m, 0.0, 1.0) * h.w);
    }

    // Tail: a narrow fan, spread a little in a glide, closed in the stoop.
    let spread = 0.022 * (1.0 - tuck);
    let tail = array<vec2<f32>, 5>(
        vec2<f32>(-0.01, -0.07), vec2<f32>(0.040, -0.09), vec2<f32>(0.050 + spread, -0.31),
        vec2<f32>(0.020, -0.325), vec2<f32>(-0.01, -0.325),
    );
    let ht = plane_hit(ray, c, fr.side, fr.f, focal);
    let dt = sd_polygon5(vec2<f32>(abs(ht.uv.x), ht.uv.y) / k, tail) * k;
    cover = max(cover, clamp(0.5 + DILATE_PX - dt / ht.px_m, 0.0, 1.0) * ht.w);

    // Body and head.
    cover = max(cover, ellipsoid_cover(ray, c, fr, vec3<f32>(0.20, 0.060, 0.055) * k, focal));
    cover = max(cover, ellipsoid_cover(
        ray, c + fr.f * (0.185 * k), fr, vec3<f32>(0.050, 0.042, 0.040) * k, focal));

    if cover <= 0.0 {
        discard;
    }
    // An opaque silhouette like the starling mesh, softened only by distance.
    let raw = cover * atmosphere * FALCON_TAU;
    var out: BirdOut;
    let dark = cover * mix(0.88, 0.99, atmosphere);
    out.hdr = vec4<f32>(dark, dark, dark, 1.0);
    out.optics = vec4<f32>(raw, 0.0, 0.0, 0.0);
    return out;
}
