// LOD level 1: a flapping impostor quad (PLAN.md phase 6).
//
// Between the mesh and the sub-pixel splat a bird is two to six pixels across:
// too big to be a dot, too small for ten triangles. The impostor draws it as a
// screen-aligned quad carrying a smooth bird silhouette — a mixture of three
// normalised Gaussians, one body and two wing lobes — so it is anti-aliased by
// construction and cannot shimmer.
//
// The mixture is *mass preserving by construction*: a normalised Gaussian
// `exp(-(du²/2σu² + dv²/2σv²))` has integral `2πσuσv`, so dividing the mixture
// by the sum of its own integrals makes the bird's total optical depth exactly
// `a·ρ²` — the same as the splat and the mesh's mean. Only the *shape* changes
// when a bird crosses into this level.
//
// The wing lobes sit along the bird's own wing axis and shorten by `cos(flap)`
// as the wings rise, so the silhouette sweeps through the wingbeat the same way
// the mesh's does.

@group(0) @binding(1) var<storage, read> birds: array<Bird>;
@group(0) @binding(2) var<storage, read> lod_lists: array<vec2<u32>>;

/// This pass draws level 1 of the instance lists.
const LEVEL: u32 = 1u;

const PI: f32 = 3.141592653589793;

// Silhouette sigmas, in units of the projected wingspan ρ: body along the body
// axis and across it, then a wing lobe's chord (along the body) and span (along
// the wing axis, before the `cos(flap)` foreshortening).
const BODY_AXIS: f32 = 0.30;
const BODY_ACROSS: f32 = 0.24;
const WING_CHORD: f32 = 0.14;
const WING_SPAN: f32 = 0.22;

/// How far out along the wing axis a lobe's centre sits, at full span.
const WING_OFFSET: f32 = 0.34;

struct ImpostorOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) centre: vec2<f32>,
    @location(1) rho: f32,
    @location(2) a: f32,
    @location(3) weight: f32,
    @location(4) fwd2: vec2<f32>,
    @location(5) side2: vec2<f32>,
    @location(6) flap: f32,
};

@vertex
fn vs_impostor(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> ImpostorOut {
    var out: ImpostorOut;
    let stride = arrayLength(&lod_lists) / 3u;
    let entry = lod_lists[LEVEL * stride + iid];
    let b = birds[entry.x];
    let s = bird_screen(b);

    // A quad wide enough to hold the whole mixture: the widest lobe centre
    // (WING_OFFSET) plus three of its sigmas (WING_SPAN).
    let r = 1.3 * max(1.0, s.rho);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0),
    );
    let off = corners[vid] * r;
    let px = s.centre.x + off.x;
    let py = s.centre.y + off.y;
    out.clip = vec4<f32>(
        px / scene.viewport.x * 2.0 - 1.0,
        1.0 - py / scene.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.centre = s.centre;
    out.rho = s.rho;
    out.a = s.a;
    out.weight = bitcast<f32>(entry.y);
    out.fwd2 = s.fwd2;
    out.side2 = s.side2;
    out.flap = s.flap;
    return out;
}

@fragment
fn fs_impostor(in: ImpostorOut) -> BirdOut {
    // Fractions of the projected wingspan, measured from the bird's centre.
    let d = (in.clip.xy - in.centre) / max(in.rho, 1e-3);
    let du = dot(d, in.fwd2);
    let dv = dot(d, in.side2);
    let span = max(cos(in.flap), 0.25);

    // The two wing lobes, foreshortened along the wing axis.
    let su = WING_CHORD;
    let sv = WING_SPAN * span;
    let off = WING_OFFSET * span;
    let wing = exp(-(du * du / (2.0 * su * su) + (dv - off) * (dv - off) / (2.0 * sv * sv)))
        + exp(-(du * du / (2.0 * su * su) + (dv + off) * (dv + off) / (2.0 * sv * sv)));
    let body = exp(
        -(du * du / (2.0 * BODY_AXIS * BODY_AXIS) + dv * dv / (2.0 * BODY_ACROSS * BODY_ACROSS)),
    );
    let mask = 0.5 * body + 0.25 * wing;

    // The mixture's own integral over the screen, in units of ρ².
    let norm = 2.0 * PI * (0.5 * BODY_AXIS * BODY_ACROSS + 0.5 * su * sv);
    return bird_out(in.a * in.weight * mask / norm);
}
