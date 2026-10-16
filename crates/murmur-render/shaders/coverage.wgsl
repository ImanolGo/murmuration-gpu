// The bird coverage pass: one instanced quad per bird deposits optical-depth
// coverage into the HDR target and raw optical depth into the optics target.
//
// Mirrors the terminal reference's `render/splat.rs`:
//   * the raw optical depth is `a = (0.15 + 0.85 * |up·d̂|) * flap * atmosphere`,
//     the Costanzo et al. (2022) wing-exposure term;
//   * the footprint blends continuously between a bilinear sub-pixel splat and
//     a resolved Gaussian, both carrying the same optical mass
//     `mass = a * rho²` (the old one-pixel switch jumped ~57 % in darkness).
//
// The turning of the bird's body frame (`forward`, `side_of`, `body_up`) comes
// from `bird.wgsl`, the shared definition also used by the simulation, so the
// wing exposure is computed the same way the reference derives it.

@group(0) @binding(1) var<storage, read> birds: array<Bird>;

// `splat.rs` constants.
const WINGSPAN: f32 = 0.4;
const ATMOSPHERE: f32 = 700.0;

const TAU2: f32 = 6.283185307179586;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Projected centre in pixels and the footprint parameters: rho (projected
    // wingspan in pixels), optical mass, and the bilinear/Gaussian mix.
    @location(0) center: vec2<f32>,
    @location(1) foot: vec4<f32>,
};

struct FsOut {
    @location(0) hdr: vec4<f32>,
    @location(1) optics: vec4<f32>,
};

@vertex
fn vs_bird(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    var out: VsOut;
    let b = birds[iid];

    let focal = (scene.viewport.y * 0.5) / tan(scene.fov * 0.5);
    let rel = b.pos - scene.eye;
    let depth = dot(rel, scene.forward);
    if (depth <= 1.0) {
        // Behind the camera: push the whole quad off-screen.
        out.clip = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.center = vec2<f32>(0.0, 0.0);
        out.foot = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return out;
    }

    let cx = scene.viewport.x * 0.5 + dot(rel, scene.right) * focal / depth;
    let cy = scene.viewport.y * 0.5 - dot(rel, scene.up) * focal / depth;
    let rho = min(WINGSPAN * focal / depth, 16.0);

    let up = body_up(forward(b.vel), b.roll);
    let dhat = normalize(scene.eye - b.pos);
    let flap = 1.0 + scene.flap_amplitude * b.flap_amp * sin(b.flap_phase);
    let wing = clamp(abs(dot(up, dhat)) * flap, 0.0, 1.0);
    let atmosphere = clamp(1.0 - depth / ATMOSPHERE, 0.25, 1.0);
    let a = (0.15 + 0.85 * wing) * atmosphere;
    let mass = a * rho * rho;

    var mix = clamp(rho - 1.0, 0.0, 1.0);
    mix = mix * mix * (3.0 - 2.0 * mix);

    // The quad must cover the Gaussian (radius ~2.5 sigma) and, for sub-pixel
    // birds, the 2x2 bilinear neighbourhood.
    let r = max(1.0, 2.5 * rho * 0.5);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let off = corners[vid] * r;
    let px = cx + off.x;
    let py = cy + off.y;
    out.clip = vec4<f32>(
        px / scene.viewport.x * 2.0 - 1.0,
        1.0 - py / scene.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.center = vec2<f32>(cx, cy);
    out.foot = vec4<f32>(rho, mass, mix, 0.0);
    return out;
}

@fragment
fn fs_bird(in: VsOut) -> FsOut {
    let rho = in.foot.x;
    let mass = in.foot.y;
    let mix = in.foot.z;
    let d = in.clip.xy - in.center;
    let d2 = dot(d, d);

    var raw = 0.0;
    if (mix < 1.0) {
        // Bilinear sub-pixel splat: the four nearest pixel centres share the
        // mass with weights `wx * wy`.
        let wx = max(0.0, 1.0 - abs(d.x));
        let wy = max(0.0, 1.0 - abs(d.y));
        raw = raw + mass * (1.0 - mix) * wx * wy;
    }
    if (mix > 0.0) {
        let sigma = rho * 0.5;
        let peak = mass * mix / (TAU2 * sigma * sigma);
        raw = raw + peak * exp(-d2 / (2.0 * sigma * sigma));
    }

    // Fold the exposure in so the multiplicative blend over birds is exactly
    // `exp(-exposure * tau)`, the reference's additive compositor.
    let c = 1.0 - exp(-scene.exposure * raw);

    var out: FsOut;
    out.hdr = vec4<f32>(c, c, c, 1.0);
    out.optics = vec4<f32>(raw, 0.0, 0.0, 0.0);
    return out;
}
