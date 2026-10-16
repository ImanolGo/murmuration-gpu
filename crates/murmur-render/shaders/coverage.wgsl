// LOD level 2: the sub-pixel coverage splat (PLAN.md phase 5), drawn here from
// the LOD classifier's instance list and scaled by its cross-fade weight.
//
// Mirrors the terminal reference's `render/splat.rs`:
//   * the raw optical depth is `a = (0.15 + 0.85 * |up·d̂|) * flap * atmosphere`,
//     the Costanzo et al. (2022) wing-exposure term (`bird_pass.wgsl`);
//   * the footprint blends continuously between a bilinear sub-pixel splat and
//     a resolved Gaussian, both carrying the same optical mass
//     `mass = a * rho²` (the old one-pixel switch jumped ~57 % in darkness).
//
// This is the level every bird smaller than `RHO_SPLAT` is drawn with, and it
// carries exactly the mass the mesh and the impostor carry on average, so the
// level a bird is drawn at never changes how dark it is.

@group(0) @binding(1) var<storage, read> birds: array<Bird>;
@group(0) @binding(2) var<storage, read> lod_lists: array<vec2<u32>>;

const TAU2: f32 = 6.283185307179586;

/// This pass draws level 2 of the instance lists.
const LEVEL: u32 = 2u;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Projected centre in pixels and the footprint parameters: rho (projected
    // wingspan in pixels), optical mass, and the bilinear/Gaussian mix.
    @location(0) center: vec2<f32>,
    @location(1) foot: vec4<f32>,
};

@vertex
fn vs_bird(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    var out: VsOut;
    let stride = arrayLength(&lod_lists) / 3u;
    let entry = lod_lists[LEVEL * stride + iid];
    let b = birds[entry.x];
    let s = bird_screen(b);
    if (s.depth <= 1.0) {
        // Behind the camera: push the whole quad off-screen.
        out.clip = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.center = vec2<f32>(0.0, 0.0);
        out.foot = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return out;
    }

    let rho = s.rho;
    let mass = s.a * rho * rho * bitcast<f32>(entry.y);

    var mix = clamp(rho - 1.0, 0.0, 1.0);
    mix = mix * mix * (3.0 - 2.0 * mix);

    // The quad must cover the Gaussian (radius ~2.5 sigma) and, for sub-pixel
    // birds, the 2x2 bilinear neighbourhood.
    let r = max(1.0, 2.5 * rho * 0.5);
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
    out.center = s.centre;
    out.foot = vec4<f32>(rho, mass, mix, 0.0);
    return out;
}

@fragment
fn fs_bird(in: VsOut) -> BirdOut {
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
    return bird_out(raw);
}
