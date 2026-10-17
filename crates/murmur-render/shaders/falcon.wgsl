// The falcon, drawn a little larger and darker than a starling (PLAN.md phase
// 7). Mirrors the terminal reference's `render/splat.rs`: `a = 1.6 * atmosphere`,
// a wingspan of `FALCON_WINGSPAN` and the same continuous unresolved/resolved
// footprint the coverage pass uses (`shaders/coverage.wgsl`), so it neither
// jumps in darkness nor shimmers.
//
// It reads the scene only — the falcon's position is a uniform, not a bird — so
// it needs no bird storage buffer and its own one-binding layout.

const FALCON_WINGSPAN: f32 = 1.0;
const TAU2: f32 = 6.283185307179586;

struct FalconOut {
    @builtin(position) clip: vec4<f32>,
    /// Projected centre \[px\] and the footprint parameters.
    @location(0) center: vec2<f32>,
    @location(1) foot: vec4<f32>,
};

@vertex
fn vs_falcon(@builtin(vertex_index) vid: u32) -> FalconOut {
    var out: FalconOut;
    let rel = scene.falcon_pos - scene.eye;
    let depth = dot(rel, scene.forward);
    if scene.falcon < 0.5 || depth <= 1.0 {
        // No falcon, or it is behind the camera: push the quad off-screen.
        out.clip = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.center = vec2<f32>(0.0, 0.0);
        out.foot = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return out;
    }
    let inv = focal_px() / max(depth, 1.0);
    let centre = vec2<f32>(
        scene.viewport.x * 0.5 + dot(rel, scene.right) * inv,
        scene.viewport.y * 0.5 - dot(rel, scene.up) * inv,
    ) + scene.jitter;
    let rho = min(FALCON_WINGSPAN * inv, 24.0);
    let atmosphere = clamp(1.0 - depth / ATMOSPHERE, 0.25, 1.0);
    let mass = 1.6 * atmosphere * rho * rho;

    var mix = clamp(rho - 1.0, 0.0, 1.0);
    mix = mix * mix * (3.0 - 2.0 * mix);

    let r = max(1.0, 2.5 * rho * 0.5);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0),
    );
    let off = corners[vid] * r;
    let px = centre.x + off.x;
    let py = centre.y + off.y;
    out.clip = vec4<f32>(
        px / scene.viewport.x * 2.0 - 1.0,
        1.0 - py / scene.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.center = centre;
    out.foot = vec4<f32>(rho, mass, mix, 0.0);
    return out;
}

@fragment
fn fs_falcon(in: FalconOut) -> BirdOut {
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
