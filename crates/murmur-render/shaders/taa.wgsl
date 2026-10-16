// Temporal anti-aliasing (PLAN.md phase 6).
//
// A bird smaller than a pixel, or a silhouette edge inside one, samples a
// different set of pixels every frame: the flock sparkles even when the camera
// is still. The bird passes already spread a bird over its footprint; what is
// left is *temporal* aliasing, and this pass averages it away.
//
// Each frame the bird passes are offset by a sub-pixel `scene.jitter` (a Halton
// sequence, see `Renderer::render`), and this pass blends the fresh HDR frame
// into a running image: `accum = mix(history, current, alpha)`. The alpha comes
// from the frame time, so the window is a fraction of a second rather than a
// frame count, and the sky — which does not move — accumulates to itself.
//
// There are no motion vectors yet, so the history can be stale where the flock
// moved. The standard defence is used: the history is clamped into the range the
// current frame shows in the pixel's 3×3 neighbourhood, which rejects samples
// that no longer describe anything. Motion-vector reprojection (with a depth
// buffer) is listed as open in `docs/progress.md`.

@group(0) @binding(0) var current: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blend: vec4<f32>;

@vertex
fn vs_taa(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vid], 0.0, 1.0);
}

@fragment
fn fs_taa(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(current));
    let p = vec2<i32>(coord.xy);
    let fresh = textureLoad(current, p, 0);
    let past = textureLoad(history, p, 0);

    // The range the current frame shows around this pixel. The history must fall
    // inside it to be believable; outside means the pixel's content changed.
    var lo = fresh;
    var hi = fresh;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), dims - vec2<i32>(1));
            let s = textureLoad(current, q, 0);
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }

    return mix(clamp(past, lo, hi), fresh, blend.x);
}
