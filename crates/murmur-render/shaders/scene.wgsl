// The per-frame camera block shared by every render pass, and the projection
// helpers the passes use.
//
// Mirrored in Rust by `murmur_render::renderer::Scene` (field for field) and
// checked by `crates/murmur-render/tests/scene_layout.rs`.

struct Scene {
    eye: vec3<f32>, _pad0: f32,
    forward: vec3<f32>, _pad1: f32,
    right: vec3<f32>, _pad2: f32,
    up: vec3<f32>, _pad3: f32,
    viewport: vec2<f32>, fov: f32, evening: f32,
    exposure: f32, flap_amplitude: f32, time: f32, _pad4: f32,
    // Sub-pixel offset of this frame's sample, applied by the bird passes. TAA
    // walks it through a jittered sequence so successive frames sample different
    // points inside a pixel; see `shaders/taa.wgsl`.
    jitter: vec2<f32>,
    // Whether a falcon is hunting, and where (`shaders/falcon.wgsl`).
    falcon: f32,
    // How visible the foreground reed bed is (0…1).
    reed: f32,
    falcon_pos: vec3<f32>,
    // Bank angle [rad], right wing down when positive.
    falcon_bank: f32,
    // Velocity [m/s]: heading and wing pose (`shaders/falcon.wgsl`).
    falcon_vel: vec3<f32>,
    // Uniform layout rounds a struct up to 16 bytes; this keeps the block the
    // same size in Rust and WGSL.
    _pad7: f32,
};

@group(0) @binding(0) var<uniform> scene: Scene;

const HALF_PI: f32 = 1.5707963267948966;

// The sign of a full-screen triangle's clip position from the vertex index.
// The three vertices cover the viewport with one oversized triangle, the usual
// full-screen-triangle trick.
fn fullscreen_triangle(vid: u32) -> vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vid], 0.0, 1.0);
}

// The unit view ray through a pixel centre, mirroring the reference's
// `sky::sky_at_components`: `forward` plus `right`/`up` scaled by
// `tan(fov/2)`, with the horizontal axis carrying the aspect ratio.
fn ray_through(coord: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(
        coord.x / scene.viewport.x * 2.0 - 1.0,
        1.0 - coord.y / scene.viewport.y * 2.0,
    );
    let tan_half = tan(scene.fov * 0.5);
    let aspect = scene.viewport.x / max(scene.viewport.y, 1.0);
    return normalize(
        scene.forward
            + scene.right * (ndc.x * tan_half * aspect)
            + scene.up * (ndc.y * tan_half),
    );
}
