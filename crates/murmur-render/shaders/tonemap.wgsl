// The tone-map pass: writes the linear HDR target to the output surface.
//
// The HDR target holds `sky * prod(1 - coverage)` in linear space. That is
// always within `[0, 1]` — no bird adds light — so the map to the display range
// is the clamp, and the `*Srgb` surface's encoding does the transfer to the
// authored palette. A filmic curve would only shift colours the reference
// authored in display space; the scene gains values above one once bloom and
// emissives arrive in the later phases, which is when a real curve is needed.

@group(0) @binding(0) var hdr: texture_2d<f32>;

@vertex
fn vs_tonemap(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vid], 0.0, 1.0);
}

@fragment
fn fs_tonemap(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(hdr, vec2<i32>(coord.xy), 0).rgb;
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
