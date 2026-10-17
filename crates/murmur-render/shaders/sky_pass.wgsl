// The sky pass: a full-screen triangle writes the dusk gradient into the HDR
// target, before the birds darken it (PLAN.md phase 5). Uses `fullscreen_triangle`,
// `ray_through` and `sky_linear_ray` from `scene.wgsl` and `sky.wgsl`.

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_triangle(vid);
}

@fragment
fn fs_sky(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    var colour = sky_linear_ray(ray_through(coord.xy));
    // Stars at night, in the upper half of the frame. Mirrors `render/sky.rs`'s
    // `star` hash: about 6 in 4096 pixels, lit as the evening deepens.
    if (scene.evening > 0.75 && coord.y < scene.viewport.y * 0.5) {
        let x = u32(coord.x);
        let y = u32(coord.y);
        var h = x * 73856093u ^ y * 19349663u;
        h ^= h >> 13u;
        h = h * 1274126177u;
        h ^= h >> 16u;
        if (h % 4096u < 6u) {
            let strength = clamp((scene.evening - 0.75) / 0.25, 0.0, 1.0);
            colour = mix(colour, vec3<f32>(0.85, 0.85, 1.0), strength);
        }
    }
    return vec4<f32>(colour, 1.0);
}
