// The sky pass: a full-screen triangle writes the dusk gradient into the HDR
// target, before the birds darken it (PLAN.md phase 5). Uses `fullscreen_triangle`,
// `ray_through` and `sky_linear_ray` from `scene.wgsl` and `sky.wgsl`.

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_triangle(vid);
}

@fragment
fn fs_sky(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(sky_linear_ray(ray_through(coord.xy)), 1.0);
}
