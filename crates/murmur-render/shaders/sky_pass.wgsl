// The sky pass: a full-screen triangle writes the dusk sky and the land below
// it into the HDR target, before the birds darken it (PLAN.md phase 5). Uses
// `fullscreen_triangle`, `ray_through` and `sky_linear_ray` from `scene.wgsl`
// and `sky.wgsl`, `atmosphere_sky` from `atmosphere.wgsl` and `landscape` from
// `landscape.wgsl`.

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_triangle(vid);
}

// Stars, fixed to the sky rather than to the screen: the celestial sphere is
// split into cells of ~0.25° by elevation and bearing, a few of which hold a
// star at a random point. They come out as the evening deepens, twinkle a
// little, and are hidden by cloud and by the land.
fn stars(ray: vec3<f32>, sky: vec3<f32>, clear: vec3<f32>) -> vec3<f32> {
    let strength = smoothstep(0.72, 0.95, scene.evening);
    if strength <= 0.0 || ray.z < 0.02 {
        return sky;
    }
    let cell_size = 0.0045;
    let elevation = asin(clamp(ray.z, -1.0, 1.0));
    let bearing = atan2(ray.y, ray.x);
    // Bearing cells shrink towards the zenith so stars stay evenly spread.
    let ring = cos(elevation);
    let q = vec2<f32>(bearing * ring, elevation) / cell_size;
    let cell = vec2<i32>(floor(q));
    var lit = 0.0;
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let c = cell + vec2<i32>(dx, dy);
            let h = sky_hash(c);
            if h < 0.025 {
                let at = vec2<f32>(c) + vec2<f32>(sky_hash(c + vec2<i32>(17, 5)), sky_hash(c + vec2<i32>(3, 29)));
                let px = length(q - at) * cell_size / (scene.fov / scene.viewport.y);
                let size = 0.45 + 0.35 * sky_hash(c + vec2<i32>(11, 11));
                let twinkle = 0.8 + 0.2 * sin(scene.time * (2.0 + 160.0 * h) + h * 900.0);
                lit = max(lit, exp(-px * px / (size * size)) * (0.2 + 0.8 * pow(1.0 - h / 0.025, 3.0)) * twinkle);
            }
        }
    }
    // Cloud brightens the sky over the clear gradient: there, no stars.
    let cloud = clamp((dot(sky - clear, vec3<f32>(0.3, 0.5, 0.2))) * 60.0, 0.0, 1.0);
    let haze = smoothstep(0.02, 0.25, ray.z);
    return sky + vec3<f32>(0.80, 0.82, 1.0) * (lit * strength * haze * (1.0 - cloud) * 0.6);
}

@fragment
fn fs_sky(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let ray = ray_through(coord.xy);
    var colour = atmosphere_sky(ray);
    colour = stars(ray, colour, sky_linear_ray(ray));
    colour = landscape(ray, colour);
    return vec4<f32>(colour, 1.0);
}
