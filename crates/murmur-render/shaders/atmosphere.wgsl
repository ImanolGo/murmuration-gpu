// Mirrors atmosphere.rs::colour: world-direction cloud layers and low-sun light.
// Three value-noise octaves; no ray marching, no external photographic texture.
fn sky_hash(cell: vec2<i32>) -> f32 {
    let u = bitcast<vec2<u32>>(cell);
    var h = u.x * 1597334677u ^ u.y * 3812015801u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h ^= h >> 13u;
    return f32(h & 65535u) / 65535.0;
}

fn sky_noise(p: vec2<f32>) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(sky_hash(cell), sky_hash(cell + vec2<i32>(1, 0)), u.x);
    let b = mix(sky_hash(cell + vec2<i32>(0, 1)), sky_hash(cell + vec2<i32>(1, 1)), u.x);
    return mix(a, b, u.y);
}

fn sky_clouds(p: vec2<f32>) -> f32 {
    return sky_noise(p) * 0.56 + sky_noise(p * 2.03 + 17.0) * 0.28
        + sky_noise(p * 4.11 - 9.0) * 0.16;
}

fn atmosphere_sky(ray: vec3<f32>) -> vec3<f32> {
    let sky = sky_linear_ray(ray);
    let light = 1.0 - smoothstep(0.55, 1.0, scene.evening) * 0.97;
    let sun = normalize(vec3<f32>(-0.65, 0.70, 0.10));
    let facing = max(dot(ray, sun), 0.0);
    let horizon_glow = exp(-max(ray.z, 0.0) * 7.0);
    let uv = ray.xy / (max(ray.z, 0.0) + 0.28) * vec2<f32>(2.8, 4.2)
        + vec2<f32>(scene.time * 0.0035, -scene.time * 0.0012);
    let field = sky_clouds(uv);
    let cover = smoothstep(0.46, 0.78, field) * 0.42 * smoothstep(0.02, 0.28, ray.z);
    let edge = clamp(0.55 + (sky_clouds(uv + vec2<f32>(0.045, 0.025)) - field) * 4.0, 0.15, 1.0);
    let cloud = mix(vec3<f32>(0.24, 0.26, 0.34), vec3<f32>(0.64, 0.58, 0.62), edge) * light
        + vec3<f32>(0.22, 0.085, 0.030) * (pow(facing, 4.0) * horizon_glow * light);
    var result = mix(sky, cloud, cover)
        + vec3<f32>(0.22, 0.11, 0.035) * (exp(-(1.0 - facing) * 10.0) * light * (1.0 - cover));
    let bearing = atan2(ray.y, ray.x);
    let skyline = 0.002 + sky_noise(vec2<f32>(bearing * 2.5, 7.0)) * 0.006;
    if ray.z < skyline {
        let ground = vec3<f32>(0.012, 0.018, 0.024) * light + sky * 0.06;
        result = mix(ground, result, smoothstep(skyline - 0.003, skyline + 0.001, ray.z));
    }
    return result;
}
