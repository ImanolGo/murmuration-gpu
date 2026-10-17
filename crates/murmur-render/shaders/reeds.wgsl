// Foreground extension of sky.rs::reed_visibility / the reference's draw_reeds.
// Reeds occupy world bearings and elevations at three distances, rather than
// repeating two-pixel screen columns. Curved tapered stems, leaves and seed
// heads sway in a shared slow wind. They never enter the flock's optical depth.

struct ReedOut {
    @location(0) hdr: vec4<f32>,
    @location(1) optics: vec4<f32>,
};

@vertex
fn vs_reeds(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vid], 0.0, 1.0);
}

@fragment
fn fs_reeds(@builtin(position) coord: vec4<f32>) -> ReedOut {
    var out: ReedOut;
    out.hdr = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.optics = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (scene.reed < 0.002) {
        return out;
    }

    let ray = ray_through(coord.xy);
    let elevation = asin(clamp(ray.z, -1.0, 1.0));
    if elevation > 0.32 { return out; }
    let bearing = atan2(ray.y, ray.x);
    let aa = scene.fov / scene.viewport.y;
    var transmission = vec3<f32>(1.0);
    let distances = array<f32, 3>(35.0, 16.0, 7.0);
    for (var layer = 0u; layer < 3u; layer += 1u) {
        let distance = distances[layer];
        let spacing = 0.012 + f32(layer) * 0.007;
        let cell = i32(floor(bearing / spacing));
        for (var neighbour = -2; neighbour <= 2; neighbour += 1) {
            let index = cell + neighbour;
            var hash = bitcast<u32>(index) * 2654435761u + layer * 2246822519u;
            hash ^= hash >> 16u;
            hash *= 2246822519u;
            hash ^= hash >> 13u;
            let seed = f32(hash & 65535u) / 65535.0;
            let height = 0.8 + seed * 2.6;
            let bottom = -atan(1.7 / distance);
            let top = atan((height - 1.7) / distance);
            let angular_length = top - bottom;
            let t = (elevation - bottom) / angular_length;
            if t < 0.0 || t > 1.02 { continue; }
            let phase = f32((hash >> 16u) & 255u) * 0.0246;
            let wind = sin(scene.time * 0.53 + phase) * 0.13 + sin(scene.time * 0.21) * 0.22;
            let lean = (seed - 0.5) * 0.32 + wind;
            let origin = (f32(index) + 0.15 + seed * 0.7) * spacing;
            let bend = lean * angular_length;
            let centre = origin + bend * t * t;
            let width = (0.010 / distance) * max(1.0 - t, 0.08);
            var coverage = 1.0 - smoothstep(max(width - aa, 0.0), width + aa, abs(bearing - centre));
            // A narrow pointed leaf branches from the stem, not a round particle.
            if t > 0.22 && t < 0.72 {
                let leaf_t = (t - 0.22) / 0.50;
                let side = select(-1.0, 1.0, (hash & 1u) != 0u);
                let leaf_x = centre + side * sin(leaf_t * 3.14159265) * angular_length * 0.16;
                let leaf_width = (0.023 / distance) * sin(leaf_t * 3.14159265);
                coverage = max(coverage, 1.0 - smoothstep(max(leaf_width - aa, 0.0), leaf_width + aa, abs(bearing - leaf_x)));
            }
            if seed > 0.62 && t > 0.73 {
                let q = vec2<f32>((bearing - centre) / (0.025 / distance), (t - 0.86) / 0.12);
                coverage = max(coverage, 1.0 - smoothstep(0.75, 1.25, length(q)));
            }
            let haze = 0.65 + f32(layer) * 0.16;
            let absorption = vec3<f32>(0.98, 0.94, 0.99) * (coverage * haze * scene.reed);
            transmission *= 1.0 - absorption;
        }
    }
    out.hdr = vec4<f32>(1.0 - transmission, 1.0);
    return out;
}
