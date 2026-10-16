// The dusk sky gradient, mirrored from `crates/murmur-render/src/sky.rs` (which
// mirrors the terminal reference's `render/sky.rs`).
//
// The stop colours below are the reference's sRGB bytes as 0…1 values; they are
// checked against `sky.rs` by `crates/murmur-render/tests/sky_palette.rs`, so
// the palette has one source of truth and cannot drift between the two.
//
// Interpolation happens in sRGB space, exactly as the reference does, and the
// result is linearised only at the end, so a frame on an `*Srgb` surface shows
// the same colours as the terminal.

const SKY_DAY: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(0.000000, 0.305882, 0.549020, 0.839216),
    vec4<f32>(0.500000, 0.611765, 0.745098, 0.878431),
    vec4<f32>(0.800000, 0.972549, 0.807843, 0.556863),
    vec4<f32>(1.000000, 1.000000, 0.886275, 0.619608),
);

const SKY_DUSK: array<vec4<f32>, 5> = array<vec4<f32>, 5>(
    vec4<f32>(0.000000, 0.094118, 0.133333, 0.239216),
    vec4<f32>(0.450000, 0.380392, 0.388235, 0.529412),
    vec4<f32>(0.720000, 0.725490, 0.517647, 0.537255),
    vec4<f32>(0.900000, 0.905882, 0.678431, 0.505882),
    vec4<f32>(1.000000, 0.968627, 0.803922, 0.615686),
);

const SKY_NIGHT: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(0.000000, 0.007843, 0.015686, 0.047059),
    vec4<f32>(0.550000, 0.019608, 0.027451, 0.078431),
    vec4<f32>(0.820000, 0.039216, 0.047059, 0.109804),
    vec4<f32>(1.000000, 0.086275, 0.078431, 0.133333),
);

// Interpolates within a four-stop table at `t` (0 zenith … 1 horizon), in sRGB
// space, taking the first window that contains `t` like the reference.
fn ramp4(stops: array<vec4<f32>, 4>, t: f32) -> vec3<f32> {
    let tc = clamp(t, 0.0, 1.0);
    var lo = stops[0];
    var hi = stops[3];
    for (var i = 0u; i < 3u; i = i + 1u) {
        let a = stops[i];
        let b = stops[i + 1u];
        if (tc >= a.x && tc <= b.x) {
            lo = a;
            hi = b;
            break;
        }
    }
    let f = (tc - lo.x) / max(hi.x - lo.x, 1e-6);
    return vec3<f32>(lo.y, lo.z, lo.w) + (vec3<f32>(hi.y, hi.z, hi.w) - vec3<f32>(lo.y, lo.z, lo.w)) * f;
}

// The five-stop variant (dusk).
fn ramp5(stops: array<vec4<f32>, 5>, t: f32) -> vec3<f32> {
    let tc = clamp(t, 0.0, 1.0);
    var lo = stops[0];
    var hi = stops[4];
    for (var i = 0u; i < 4u; i = i + 1u) {
        let a = stops[i];
        let b = stops[i + 1u];
        if (tc >= a.x && tc <= b.x) {
            lo = a;
            hi = b;
            break;
        }
    }
    let f = (tc - lo.x) / max(hi.x - lo.x, 1e-6);
    return vec3<f32>(lo.y, lo.z, lo.w) + (vec3<f32>(hi.y, hi.z, hi.w) - vec3<f32>(lo.y, lo.z, lo.w)) * f;
}

// The sky colour at `t` for evening brightness (0 bright gold … 1 night), sRGB
// 0…1. Mirrors the reference's `sky_components`.
fn sky_srgb(t: f32, evening: f32) -> vec3<f32> {
    if (evening < 0.5) {
        let f = clamp(evening / 0.5, 0.0, 1.0);
        return mix(ramp4(SKY_DAY, t), ramp5(SKY_DUSK, t), f);
    }
    let f = clamp((evening - 0.5) / 0.5, 0.0, 1.0);
    return mix(ramp5(SKY_DUSK, t), ramp4(SKY_NIGHT, t), f);
}

// The sRGB transfer function on one channel.
fn srgb_channel(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

// The sky colour at `t`, linear 0…1, ready for the HDR target.
fn sky_linear(t: f32, evening: f32) -> vec3<f32> {
    let c = sky_srgb(t, evening);
    return vec3<f32>(srgb_channel(c.x), srgb_channel(c.y), srgb_channel(c.z));
}

// The sky colour along a unit view ray: elevation above the horizon sets the
// ramp parameter, so looking up does not carry the horizon along with the flock
// (the reference's `sky_at_components`).
fn sky_linear_ray(ray: vec3<f32>) -> vec3<f32> {
    let elevation = max(asin(clamp(ray.z, -1.0, 1.0)), 0.0);
    return sky_linear(1.0 - elevation / HALF_PI, scene.evening);
}
