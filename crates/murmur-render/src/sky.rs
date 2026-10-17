//! The dusk sky palette and its gradient.
//!
//! The colour stops and the evening blend are shared with the terminal
//! reference (murmuration's `render/sky.rs`): a day → dusk → night ramp sampled
//! by *viewing elevation*, not by screen row, so looking up does not carry the
//! orange horizon along with the flock. [`sky_srgb`] is the CPU mirror used for
//! tests and for the `dusk_clear` fallback; `shaders/sky.wgsl` carries the same
//! numbers for the GPU and is checked against them by
//! `tests/sky_palette.rs`.

/// How much of the foreground sky this palette describes. Strings are stored as
/// sRGB bytes, exactly as the terminal reference carries them.
type Rgb = [u8; 3];

/// Bright arrival sky, from the zenith (`t = 0`) to the horizon (`t = 1`).
pub const DAY_STOPS: &[(f32, Rgb)] = &[
    (0.00, [78, 140, 214]),
    (0.50, [156, 190, 224]),
    (0.80, [248, 206, 142]),
    (1.00, [255, 226, 158]),
];

/// Dusk sky (the display phase).
pub const DUSK_STOPS: &[(f32, Rgb)] = &[
    (0.00, [24, 34, 61]),
    (0.45, [97, 99, 135]),
    (0.72, [185, 132, 137]),
    (0.90, [231, 173, 129]),
    (1.00, [247, 205, 157]),
];

/// Night sky.
pub const NIGHT_STOPS: &[(f32, Rgb)] = &[
    (0.00, [2, 4, 12]),
    (0.55, [5, 7, 20]),
    (0.82, [10, 12, 28]),
    (1.00, [22, 20, 34]),
];

/// The colour the window clears to until the sky pass exists: the dusk ramp
/// at `t = 0.72`, the rose band above the horizon.
pub const DUSK_CLEAR: Rgb = DUSK_STOPS[2].1;

/// Interpolates within a stop table at `t` (`0` zenith … `1` horizon), in sRGB
/// space, exactly as the reference does, and returns `0…1` values (the
/// reference keeps them as `0…255` floats; the GPU works in `0…1`, so the
/// division happens here and the two sides stay comparable).
pub fn stops_colour(stops: &[(f32, Rgb)], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let mut lo = stops[0];
    let mut hi = stops[stops.len() - 1];
    for pair in stops.windows(2) {
        if t >= pair[0].0 && t <= pair[1].0 {
            lo = pair[0];
            hi = pair[1];
            break;
        }
    }
    let span = (hi.0 - lo.0).max(f32::EPSILON);
    let f = (t - lo.0) / span;
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f) / 255.0;
    [
        mix(lo.1[0], hi.1[0]),
        mix(lo.1[1], hi.1[1]),
        mix(lo.1[2], hi.1[2]),
    ]
}

/// Linear blend from `a` (`t = 0`) to `b` (`t = 1`).
fn blend(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let m = |x: f32, y: f32| x + (y - x) * t;
    [m(a[0], b[0]), m(a[1], b[1]), m(a[2], b[2])]
}

/// The sky colour at `t` for evening brightness (`0` bright gold … `1` night),
/// in sRGB 0…1. Mirrors the reference's `sky_components`.
pub fn sky_srgb(t: f32, evening: f32) -> [f32; 3] {
    let (a, b, f) = if evening < 0.5 {
        (
            stops_colour(DAY_STOPS, t),
            stops_colour(DUSK_STOPS, t),
            evening / 0.5,
        )
    } else {
        (
            stops_colour(DUSK_STOPS, t),
            stops_colour(NIGHT_STOPS, t),
            (evening - 0.5) / 0.5,
        )
    };
    blend(a, b, f)
}

/// The sky colour at `t` in linear space 0…1, for accumulation into the HDR
/// target and display on an `*Srgb` surface.
pub fn sky_linear(t: f32, evening: f32) -> [f32; 3] {
    sky_srgb(t, evening).map(|c| srgb_to_linear_f32(f64::from(c)) as f32)
}

/// The sky colour at `t` as sRGB bytes (the reference's `sky_colour`).
pub fn sky_colour(t: f32, evening: f32) -> Rgb {
    sky_srgb(t, evening).map(|v| (v * 255.0).round() as u8)
}

/// The elevation, in radians above the horizon, of a unit view ray.
pub fn elevation_of(ray: glam::Vec3) -> f32 {
    ray.z.clamp(-1.0, 1.0).asin().max(0.0)
}

/// The sky ramp parameter `t` for a view ray: `0` at the zenith, `1` at the
/// horizon and below.
pub fn ramp_of(ray: glam::Vec3) -> f32 {
    1.0 - elevation_of(ray) / std::f32::consts::FRAC_PI_2
}

/// How much of the foreground reed bed belongs in this view. Mirrors the
/// terminal reference's `sky::reed_visibility`: a soft angular band, so the
/// stems do not switch on and off when camera tracking crosses a threshold.
pub fn reed_visibility(camera: &crate::camera::Camera) -> f32 {
    let bottom = (camera.forward - camera.up * (camera.fov * 0.5).tan()).normalize();
    let t = ((0.25 - bottom.z) / 0.20).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The clear colour for a surface of `format`: the dusk colour in the
/// encoding wgpu expects.
///
/// Clear colours are interpreted in linear space for `*Srgb` formats and as
/// raw channel values otherwise, so the sRGB palette values are converted
/// only when the surface format decodes them.
pub fn dusk_clear(format: wgpu::TextureFormat) -> wgpu::Color {
    let [r, g, b] = DUSK_CLEAR;
    let (r, g, b) = if format.is_srgb() {
        (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
    } else {
        (
            f64::from(r) / 255.0,
            f64::from(g) / 255.0,
            f64::from(b) / 255.0,
        )
    };
    wgpu::Color { r, g, b, a: 1.0 }
}

/// sRGB byte to its linear value (the standard sRGB transfer function).
pub(crate) fn srgb_to_linear(byte: u8) -> f64 {
    srgb_to_linear_f32(f64::from(byte) / 255.0)
}

/// The sRGB transfer function on a 0…1 value, returning `f64`.
fn srgb_to_linear_f32(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_transfer_function_endpoints_and_midpoint() {
        assert_eq!(srgb_to_linear(0), 0.0);
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-12);
        // 128 encodes about 0.2158 linear.
        assert!((srgb_to_linear(128) - 0.2158).abs() < 1e-3);
        // Monotonic across the piecewise join.
        assert!(srgb_to_linear(10) < srgb_to_linear(11));
        assert!(srgb_to_linear(11) < srgb_to_linear(12));
    }

    #[test]
    fn clear_colour_matches_the_surface_encoding() {
        // An sRGB surface gets the linear value; a plain one the raw value.
        let srgb = dusk_clear(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(srgb.r, srgb_to_linear(185));
        assert_eq!(srgb.g, srgb_to_linear(132));
        assert_eq!(srgb.b, srgb_to_linear(137));
        let plain = dusk_clear(wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(plain.r, f64::from(185) / 255.0);
        assert_eq!(plain.g, f64::from(132) / 255.0);
    }

    #[test]
    fn gradient_endpoints_match_stops() {
        // Mirrors the reference's `gradient_endpoints_match_stops`.
        assert_eq!(sky_colour(0.0, 0.0), [78, 140, 214]);
        assert_eq!(sky_colour(1.0, 0.5), [247, 205, 157]);
    }

    #[test]
    fn the_horizon_is_warmer_than_the_zenith() {
        // Looking up must not carry the horizon colour; the horizon is always
        // far redder than the zenith at dusk.
        let zenith = sky_linear(ramp_of(glam::Vec3::Z), 0.5);
        let horizon = sky_linear(ramp_of(glam::Vec3::Y), 0.5);
        assert!(
            horizon[0] > zenith[0] * 3.0,
            "horizon not warmer: {horizon:?} vs {zenith:?}"
        );
    }
}
