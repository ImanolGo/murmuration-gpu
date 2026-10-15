//! The dusk sky palette.
//!
//! The colour stops are shared with the terminal reference (murmuration's
//! `render/sky.rs` dusk ramp) so both projects show the same evening; phase 5
//! samples them by viewing elevation. Until the sky pass exists, the app
//! clears to a single dusk colour: the rose band at `t = 0.72`.

/// Dusk sky stops, from the zenith (`t = 0`) to the horizon (`t = 1`).
pub const DUSK_STOPS: &[(f32, [u8; 3])] = &[
    (0.00, [24, 34, 61]),
    (0.45, [97, 99, 135]),
    (0.72, [185, 132, 137]),
    (0.90, [231, 173, 129]),
    (1.00, [247, 205, 157]),
];

/// The colour the window clears to until the sky pass exists: the dusk ramp
/// at `t = 0.72`, the rose band above the horizon.
pub const DUSK_CLEAR: [u8; 3] = DUSK_STOPS[2].1;

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
fn srgb_to_linear(byte: u8) -> f64 {
    let c = f64::from(byte) / 255.0;
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
}
