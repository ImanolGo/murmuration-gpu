//! Checks that the dusk sky palette has one source of truth: the numbers in
//! `shaders/sky.wgsl` must equal the Rust constants in `src/sky.rs` (which in
//! turn mirror the terminal reference). Editing one without the other fails
//! here.

use murmur_render::sky::{DAY_STOPS, DUSK_STOPS, NIGHT_STOPS};

const SHADER: &str = include_str!("../shaders/sky.wgsl");

/// The `vec4<f32>(t, r, g, b)` stops of `const <name>` in the shader, in order.
fn shader_stops(name: &str) -> Vec<[f32; 4]> {
    let header = format!("const {name}");
    let start = SHADER
        .find(&header)
        .unwrap_or_else(|| panic!("no `{header}` in the shader"));
    let rest = &SHADER[start..];
    let end = rest.find(");").expect("unterminated stop array");
    let region = &rest[..end];

    let mut stops = Vec::new();
    let mut cursor = region;
    while let Some(i) = cursor.find("vec4<f32>(") {
        let after = &cursor[i + "vec4<f32>(".len()..];
        let close = after.find(')').expect("unterminated vec4");
        let values: Vec<f32> = after[..close]
            .split(',')
            .map(|s| s.trim().parse().expect("a stop component"))
            .collect();
        assert_eq!(values.len(), 4, "a stop needs t, r, g, b");
        stops.push([values[0], values[1], values[2], values[3]]);
        cursor = &after[close..];
    }
    stops
}

/// Asserts a shader stop table equals a Rust palette: same count, same ramp
/// positions and the same colours as sRGB bytes divided by 255.
#[track_caller]
fn matches(rust: &[(f32, [u8; 3])], shader_name: &str) {
    let stops = shader_stops(shader_name);
    assert_eq!(
        stops.len(),
        rust.len(),
        "{shader_name}: stop count differs from the Rust palette"
    );
    for (i, (t, rgb)) in rust.iter().enumerate() {
        let got = stops[i];
        let want = [
            *t,
            f32::from(rgb[0]) / 255.0,
            f32::from(rgb[1]) / 255.0,
            f32::from(rgb[2]) / 255.0,
        ];
        for c in 0..4 {
            assert!(
                (got[c] - want[c]).abs() < 1e-5,
                "{shader_name} stop {i} component {c}: shader {} vs Rust {}",
                got[c],
                want[c],
            );
        }
    }
}

#[test]
fn day_palette_matches_shader() {
    matches(DAY_STOPS, "SKY_DAY");
}

#[test]
fn dusk_palette_matches_shader() {
    matches(DUSK_STOPS, "SKY_DUSK");
}

#[test]
fn night_palette_matches_shader() {
    matches(NIGHT_STOPS, "SKY_NIGHT");
}
