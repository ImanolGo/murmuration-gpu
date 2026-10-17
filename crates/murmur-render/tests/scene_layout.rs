//! Checks the shared camera block against the shader (AGENTS.md: GPU structs are
//! defined once in Rust and checked against WGSL by a test).
//!
//! `check_struct!` asserts, for every field, both its byte offset in the Rust
//! type and that the shader declares the fields in the same order with the same
//! names. Renaming or reordering either side without the other fails here.
//!
//! The `_pad0…_pad4` fields are what make a Rust `[f32; 3]` (4-byte aligned,
//! 12 bytes) occupy the same 16-byte slot a WGSL `vec3<f32>` does, so the
//! offsets below and the shader's agree.

use std::mem::{offset_of, size_of};

use murmur_render::renderer::Scene;

/// The shader every render struct is checked against.
const SHADER: &str = include_str!("../shaders/scene.wgsl");

/// Extracts the field names of `struct <name> { ... }` from WGSL, handling
/// several fields per line and ignoring comments.
fn wgsl_fields(source: &str, name: &str) -> Vec<String> {
    let header = format!("struct {name} {{");
    let start = source
        .find(&header)
        .unwrap_or_else(|| panic!("no `{header}` in the shader"));
    let body = &source[start + header.len()..];
    let end = body.find('}').expect("unterminated struct");
    let body: String = body[..end]
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ");
    body.split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .map(|field| field.split(':').next().unwrap_or("").trim().to_string())
        .collect()
}

macro_rules! check_struct {
    ($ty:ty, $name:literal, [$(($field:ident, $offset:expr)),+ $(,)?]) => {{
        $(
            assert_eq!(
                offset_of!($ty, $field),
                $offset,
                concat!("Rust offset of ", stringify!($field))
            );
        )+
        let want: Vec<String> = vec![$(stringify!($field).to_string()),+];
        assert_eq!(
            wgsl_fields(SHADER, $name),
            want,
            concat!("WGSL `struct ", $name, "` no longer matches the Rust ", stringify!($ty))
        );
    }};
}

#[test]
fn scene_matches_wgsl() {
    assert_eq!(size_of::<Scene>(), 128);
    check_struct!(
        Scene,
        "Scene",
        [
            (eye, 0),
            (_pad0, 12),
            (forward, 16),
            (_pad1, 28),
            (right, 32),
            (_pad2, 44),
            (up, 48),
            (_pad3, 60),
            (viewport, 64),
            (fov, 72),
            (evening, 76),
            (exposure, 80),
            (flap_amplitude, 84),
            (time, 88),
            (_pad4, 92),
            (jitter, 96),
            (falcon, 104),
            (_pad5, 108),
            (falcon_pos, 112),
            (_pad6, 124),
        ]
    );
}
