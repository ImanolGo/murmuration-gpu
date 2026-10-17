//! Checks the GPU structs against the shader (AGENTS.md: `Params` and `Bird`
//! are defined once in Rust and checked against WGSL by a test).
//!
//! `check_struct!` asserts, for every field, both its byte offset in the Rust
//! type and that the shader declares the fields in the same order with the same
//! names. Renaming or reordering either side without the other fails here.

use std::mem::{offset_of, size_of};

use murmur_gpu::arc::ArcState;
use murmur_gpu::bird::Bird;
use murmur_gpu::falcon::Falcon;
use murmur_gpu::params::Params;

/// The shader every GPU struct is checked against: `common.wgsl` carries
/// `Params` and `Falcon`, `bird.wgsl` carries `Bird`.
const SHADER: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    "\n",
    include_str!("../shaders/bird.wgsl"),
    "\n",
    include_str!("../shaders/arc.wgsl"),
);

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
fn params_matches_wgsl() {
    assert_eq!(size_of::<Params>(), 224);
    check_struct!(
        Params,
        "Params",
        [
            (dt, 0),
            (du, 4),
            (time, 8),
            (turning, 12),
            (v0, 16),
            (mass, 20),
            (g, 24),
            (l0, 28),
            (d0, 32),
            (tau, 36),
            (tau_roll, 40),
            (max_roll_rate, 44),
            (tau_spin, 48),
            (max_bank, 52),
            (v_min, 56),
            (v_max, 60),
            (n_c, 64),
            (r_max, 68),
            (radius_lerp, 72),
            (r_h, 76),
            (sigma, 80),
            (w_sep, 84),
            (w_align, 88),
            (w_coh, 92),
            (blind_cos, 96),
            (w_xi, 100),
            (r_roost, 104),
            (w_roost_h, 108),
            (roost_turn_margin, 112),
            (w_roost_v, 116),
            (h_roost_v, 120),
            (z0, 124),
            (flap_hz, 128),
            (flap_amplitude, 132),
            (r_threat, 136),
            (w_flee, 140),
            (r_flash, 144),
            (v_flash, 148),
            (w_flash, 152),
            (flash_duration, 156),
            (zigzag_max_roll, 160),
            (zigzag_duration, 164),
            (zigzag_cooldown, 168),
            (reed_top, 172),
            (n, 176),
            (zigzag_k, 180),
            (seed, 184),
            (buckets, 188),
            (inv_cell, 192),
            (falcon_cruise, 196),
            (falcon_climb, 200),
            (falcon_stoop, 204),
            (falcon_attack_height, 208),
            (falcon_attack_min, 212),
            (falcon_attack_max, 216),
            (pad3, 220),
        ]
    );
}

#[test]
fn bird_matches_wgsl() {
    assert_eq!(size_of::<Bird>(), 112);
    // The three vec3 fields sit on 16-byte boundaries, so Rust's 4-byte
    // alignment and WGSL's 16-byte vec3 alignment agree.
    check_struct!(
        Bird,
        "Bird",
        [
            (pos, 0),
            (roll, 12),
            (vel, 16),
            (spin, 28),
            (steering, 32),
            (radius, 44),
            (next_decision, 48),
            (centrality, 52),
            (flap_phase, 56),
            (flap_amp, 60),
            (neighbour_count, 64),
            (returning, 68),
            (id, 72),
            (rng, 76),
            (alarmed, 80),
            (zigzag_active, 84),
            (zigzag_sign, 88),
            (zigzag_start, 92),
            (cooldown_until, 96),
            (flash_until, 100),
            (live, 104),
            (pad1, 108),
        ]
    );
}

#[test]
fn arc_matches_wgsl() {
    assert_eq!(size_of::<ArcState>(), 48);
    check_struct!(
        ArcState,
        "ArcState",
        [
            (z0, 0),
            (duration, 4),
            (bearing, 8),
            (radius, 12),
            (spawn_timer, 16),
            (next, 20),
            (epoch, 24),
            (rng, 28),
            (present, 32),
            (pad0, 36),
            (pad1, 40),
            (pad2, 44),
        ]
    );
}

#[test]
fn falcon_matches_wgsl() {
    assert_eq!(size_of::<Falcon>(), 96);
    check_struct!(
        Falcon,
        "Falcon",
        [
            (pos, 0),
            (speed, 12),
            (vel, 16),
            (mode, 28),
            (aim, 32),
            (state, 44),
            (dive, 48),
            (until, 60),
            (centroid, 64),
            (rng, 76),
            (extent, 80),
            (present, 84),
            (pad0, 88),
            (pad1, 92),
        ]
    );
}
