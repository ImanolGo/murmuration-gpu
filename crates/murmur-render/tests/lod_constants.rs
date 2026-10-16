//! The LOD classifier's constants are split between Rust and WGSL: the Rust
//! side sizes the buffers, dispatches the classify pass and builds the indirect
//! draws, the WGSL side decides which level a bird is drawn at. This test
//! parses the shader's copy of each number and compares it with the Rust one, so
//! a level can never be drawn by the wrong pipeline or a bird missed because the
//! dispatch did not cover the list.

use murmur_render::renderer::{LOD_LEVELS, LOD_WORKGROUP};

const LOD: &str = include_str!("../shaders/lod.wgsl");
const MESH: &str = include_str!("../shaders/bird_mesh.wgsl");
const IMPOSTOR: &str = include_str!("../shaders/impostor.wgsl");
const COVERAGE: &str = include_str!("../shaders/coverage.wgsl");

/// The value of `const <name>: <type> = <value>;` in a shader source.
fn shader_const(source: &str, name: &str) -> f32 {
    let line = source
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("const {name}:")))
        .unwrap_or_else(|| panic!("no `const {name}` in the shader"));
    let value = line
        .split('=')
        .nth(1)
        .expect("the const has a value")
        .trim()
        .trim_end_matches(';')
        .trim_end_matches(['u', 'f']);
    value
        .parse::<f32>()
        .unwrap_or_else(|_| panic!("cannot parse {name} out of {line:?}"))
}

/// The level a pass draws, from its `const LEVEL: u32 = k;`.
fn level_of(source: &str) -> u32 {
    shader_const(source, "LEVEL") as u32
}

#[test]
fn the_three_passes_draw_the_three_levels_in_order() {
    assert_eq!(level_of(MESH), 0, "the mesh is level 0");
    assert_eq!(level_of(IMPOSTOR), 1, "the impostor is level 1");
    assert_eq!(level_of(COVERAGE), 2, "the splat is level 2");
    // The renderer draws `LOD_LEVELS` indirect draws built from these levels, so
    // a fourth pass (or a missing one) must change both sides.
    assert_eq!(LOD_LEVELS, 3);
}

#[test]
fn the_dispatch_covers_the_workgroup_the_shader_declares() {
    let line = LOD
        .lines()
        .find(|l| l.contains("@workgroup_size"))
        .expect("lod.wgsl declares a workgroup size");
    let declared: u32 = line
        .split("@workgroup_size(")
        .nth(1)
        .and_then(|v| v.split(')').next())
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| panic!("cannot parse a workgroup size out of {line:?}"));
    assert_eq!(
        declared, LOD_WORKGROUP,
        "lod.wgsl classifies {declared} birds per workgroup, the renderer dispatches for {LOD_WORKGROUP}"
    );
}

/// The thresholds must leave each level a band of sizes of its own: a bird is
/// never classified into two levels' *exclusive* ranges at once, and the mesh
/// cross-fade sits above the splat's threshold rather than overlapping it.
#[test]
fn the_thresholds_do_not_overlap() {
    let splat = shader_const(LOD, "RHO_SPLAT");
    let mesh = shader_const(LOD, "RHO_MESH");
    let band = shader_const(LOD, "BAND");
    assert_eq!(splat, murmur_render::renderer::RHO_SPLAT);
    assert_eq!(mesh, murmur_render::renderer::RHO_MESH);
    assert_eq!(band, murmur_render::renderer::LOD_BAND);
    assert!(band > 0.0, "a cross-fade needs a width");
    assert!(
        mesh - band > splat,
        "the mesh cross-fade ({}) reaches down into the splat range (below {splat})",
        mesh - band
    );
    // The mesh is for birds that have resolved shape; the splat is for sub-pixel
    // ones. They cannot be the same size.
    assert!(mesh > 2.0 * splat, "mesh {mesh} vs splat {splat}");
}

/// The impostor's silhouette must be mass preserving by construction: its
/// normalisation divides by the mixture's own integral, so only the shapes of
/// the three Gaussians matter. This checks the shader still does that rather
/// than a tuned peak.
#[test]
fn the_impostor_normalises_its_own_mixture() {
    assert!(
        IMPOSTOR.contains("let norm = 2.0 * PI *"),
        "the impostor must divide the mixture by its own integral"
    );
    assert!(
        IMPOSTOR.contains("in.a * in.weight * mask / norm"),
        "the impostor's optical depth must be mass times weight over the integral"
    );
}
