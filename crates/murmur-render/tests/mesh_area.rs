//! The mesh's geometry and its optical-depth constant.
//!
//! The mesh deposits the model's *per-pixel* optical depth, `2a/π` — the
//! reference disc's peak — so a mesh bird is as dark per pixel as a dot bird of
//! the same size and the auto-exposure rule sees the optical depth it expects.
//! That constant is compared here against the shader's, so the two cannot drift.
//! The mean projected area is also measured: it is how much thinner than a disc
//! the silhouette is, which is why a mesh bird's *total* ink is lower.

use murmur_render::mesh;

/// The `const DISC_PEAK: f32 = …;` out of the mesh shader.
fn shader_constant(name: &str) -> f32 {
    let source = include_str!("../shaders/bird_mesh.wgsl");
    let line = source
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("const {name}:")))
        .unwrap_or_else(|| panic!("bird_mesh.wgsl declares {name}"));
    line.split('=')
        .nth(1)
        .expect("the const has a value")
        .trim()
        .trim_end_matches(';')
        .trim_end_matches('f')
        .parse::<f32>()
        .unwrap_or_else(|_| panic!("cannot parse {name} out of {line:?}"))
}

#[test]
fn the_shader_deposits_the_reference_disc_peak() {
    let shader = shader_constant("DISC_PEAK");
    assert!(
        (shader - mesh::DISC_PEAK).abs() < 1e-5,
        "bird_mesh.wgsl says DISC_PEAK = {shader}, the model's disc peak is {}",
        mesh::DISC_PEAK
    );
    // `mass / (2 pi sigma^2)` with `mass = a rho^2` and `sigma = rho / 2`.
    assert!((mesh::DISC_PEAK - 2.0 / std::f32::consts::PI).abs() < 1e-6);
}

#[test]
fn a_vertex_is_a_position_and_a_wing_selector() {
    assert_eq!(std::mem::size_of::<mesh::Vertex>(), 16);
    assert_eq!(mesh::VERTEX_COUNT, 30);
    assert_eq!(mesh::TRIANGLES, 10);
}

/// The mesh is a *thin* bird: its mean projected area is a fraction of the disc
/// of the same wingspan, which is why its total ink is lower at the same
/// per-pixel optical depth. Pinned so the difference stays a measured number.
#[test]
fn the_mesh_is_thinner_than_a_disc_of_its_wingspan() {
    let mesh_area = mesh::mean_projected_area();
    let disc_area = std::f32::consts::PI / 4.0; // a disc of diameter rho
    assert!(
        mesh_area > 0.02 && mesh_area < 0.3 * disc_area,
        "mesh mean area {mesh_area} vs the disc's {disc_area}"
    );
}
