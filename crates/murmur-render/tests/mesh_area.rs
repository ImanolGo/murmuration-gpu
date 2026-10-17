//! The mesh's geometry and its optical-depth constant.
//!
//! Resolved silhouettes use a near-opaque material; wing exposure comes from
//! their projected area. The CPU optical-depth definition is checked against
//! the shader, and the geometry's mean area is still independently measured.

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
fn the_resolved_silhouette_material_matches_the_cpu_definition() {
    let shader = shader_constant("MESH_TAU");
    assert!(
        (shader - mesh::MESH_TAU).abs() < 1e-5,
        "shader MESH_TAU {shader}, CPU {}",
        mesh::MESH_TAU
    );
    assert!(((-mesh::MESH_TAU).exp() - 0.01).abs() < 1e-6);
    // `mass / (2 pi sigma^2)` with `mass = a rho^2` and `sigma = rho / 2`.
    assert!((mesh::DISC_PEAK - 2.0 / std::f32::consts::PI).abs() < 1e-6);
}

#[test]
fn intermediate_silhouette_matches_the_cpu_definition() {
    let source = include_str!("../shaders/impostor.wgsl");
    for (name, expected) in mesh::IMPOSTOR_SHAPE {
        let line = source
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("const {name}:")))
            .unwrap();
        let value: f32 = line
            .split('=')
            .nth(1)
            .unwrap()
            .trim()
            .trim_end_matches(';')
            .parse()
            .unwrap();
        assert!((value - expected).abs() < 1e-6, "{name}");
    }
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
