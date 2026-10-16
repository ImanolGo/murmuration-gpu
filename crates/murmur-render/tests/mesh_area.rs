//! The mesh's optical-mass constant is a *measurement*, not a taste.
//!
//! The coverage pass gives every bird a disc of optical mass `a·ρ²`. The mesh
//! gives it a silhouette at a constant optical depth per fragment, so its
//! fragment shader divides by the mesh's mean projected area `ρ²`. That number
//! is computed from the vertex data (over viewing angles and wingbeat phases)
//! and compared here against the constant the shader actually uses, so the two
//! cannot drift apart.

use murmur_render::mesh;

/// The `const MEAN_AREA: f32 = …;` out of the mesh shader.
fn shader_constant() -> f32 {
    let source = include_str!("../shaders/bird_mesh.wgsl");
    let line = source
        .lines()
        .find(|l| l.trim_start().starts_with("const MEAN_AREA"))
        .expect("bird_mesh.wgsl declares MEAN_AREA");
    line.split('=')
        .nth(1)
        .and_then(|v| v.trim().trim_end_matches(';').parse::<f32>().ok())
        .unwrap_or_else(|| panic!("cannot parse MEAN_AREA out of {line:?}"))
}

#[test]
fn shader_mass_constant_is_the_measured_mean_projected_area() {
    let measured = mesh::mean_projected_area();
    let shader = shader_constant();
    assert!(
        (measured - shader).abs() < 0.01,
        "bird_mesh.wgsl says MEAN_AREA = {shader}, the geometry measures {measured}"
    );
}

#[test]
fn a_vertex_is_a_position_and_a_wing_selector() {
    assert_eq!(std::mem::size_of::<mesh::Vertex>(), 16);
    assert_eq!(mesh::VERTEX_COUNT, 30);
    assert_eq!(mesh::TRIANGLES, 10);
}

/// The mesh and the coverage splat must carry the same optical mass on average
/// (`a·ρ²` each): a bird that changes level must not change how much light it
/// absorbs *in total*. The splat spreads that mass over a disc of diameter `ρ`;
/// the mesh concentrates it through a silhouette, so at equal mass the mesh is
/// much darker per pixel — that difference in *shape* is the phase-6
/// cross-fade's reason to exist and is measured by `--lod-sweep`
/// (`docs/progress.md`, phase 6), not hidden here.
#[test]
fn the_mesh_and_a_splat_carry_the_same_mass_on_average() {
    let mesh_mass = mesh::mean_projected_area(); // units of ρ², mass = a × this
    let splat_mass = std::f32::consts::PI / 4.0; // a disc of diameter ρ
    // The mesh is thinner than a disc of the same wingspan: the wings are planes
    // seen from varying angles, not a filled circle.
    assert!(
        mesh_mass > 0.1 * splat_mass && mesh_mass < 2.0 * splat_mass,
        "mesh {mesh_mass} vs splat {splat_mass}"
    );
}
