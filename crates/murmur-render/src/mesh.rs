//! The starling mesh (PLAN.md phase 6): a ten-triangle bird drawn once per
//! instance, in the bird's own flight frame.
//!
//! The terminal reference has no mesh — it draws braille dots — so this
//! geometry is new. What it does take from the reference is the *wingbeat*: the
//! dihedral angle is driven by the model's `flap_phase` and `flap_amp` (10 Hz,
//! `FLAP_HZ`), the same two numbers `render/splat.rs` uses to flicker the wing
//! exposure. the reference scales an *exposure*; a mesh needs a *geometry*, so
//! the peak dihedral ([`FLAP_DIHEDRAL`]) is this renderer's own constant and is
//! recorded in `docs/model.md`.
//!
//! Local frame (metres, right-handed, matching `forward`/`side_of`/`body_up`):
//! `+Y` out of the nose, `+X` along the right wing, `+Z` up out of the back.
//! The tips sit at `±0.2 m`, so the wingspan is exactly the `WINGSPAN = 0.4 m`
//! the coverage pass uses to size a bird on screen.
//!
//! Optical depth: a bird's optical depth in the model is `a`, and the reference
//! spreads the mass `a·ρ²` over a disc of diameter `ρ`, whose peak is
//! `2a/π` ([`DISC_PEAK`]). The mesh deposits that same peak per fragment
//! through its silhouette — the *per-pixel* optical depth the model's exposure
//! rule expects, so the auto-exposure behaves as it does for a dot flock. The
//! mesh's total mass is then smaller than a disc's (its silhouette is thinner
//! than a disc of the same wingspan), which is what an actual bird looks like:
//! `docs/model.md` records this and `tests/mesh_area.rs` pins the constant.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Peak wing dihedral \[rad\]. The reference has no wings to rotate, so this is
/// a renderer constant; the phase and rate come from the model. See the module
/// docs and `docs/model.md` ("Phase 6 adaptations").
pub const FLAP_DIHEDRAL: f32 = 0.7;

/// The reference disc's peak optical depth per unit `a`: `mass / (2πσ²)` with
/// `mass = a·ρ²` and `σ = ρ/2`. The mesh deposits this per fragment, so a mesh
/// bird is as dark per pixel as a dot bird of the same size. Checked against the
/// shader's `DISC_PEAK` by `tests/mesh_area.rs`.
pub const DISC_PEAK: f32 = 2.0 / std::f32::consts::PI;

/// Starling wingspan \[m\]: the tip-to-tip width of [`VERTICES`], and the same
/// number the coverage pass sizes a bird on screen with. Checked against the
/// shader's `WINGSPAN` by `tests/mesh_area.rs`.
pub const WINGSPAN: f32 = 0.4;

/// A mesh vertex: position in the bird's frame, plus the wing selector.
///
/// `wing` is `1.0` for the right wing, `-1.0` for the left and `0.0` for the
/// body. The vertex shader rotates a wing vertex about the bird's forward axis
/// by `flap * wing`, so both wings rise and fall together.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Position in the bird's frame \[m\].
    pub pos: [f32; 3],
    /// Wing selector: `-1` left wing, `0` body, `+1` right wing.
    pub wing: f32,
}

/// Body vertices: nose, tail left, tail right, belly.
const NOSE: Vertex = v([0.000, 0.080, 0.005], 0.0);
const TAIL_L: Vertex = v([-0.012, -0.075, 0.000], 0.0);
const TAIL_R: Vertex = v([0.012, -0.075, 0.000], 0.0);
const BELLY: Vertex = v([0.000, -0.020, -0.025], 0.0);

/// Right wing vertices: root front, root back, mid front, mid back, tip.
const ROOT_F: Vertex = v([0.020, 0.035, 0.008], 1.0);
const ROOT_B: Vertex = v([0.020, -0.030, 0.008], 1.0);
const MID_F: Vertex = v([0.105, 0.012, 0.010], 1.0);
const MID_B: Vertex = v([0.105, -0.070, 0.010], 1.0);
const TIP: Vertex = v([0.200, -0.060, 0.012], 1.0);

/// The same five wing vertices mirrored onto the left wing (`wing = -1`).
const fn mirror(p: Vertex) -> Vertex {
    Vertex {
        pos: [-p.pos[0], p.pos[1], p.pos[2]],
        wing: -p.wing,
    }
}

const fn v(pos: [f32; 3], wing: f32) -> Vertex {
    Vertex { pos, wing }
}

/// The mesh: ten triangles, thirty vertices, no index buffer.
///
/// Winding is not meaningful here: the passes draw with culling off, because the
/// observer looks up at the flock and a flat wing is legitimately seen from
/// either side. A closed body therefore deposits on both of its faces, which is
/// part of the measured [`mean_projected_area`] rather than a special case.
pub const VERTICES: &[Vertex] = &[
    // Body: a tetrahedron (four triangles, closed).
    NOSE,
    TAIL_L,
    TAIL_R, //
    NOSE,
    TAIL_R,
    BELLY, //
    NOSE,
    BELLY,
    TAIL_L, //
    TAIL_L,
    BELLY,
    TAIL_R, //
    // Right wing: root, then the swept tip (three triangles).
    ROOT_F,
    ROOT_B,
    MID_F, //
    ROOT_B,
    MID_B,
    MID_F, //
    MID_F,
    MID_B,
    TIP, //
    // Left wing: the same, mirrored.
    mirror(ROOT_F),
    mirror(ROOT_B),
    mirror(MID_F),
    mirror(ROOT_B),
    mirror(MID_B),
    mirror(MID_F),
    mirror(MID_F),
    mirror(MID_B),
    mirror(TIP),
];

/// Vertices per instance (one triangle list, no index buffer).
pub const VERTEX_COUNT: u32 = VERTICES.len() as u32;

/// Triangles per instance, for the record and for the layout test.
pub const TRIANGLES: u32 = VERTEX_COUNT / 3;

/// The peak dihedral the shader uses, in degrees, for the documentation.
pub const FLAP_DIHEDRAL_DEG: f32 = FLAP_DIHEDRAL * 180.0 / std::f32::consts::PI;

/// Turns a local vertex into the bird's frame, with the wing at `flap` \[rad\].
///
/// The mirror of the shader's `mesh_vertex`: a wing vertex is rotated about the
/// forward axis (`+Y`) by `flap * wing`, which lifts the right wing by `flap` and
/// the left by `-flap` — the same direction for both tips. Body vertices have
/// `wing == 0` and do not move.
pub fn mesh_vertex(vertex: Vertex, flap: f32) -> Vec3 {
    let p = Vec3::from_array(vertex.pos);
    let angle = flap * vertex.wing;
    let (sin, cos) = angle.sin_cos();
    // Rotation about +Y by `angle`, taking +X towards +Z.
    Vec3::new(p.x * cos - p.z * sin, p.y, p.x * sin + p.z * cos)
}

/// The mesh's projected silhouette area on a screen where the wingspan is one
/// unit (so the area is in units of `ρ²`).
///
/// `view` is the unit direction from the bird towards the observer, expressed in
/// the bird's frame. Triangles are projected onto the plane perpendicular to it
/// and their areas summed; a silhouette that overlaps itself is over-counted,
/// which is right for an optically thin deposit (the shader deposits per
/// fragment, and an overlap deposits twice).
pub fn projected_area(view: Vec3, flap: f32) -> f32 {
    let n = view.normalize_or_zero();
    // Two orthonormal axes spanning the screen plane.
    let up = if n.z.abs() > 0.9 { Vec3::X } else { Vec3::Z };
    let e1 = n.cross(up).normalize_or_zero();
    let e2 = n.cross(e1).normalize_or_zero();
    let mut area = 0.0;
    let triangles = VERTICES.as_chunks::<3>().0;
    for t in triangles {
        let a = mesh_vertex(t[0], flap);
        let b = mesh_vertex(t[1], flap);
        let c = mesh_vertex(t[2], flap);
        // The signed projected area; `abs` handles the outward direction.
        area += ((b - a).dot(e1) * (c - a).dot(e2) - (b - a).dot(e2) * (c - a).dot(e1)).abs() * 0.5;
    }
    // Scale from metres² to units of the wingspan squared.
    area / WINGSPAN.powi(2)
}

/// The mean projected area over viewing angles and wingbeat phases: a
/// *measurement* of how much thinner than a disc (`π/4`) the mesh is, kept so
/// the difference in total ink between a mesh bird and a dot bird is a number
/// and not an impression.
pub fn mean_projected_area() -> f32 {
    let mut total = 0.0;
    for i in 0..AREA_SAMPLES {
        // Fibonacci sphere: a deterministic, evenly spread set of directions.
        let k = (i as f32 + 0.5) / AREA_SAMPLES as f32;
        let z = 1.0 - 2.0 * k;
        let r = (1.0 - z * z).sqrt();
        let phi = std::f32::consts::TAU * i as f32 * 0.618_034;
        let view = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        for j in 0..4 {
            let flap = FLAP_DIHEDRAL * (j as f32 / 4.0 * std::f32::consts::TAU).sin();
            total += projected_area(view, flap);
        }
    }
    total / (AREA_SAMPLES as f32 * 4.0)
}

/// Viewing directions sampled by [`mean_projected_area`].
pub const AREA_SAMPLES: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mesh_is_a_bird_sized_ten_triangles() {
        assert_eq!(TRIANGLES, 10);
        for v in VERTICES {
            assert!(v.pos[0].abs() <= 0.2 + 1e-6, "inside the wingspan: {v:?}");
            assert!(v.pos[1].abs() <= 0.09, "inside the body length: {v:?}");
        }
        // Tips at exactly half the wingspan, so `ρ` in the coverage pass is the
        // mesh's own width.
        assert!((TIP.pos[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn wings_rise_together_and_the_body_stays_put() {
        let flap = 0.5;
        for wing in [1.0, -1.0] {
            let tip = mesh_vertex(
                Vertex {
                    pos: [0.2 * wing, 0.0, 0.0],
                    wing,
                },
                flap,
            );
            assert!(tip.z > 0.09, "wing {wing} did not rise: {tip:?}");
        }
        assert_eq!(mesh_vertex(BELLY, flap), Vec3::from_array(BELLY.pos));
    }
}
