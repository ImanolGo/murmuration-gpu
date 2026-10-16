// LOD level 0: the low-poly starling mesh (PLAN.md phase 6), one instanced draw
// for the whole flock.
//
// Mirrors `murmur_render::mesh` — the ten triangles, the wingbeat dihedral and
// the mean projected area — which `tests/mesh_area.rs` checks against this file.
// The wingbeat *phase and rate* are the model's (`flap_phase`, `FLAP_HZ = 10 Hz`);
// the dihedral amplitude is this renderer's own, because the reference draws
// dots and has no wings to rotate (`docs/model.md`).
//
// The bird deposits the same optical mass as a splat, `a·ρ²`, but through its
// silhouette: a constant optical depth `a / MEAN_AREA` per fragment, where
// MEAN_AREA is the mesh's mean projected area in units of ρ². Seen edge-on a
// bird carries less mass than seen from above, which is what an optically thin
// bird does; on average over angles it matches the splat, so switching levels
// does not change how dark the bird is.

@group(0) @binding(1) var<storage, read> birds: array<Bird>;
@group(0) @binding(2) var<storage, read> lod_lists: array<vec2<u32>>;

/// The mesh's mean projected silhouette area, in units of the projected
/// wingspan squared (`mesh::mean_projected_area`). Measured, not tuned.
const MEAN_AREA: f32 = 0.0851;

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    /// Optical depth per fragment, flat across the silhouette.
    @location(0) @interpolate(flat) raw: f32,
};

/// This pass draws level 0 of the instance lists.
const LEVEL: u32 = 0u;

@vertex
fn vs_mesh(
    @location(0) pos: vec3<f32>,
    @location(1) wing: f32,
    @builtin(instance_index) iid: u32,
) -> MeshOut {
    var out: MeshOut;
    let stride = arrayLength(&lod_lists) / 3u;
    let entry = lod_lists[LEVEL * stride + iid];
    let b = birds[entry.x];
    let weight = bitcast<f32>(entry.y);
    let s = bird_screen(b);

    let f = forward(b.vel);
    let side = side_of(f);
    let up = body_up(f, b.roll);

    // Wingbeat: both wings rise and fall together, so the dihedral is applied
    // with the vertex's own side (`mesh.wing`).
    let angle = s.flap * wing;
    let sn = sin(angle);
    let cs = cos(angle);
    let local = vec3<f32>(pos.x * cs - pos.z * sn, pos.y, pos.x * sn + pos.z * cs);
    let world = b.pos + f * local.y + side * local.x + up * local.z;

    let p = project_to(world);
    if (p.z <= 1.0) {
        out.clip = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.raw = 0.0;
        return out;
    }
    let px = p.x + scene.jitter.x;
    let py = p.y + scene.jitter.y;
    out.clip = vec4<f32>(
        px / scene.viewport.x * 2.0 - 1.0,
        1.0 - py / scene.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.raw = s.a * weight / MEAN_AREA;
    return out;
}

@fragment
fn fs_mesh(in: MeshOut) -> BirdOut {
    return bird_out(in.raw);
}
