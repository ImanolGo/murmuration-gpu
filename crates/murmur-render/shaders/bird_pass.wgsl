// The per-bird screen setup shared by the three LOD passes (PLAN.md phase 6),
// so the projection, the wing-exposure term and the wingbeat are written once.
//
// `bird_screen` mirrors `crates/murmur-render/src/renderer.rs`'s `Scene` view of
// a bird and the reference's `render/splat.rs` term
// `a = (0.15 + 0.85·|up·d̂|·flap) · atmosphere`; the LOD passes then differ only
// in how they deposit that bird's optical mass.

const WINGSPAN: f32 = 0.4;
const ATMOSPHERE: f32 = 700.0;

/// Peak wing dihedral \[rad\]: this renderer's own constant, since the reference
/// draws dots and has no wings. Mirrored by `mesh::FLAP_DIHEDRAL`, checked by
/// `tests/mesh_area.rs`.
const FLAP_DIHEDRAL: f32 = 0.7;

struct BirdScreen {
    /// Projected centre \[px\], jitter included.
    centre: vec2<f32>,
    /// Depth along the view direction \[m\].
    depth: f32,
    /// Projected wingspan \[px\].
    rho: f32,
    /// Wing exposure at distance: the bird's optical mass is `a·rho²`.
    a: f32,
    /// Wing dihedral this frame \[rad\].
    flap: f32,
    /// The bird's forward direction, projected into screen space.
    fwd2: vec2<f32>,
    /// Its wing direction, projected into screen space.
    side2: vec2<f32>,
};

fn unit2(v: vec2<f32>, fallback: vec2<f32>) -> vec2<f32> {
    let l = length(v);
    if (l > 1e-6) {
        return v / l;
    }
    return fallback;
}

/// Focal length in pixels for the current field of view.
fn focal_px() -> f32 {
    return (scene.viewport.y * 0.5) / tan(scene.fov * 0.5);
}

/// Projects a world point to `(x px, y px, depth m)`, without the jitter.
fn project_to(p: vec3<f32>) -> vec3<f32> {
    let rel = p - scene.eye;
    let depth = dot(rel, scene.forward);
    let inv = focal_px() / max(depth, 1e-4);
    return vec3<f32>(
        scene.viewport.x * 0.5 + dot(rel, scene.right) * inv,
        scene.viewport.y * 0.5 - dot(rel, scene.up) * inv,
        depth,
    );
}

/// The per-bird screen data, with the model's wingbeat folded in.
fn bird_screen(b: Bird) -> BirdScreen {
    var s: BirdScreen;
    let f = forward(b.vel);
    let side = side_of(f);
    let up = body_up(f, b.roll);

    let rel = b.pos - scene.eye;
    let depth = dot(rel, scene.forward);
    let inv = focal_px() / max(depth, 1.0);
    s.depth = depth;
    s.centre = vec2<f32>(
        scene.viewport.x * 0.5 + dot(rel, scene.right) * inv,
        scene.viewport.y * 0.5 - dot(rel, scene.up) * inv,
    ) + scene.jitter;
    s.rho = min(WINGSPAN * inv, 16.0);

    // The reference's wing exposure, flap shimmer included.
    let dhat = normalize(scene.eye - b.pos);
    let flap = 1.0 + scene.flap_amplitude * b.flap_amp * sin(b.flap_phase);
    let wing = clamp(abs(dot(up, dhat)) * flap, 0.0, 1.0);
    let atmosphere = clamp(1.0 - depth / ATMOSPHERE, 0.25, 1.0);
    s.a = (0.15 + 0.85 * wing) * atmosphere;
    s.flap = FLAP_DIHEDRAL * b.flap_amp * sin(b.flap_phase);

    // The bird's own axes in screen space, for the impostor's silhouette.
    let fwd = project_to(b.pos + f * 0.2) - project_to(b.pos);
    let lateral = project_to(b.pos + side * 0.2) - project_to(b.pos);
    s.fwd2 = unit2(fwd.xy, vec2<f32>(0.0, -1.0));
    s.side2 = unit2(lateral.xy, vec2<f32>(1.0, 0.0));
    return s;
}

/// The fragment pair every bird pass writes: darkness into the HDR target and
/// raw optical depth into the optics target.
struct BirdOut {
    @location(0) hdr: vec4<f32>,
    @location(1) optics: vec4<f32>,
};

fn bird_out(raw: f32) -> BirdOut {
    let c = 1.0 - exp(-scene.exposure * max(raw, 0.0));
    var out: BirdOut;
    out.hdr = vec4<f32>(c, c, c, 1.0);
    out.optics = vec4<f32>(max(raw, 0.0), 0.0, 0.0, 0.0);
    return out;
}
