// The foreground reed bed (PLAN.md phase 8), mirrored from the terminal
// reference's `render/sky.rs::draw_reeds_with_opacity`: a sparse band of stems
// along the bottom of the frame whose tips lean and sway with the wind. Drawn
// last, so the birds cannot paint over the foreground.
//
// The visibility comes from the app (`sky::reed_visibility`), because it depends
// on where the camera is looking; a stem then writes a dark silhouette into the
// HDR target through the same multiplicative blend the birds use.

struct ReedOut {
    @location(0) hdr: vec4<f32>,
    @location(1) optics: vec4<f32>,
};

@vertex
fn vs_reeds(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vid], 0.0, 1.0);
}

@fragment
fn fs_reeds(@builtin(position) coord: vec4<f32>) -> ReedOut {
    var out: ReedOut;
    out.hdr = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.optics = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (scene.reed < 0.002) {
        return out;
    }

    let height = scene.viewport.y;
    // A shallow band: the tallest clump is a few dozen rows (the terminal's
    // three rows of four dots).
    let band = max(6.0, height * 0.09);
    let k = height - coord.y; // 0 at the bottom row, rising
    if (k > 26.0 || k < 0.0) {
        return out;
    }

    // A base column every two pixels; about a third of them carry a stem.
    let base = floor(coord.x / 2.0);
    var hash = u32(base) * 2654435761u;
    hash ^= hash >> 16u;
    hash = hash * 2246822519u;
    hash ^= hash >> 13u;
    if (hash % 100u >= 32u) {
        return out;
    }
    let length = 3.0 + f32((hash >> 27u) % 20u);
    if (k > length) {
        return out;
    }
    let phase = f32(hash & 255u) * 0.025;
    let lean = f32((hash >> 16u) & 15u) / 7.5 - 1.0;
    // Wind bends the tip; the stem's length never changes.
    let sway = lean * 1.8 + sin(scene.time * 0.65 + phase) * 0.85;
    let f = k / length;
    let stem = base + sway * f * f;
    let dx = abs(coord.x / 2.0 - stem);
    if (dx > 0.45) {
        return out;
    }
    // A narrow seed head at the tip reads as a reed, not a fence post.
    let head = k > length - 2.0;
    let coverage = scene.reed * select(0.85, 1.0, head);
    out.hdr = vec4<f32>(coverage, coverage, coverage, 1.0);
    // The reed bed is not part of the flock's optical depth.
    out.optics = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}
