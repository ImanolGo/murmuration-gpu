// The LOD classifier (PLAN.md phase 6): one invocation per bird picks which of
// the three representations draws it, and appends it to that level's list.
//
// The levels are, from the largest projected wingspan down:
//
//   0  the ten-triangle mesh ([`bird_mesh.wgsl`]);
//   1  a flapping impostor quad ([`impostor.wgsl`]);
//   2  the sub-pixel coverage splat ([`coverage.wgsl`], the phase-5 pass).
//
// Two of the thresholds are *cross-fades*, not switches: a bird within `BAND`
// pixels of a threshold is written to both lists, with optical-depth weights
// that sum to one. A resolved bird's silhouette carries less total ink than a
// dot of the same wingspan (it is thinner), so the fade spreads that difference
// over the band rather than stepping at the threshold; `--lod-sweep` measures
// it. The thresholds are chosen where the two shapes differ by about a pixel.
//
// The lists are the indirect draws' instance buffers: level `k` occupies
// `lod_lists[k * birds .. (k + 1) * birds]` as `(bird index, weight bits)`
// pairs, and `lod_args` holds the three `DrawIndirect` argument blocks. The
// instance count lives in the argument block itself, so the atomicAdd that
// hands out a slot is also the draw count — no second pass to collect them.

@group(0) @binding(1) var<storage, read> birds: array<Bird>;
@group(0) @binding(2) var<storage, read_write> lod_lists: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> lod_args: array<atomic<u32>>;

/// Projected wingspan \[px\] at which the mesh takes over, with its cross-fade
/// half-width. Below this a bird is a few pixels and the mesh is not worth ten
/// triangles.
const RHO_MESH: f32 = 6.0;
const BAND: f32 = 2.0;

/// Below this a bird is smaller than a pixel and the analytic splat is both
/// cheaper and more accurate than a silhouette.
const RHO_SPLAT: f32 = 1.5;

/// Appends a bird to a level's instance list, taking the slot atomically so the
/// count in the argument block is the draw's instance count.
fn append(level: u32, stride: u32, index: u32, weight: f32) {
    let slot = atomicAdd(&lod_args[level * 4u + 1u], 1u);
    lod_lists[level * stride + slot] = vec2<u32>(index, bitcast<u32>(weight));
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&lod_lists) / 3u;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let b = birds[i];
    if (b.live == 0u) {
        // A slot the evening arc has not filled, or a bird that has landed.
        return;
    }
    let depth = dot(b.pos - scene.eye, scene.forward);
    if (depth <= 1.0) {
        // Behind (or on top of) the observer: no level draws it.
        return;
    }
    let rho = min(WINGSPAN * focal_px() / depth, 16.0);

    if (rho >= RHO_MESH + BAND) {
        append(0u, n, i, 1.0);
    } else if (rho >= RHO_MESH - BAND) {
        let w = smoothstep(RHO_MESH - BAND, RHO_MESH + BAND, rho);
        append(0u, n, i, w);
        append(1u, n, i, 1.0 - w);
    } else if (rho >= RHO_SPLAT) {
        append(1u, n, i, 1.0);
    } else {
        append(2u, n, i, 1.0);
    }
}
