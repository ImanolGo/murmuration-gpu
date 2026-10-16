// The exposure histogram: bins the raw optical depth of the optics target so
// the CPU can find the 80th percentile and adapt the exposure, as the reference
// does (a GPU percentile is a phase 9 concern). Zero optical-depth pixels are
// skipped, matching the reference's `samples = depth > 0`.

@group(0) @binding(0) var optics: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;

const BINS: u32 = 256u;
const TAU_MAX: f32 = 8.0;

@compute @workgroup_size(8, 8)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(optics);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let tau = textureLoad(optics, vec2<i32>(gid.xy), 0).x;
    if (tau <= 0.0) {
        return;
    }
    let bin = u32(clamp(tau / TAU_MAX, 0.0, 0.999999) * f32(BINS));
    atomicAdd(&bins[bin], 1u);
}
