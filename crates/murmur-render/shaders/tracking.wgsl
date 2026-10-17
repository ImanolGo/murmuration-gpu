// Camera sampling for camera.rs::Camera::update. This only observes the flock;
// it does not change the reference model. At most 4096 evenly spaced records
// cross to the host, asynchronously, instead of the complete bird buffer.
struct TrackingPoint {
    pos: vec3<f32>, live: u32,
    vel: vec3<f32>, pad: u32,
}
@group(0) @binding(0) var<storage, read> birds: array<Bird>;
@group(0) @binding(1) var<storage, read_write> points: array<TrackingPoint>;

@compute @workgroup_size(256)
fn sample_flock(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&birds);
    let count = min(n, arrayLength(&points));
    if gid.x >= arrayLength(&points) {
        return;
    }
    var point: TrackingPoint;
    if gid.x < count {
        // Avoid overflowing u32 when n is in the millions.
        let index = gid.x * (n / count) + gid.x * (n % count) / count;
        let b = birds[index];
        point.pos = b.pos;
        point.vel = b.vel;
        point.live = b.live;
    }
    points[gid.x] = point;
}
