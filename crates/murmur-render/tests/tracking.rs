//! Camera telemetry is bounded and observational, not part of the simulation.
use glam::Vec3;
use murmur_gpu::{
    World,
    device::{self, DeviceError},
    sim::Sim,
};
use murmur_render::tracking::FlockTracker;

#[test]
fn asynchronous_exposure_matches_the_synchronous_static_frame() {
    let gpu = match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => gpu,
        Err(DeviceError::NoAdapter(error)) => {
            eprintln!("SKIP: no GPU for asynchronous exposure ({error})");
            return;
        }
        Err(error) => panic!("{error}"),
    };
    use murmur_render::{
        camera::{Camera, FlockView},
        renderer::{Renderer, Scene},
    };
    let world = World::new(500, 5);
    let sim = Sim::from_snapshot(&gpu, &world.snapshot());
    let mut camera = Camera::new();
    camera.update(
        &FlockView {
            positions: &world.position,
            velocities: &world.velocity,
            time: 0.0,
        },
        1.0 / 30.0,
        128.0 / 72.0,
    );
    let mut sync = Renderer::new(&gpu, 128, 72, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();
    let mut asynchronous =
        Renderer::new(&gpu, 128, 72, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();
    asynchronous.set_async_exposure(true);
    let sync_target = sync.create_target(&gpu, 128, 72);
    let async_target = asynchronous.create_target(&gpu, 128, 72);
    for renderer in [&mut sync, &mut asynchronous] {
        renderer.set_birds(&gpu, sim.bird_buffer(), sim.len() as u32);
        renderer.set_jitter(Some([0.0; 2]));
        renderer.set_history_weight(Some(1.0));
        renderer.set_scene(
            &gpu,
            Scene::from_camera(&camera, 128, 72, 0.5, renderer.exposure(), 0.15, 0.0),
        );
    }
    for _ in 0..6 {
        for renderer in [&mut sync, &mut asynchronous] {
            let mut scene = *renderer.scene();
            scene.exposure = renderer.exposure();
            renderer.set_scene(&gpu, scene);
        }
        sync.render(&gpu, &sync_target.view, sim.len() as u32, 1.0 / 30.0);
        asynchronous.render(&gpu, &async_target.view, sim.len() as u32, 1.0 / 30.0);
        // Wait only in this test to make callback completion deterministic.
        gpu.wait_bounded(device::WAIT_BUDGET).unwrap();
    }
    assert!((sync.exposure() - asynchronous.exposure()).abs() < 1e-4);
    let a = sync.read_target(&gpu, &sync_target, 128, 72);
    let b = asynchronous.read_target(&gpu, &async_target, 128, 72);
    let mean_diff = a
        .iter()
        .zip(&b)
        .map(|(a, b)| a.abs_diff(*b) as f32)
        .sum::<f32>()
        / a.len() as f32;
    assert!(mean_diff < 0.2, "async/static image difference {mean_diff}");
}

#[test]
fn tracking_samples_live_birds_without_changing_them() {
    let gpu = match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => gpu,
        Err(DeviceError::NoAdapter(error)) => {
            eprintln!("SKIP: no GPU for camera tracking ({error})");
            return;
        }
        Err(error) => panic!("{error}"),
    };
    for n in [17, 10000] {
        let world = World::new_density_scaled(n, 5);
        let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
        let mut birds = sim.read_birds(&gpu).unwrap();
        for (i, b) in birds.iter_mut().enumerate() {
            b.live = u32::from(i % 3 != 0);
        }
        sim.write_birds(&gpu, &birds);
        let mut tracker = FlockTracker::new(&gpu);
        assert!(!tracker.refresh(&gpu).unwrap());
        tracker.request(&gpu, sim.bird_buffer(), 0.0);
        // A second request while busy must not map or write the staging twice.
        tracker.request(&gpu, sim.bird_buffer(), 1.0);
        gpu.wait_bounded(device::WAIT_BUDGET).unwrap();
        assert!(tracker.refresh(&gpu).unwrap());
        let count = n.min(4096);
        let expected: Vec<_> = (0..count)
            .map(|i| &birds[i * n / count])
            .filter(|b| b.live != 0)
            .collect();
        let positions = tracker.predicted_positions(0.0);
        assert_eq!(positions.len(), expected.len());
        for (p, b) in positions.iter().zip(&expected) {
            assert_eq!(*p, Vec3::from_array(b.pos));
        }
        let predicted = tracker.predicted_positions(0.1);
        for (p, b) in predicted.iter().zip(&expected) {
            assert!(
                (*p - (Vec3::from_array(b.pos) + Vec3::from_array(b.vel) * 0.1)).length() < 1e-4
            );
        }
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&birds),
            bytemuck::cast_slice::<_, u8>(&sim.read_birds(&gpu).unwrap())
        );
    }
}
