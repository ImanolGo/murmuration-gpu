//! Compare the procedural background with its CPU reference at actual pixels.
use glam::Vec3;
use murmur_gpu::{
    device::{self, DeviceError},
    params::Params,
    sim::Sim,
};
use murmur_render::{
    atmosphere,
    camera::Camera,
    renderer::{Renderer, Scene},
};

#[test]
fn procedural_sky_pixels_match_the_cpu_reference() {
    let gpu = match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => gpu,
        Err(DeviceError::NoAdapter(error)) => {
            eprintln!("SKIP: no GPU for atmosphere test ({error})");
            return;
        }
        Err(error) => panic!("{error}"),
    };
    let (width, height) = (96, 64);
    let mut camera = Camera::new();
    camera.forward = Vec3::new(0.0, 1.0, 0.2).normalize();
    camera.right = camera.forward.cross(Vec3::Z).normalize();
    camera.up = camera.right.cross(camera.forward);
    let sim = Sim::new(&gpu, Params::new(0, 1));
    let mut renderer =
        Renderer::new(&gpu, width, height, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();
    renderer.set_birds(&gpu, sim.bird_buffer(), 0);
    renderer.set_jitter(Some([0.0; 2]));
    renderer.set_history_weight(Some(1.0));
    let target = renderer.create_target(&gpu, width, height);
    for (evening, time) in [(0.0, 0.0), (0.4, 5.0), (0.8, 250.0), (1.0, 0.0)] {
        renderer.set_scene(
            &gpu,
            Scene::from_camera(
                &camera,
                width,
                height,
                evening,
                renderer.exposure(),
                0.0,
                time,
            ),
        );
        renderer.render(&gpu, &target.view, 0, 1.0 / 60.0);
        let pixels = renderer.read_target(&gpu, &target, width, height);
        for (x, y) in [(3, 3), (45, 20), (80, 60)] {
            let nx = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ny = 1.0 - (y as f32 + 0.5) / height as f32 * 2.0;
            let t = (camera.fov * 0.5).tan();
            let ray = (camera.forward
                + camera.right * nx * t * width as f32 / height as f32
                + camera.up * ny * t)
                .normalize();
            let expected = atmosphere::colour(ray, evening, time).to_array().map(|v| {
                let v = v.clamp(0.0, 1.0);
                let srgb = if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                (srgb * 255.0).round() as u8
            });
            let index = ((y * width + x) * 4) as usize;
            for channel in 0..3 {
                assert!(
                    pixels[index + channel].abs_diff(expected[channel]) <= 3,
                    "pixel ({x},{y}), evening {evening}: GPU {:?}, CPU {expected:?}",
                    &pixels[index..index + 3]
                );
            }
        }
    }
}
