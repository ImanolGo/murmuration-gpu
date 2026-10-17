//! CPU oracle for the procedural sky in shaders/atmosphere.wgsl. Rendering only:
//! soft cloud layers, low-sun scattering and a distant horizon, not flock forces.
use glam::{Vec2, Vec3};

fn smooth(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(1_597_334_677) ^ (y as u32).wrapping_mul(3_812_015_801);
    h = (h ^ (h >> 16)).wrapping_mul(2_246_822_519);
    h ^= h >> 13;
    (h & 65535) as f32 / 65535.0
}

fn noise(p: Vec2) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let u = f * f * (Vec2::splat(3.0) - f * 2.0);
    let (x, y) = (cell.x as i32, cell.y as i32);
    let a = hash(x, y) + (hash(x + 1, y) - hash(x, y)) * u.x;
    let b = hash(x, y + 1) + (hash(x + 1, y + 1) - hash(x, y + 1)) * u.x;
    a + (b - a) * u.y
}

fn clouds(p: Vec2) -> f32 {
    noise(p) * 0.56
        + noise(p * 2.03 + Vec2::splat(17.0)) * 0.28
        + noise(p * 4.11 - Vec2::splat(9.0)) * 0.16
}

/// Linear RGB along a world-space unit ray. The noise is anchored to direction,
/// not the screen, so camera motion does not drag the clouds with the flock.
pub fn colour(ray: Vec3, evening: f32, time: f32) -> Vec3 {
    let sky = Vec3::from_array(crate::sky::sky_linear(crate::sky::ramp_of(ray), evening));
    let light = 1.0 - smooth(0.55, 1.0, evening) * 0.97;
    let sun = Vec3::new(-0.65, 0.70, 0.10).normalize();
    let facing = ray.dot(sun).max(0.0);
    let horizon_glow = (-ray.z.max(0.0) * 7.0).exp();
    let uv = Vec2::new(ray.x, ray.y) / (ray.z.max(0.0) + 0.28) * Vec2::new(2.8, 4.2)
        + Vec2::new(time * 0.0035, -time * 0.0012);
    let field = clouds(uv);
    let cover = smooth(0.46, 0.78, field) * 0.42 * smooth(0.02, 0.28, ray.z);
    let edge = (0.55 + (clouds(uv + Vec2::new(0.045, 0.025)) - field) * 4.0).clamp(0.15, 1.0);
    let cloud = Vec3::new(0.24, 0.26, 0.34).lerp(Vec3::new(0.64, 0.58, 0.62), edge) * light
        + Vec3::new(0.22, 0.085, 0.030) * (facing.powi(4) * horizon_glow * light);
    let mut result = sky.lerp(cloud, cover)
        + Vec3::new(0.22, 0.11, 0.035) * ((-(1.0 - facing) * 10.0).exp() * light * (1.0 - cover));
    let bearing = ray.y.atan2(ray.x);
    let skyline = 0.002 + noise(Vec2::new(bearing * 2.5, 7.0)) * 0.006;
    if ray.z < skyline {
        let ground = Vec3::new(0.012, 0.018, 0.024) * light + sky * 0.06;
        result = ground.lerp(result, smooth(skyline - 0.003, skyline + 0.001, ray.z));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sky_is_finite_and_clouds_are_directional_and_slow() {
        for time in [0.0, 10.0, 1000.0] {
            for evening in [0.0, 0.4, 0.8, 1.0] {
                for ray in [Vec3::Y, Vec3::Z, Vec3::new(-0.4, 0.8, 0.3).normalize()] {
                    let c = colour(ray, evening, time);
                    assert!(c.is_finite() && c.min_element() >= 0.0);
                }
            }
        }
        let ray = Vec3::new(-0.4, 0.8, 0.3).normalize();
        assert!((colour(ray, 0.4, 0.0) - colour(ray, 0.4, 1.0)).length() < 0.01);
        assert!(colour(ray, 1.0, 0.0).length() < colour(ray, 0.4, 0.0).length());
    }
}
