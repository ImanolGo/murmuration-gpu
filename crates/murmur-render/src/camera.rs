//! A ground observer looking up at the flock.
//!
//! Mirrors `murmuration`'s `render/camera.rs` (the reference) function for
//! function: the observer stands still at a fixed spot on the ground, 150 m from
//! the roost centre, and turns to face the flock. The field of view is eased so
//! the flock's projected 5–95 % bounding box fills about 60 % of the binding
//! screen dimension — what a real observer does with attention, and a zoom lens
//! does for a camera. The flock is never kept small by a stand-off that grows
//! with it.
//!
//! The reference camera takes a `sim::World`; here it takes a [`FlockView`], so
//! the renderer can be driven from a GPU readback (the still capture) or from a
//! CPU flock without depending on either engine's type. The arithmetic and the
//! order of operations are the reference's, so the framing matches.

use glam::Vec3;

/// Eye height of the observer \[m\].
pub const EYE_HEIGHT: f32 = 1.7;

/// Horizontal distance of the observer from the roost centre \[m\].
pub const EYE_DISTANCE: f32 = 150.0;

/// Nominal vertical field of view \[rad\].
pub const FOV_V: f32 = 42.0 * std::f32::consts::PI / 180.0;

/// Scale the visible plane, not the angle. Multiplying a 45° angle by four
/// produces a degenerate 180° projection; scaling tan(fov/2) keeps it finite.
pub fn scaled_fov(fov: f32, scale: f32) -> f32 {
    2.0 * ((fov * 0.5).tan() * scale).atan()
}

/// Fraction of the binding screen dimension the flock should span.
const FILL: f32 = 0.6;

/// Bounds on the eased field of view \[rad\].
const FOV_MIN: f32 = 16.0 * std::f32::consts::PI / 180.0;
const FOV_MAX: f32 = 100.0 * std::f32::consts::PI / 180.0;

/// The flock the camera frames: positions and velocities, and the world clock.
pub struct FlockView<'a> {
    /// Bird positions \[m\].
    pub positions: &'a [Vec3],
    /// Bird velocities \[m/s\].
    pub velocities: &'a [Vec3],
    /// Simulated time \[s\], used to feed the flock's translation forward.
    pub time: f32,
}

impl FlockView<'_> {
    /// Number of birds in view.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the view holds no birds.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// A perspective camera on the ground.
pub struct Camera {
    /// Eye position \[m\].
    pub eye: Vec3,
    /// Unit view direction.
    pub forward: Vec3,
    /// Unit right direction.
    pub right: Vec3,
    /// Unit up direction.
    pub up: Vec3,
    /// Bearing around the roost \[rad\], changed by `c`, mouse drag, or `A`/`D`/Left/Right.
    pub bearing: f32,
    /// Distance of the observer from the roost centre \[m\].
    pub distance: f32,
    /// User zoom multiplier on the auto-framed field of view (<1 zooms in, >1 zooms out).
    pub zoom_factor: f32,
    /// Manual pitch offset \[rad\] on top of tracking elevation.
    pub pitch_offset: f32,
    /// Current vertical field of view \[rad\], eased toward the target.
    pub fov: f32,
    smoothed_target: Vec3,
    ready: bool,
    last_world_time: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    /// A camera south of the roost looking north.
    pub fn new() -> Self {
        Self {
            eye: Vec3::new(0.0, -EYE_DISTANCE, EYE_HEIGHT),
            forward: Vec3::Y,
            right: Vec3::X,
            up: Vec3::Z,
            bearing: -std::f32::consts::FRAC_PI_2,
            distance: EYE_DISTANCE,
            zoom_factor: 1.0,
            pitch_offset: 0.0,
            fov: FOV_V,
            smoothed_target: Vec3::ZERO,
            ready: false,
            last_world_time: 0.0,
        }
    }

    /// Resets interactive camera offsets (distance, zoom, pitch) back to defaults.
    pub fn reset_view(&mut self) {
        self.distance = EYE_DISTANCE;
        self.zoom_factor = 1.0;
        self.pitch_offset = 0.0;
    }

    /// Orbits the observer horizontally (`delta_bearing` in rad) and tilts elevation (`delta_pitch` in rad).
    pub fn orbit(&mut self, delta_bearing: f32, delta_pitch: f32) {
        self.bearing = (self.bearing + delta_bearing).rem_euclid(std::f32::consts::TAU);
        self.pitch_offset = (self.pitch_offset + delta_pitch).clamp(-0.45, 0.55);
    }

    /// Zooms the lens in (`factor < 1`) or out (`factor > 1`).
    pub fn zoom(&mut self, factor: f32) {
        self.zoom_factor = (self.zoom_factor * factor).clamp(0.25, 2.5);
    }

    /// Moves the observer closer to or further from the roost \[m\].
    pub fn dolly(&mut self, delta_m: f32) {
        self.distance = (self.distance + delta_m).clamp(40.0, 360.0);
    }

    /// Aim the initial view before asynchronous telemetry arrives, avoiding a
    /// snap from the horizon to the first bird. Normal tracking then takes over.
    pub fn look_at(&mut self, target: Vec3) {
        self.smoothed_target = target;
        self.ready = true;
        self.forward = (target - self.eye).normalize_or_zero();
        if self.forward.length_squared() < 1e-6 {
            self.forward = Vec3::Y;
        }
        self.right = self.forward.cross(Vec3::Z).normalize_or_zero();
        if self.right.length_squared() < 1e-6 {
            self.right = Vec3::X;
        }
        self.up = self.right.cross(self.forward);
    }

    /// Advances the camera by `dt` seconds to follow `view`. `aspect` is the
    /// pixel grid's width divided by its height.
    pub fn update(&mut self, view: &FlockView, dt: f32, aspect: f32) {
        if view.is_empty() {
            return;
        }
        let n = view.len() as f32;
        let centroid = view.positions.iter().copied().sum::<Vec3>() / n;
        let initializing = !self.ready;
        if !self.ready {
            self.smoothed_target = centroid;
            self.ready = true;
        }
        // Smooth the look target (time constant ~2 s). Follow the flock's
        // translation without a permanent two-second lag, or a narrow lens looks
        // at where the birds used to be and zooms out to cover that error.
        let velocity = view.velocities.iter().copied().sum::<Vec3>() / n;
        if !initializing {
            self.smoothed_target += velocity * (view.time - self.last_world_time).clamp(0.0, 0.25);
        }
        self.last_world_time = view.time;
        self.smoothed_target += (centroid - self.smoothed_target) * (1.0 - (-dt / 2.0).exp());

        // Observer spot on the ground, chosen by bearing and interactive distance.
        self.eye = Vec3::new(
            self.distance * self.bearing.cos(),
            self.distance * self.bearing.sin(),
            EYE_HEIGHT,
        );

        let mut forward = self.smoothed_target - self.eye;
        if forward.length_squared() < 1e-6 {
            forward = Vec3::Y;
        }
        forward = forward.normalize();
        if self.pitch_offset.abs() > 1e-5 {
            let horiz_right = forward.cross(Vec3::Z).normalize_or_zero();
            let up0 = horiz_right.cross(forward).normalize_or_zero();
            forward = (forward * self.pitch_offset.cos() + up0 * self.pitch_offset.sin())
                .normalize_or_zero();
        }
        self.forward = forward;
        let mut right = self.forward.cross(Vec3::Z);
        if right.length_squared() < 1e-6 {
            right = Vec3::X;
        }
        self.right = right.normalize();
        self.up = self.right.cross(self.forward);

        // Required vertical field of view so the flock's projected 5–95 %
        // bounding box fills `FILL` of the binding (smaller) screen dimension.
        // Work in normalised device coordinates (focal length 1), where the
        // screen half-height is `tan(fov/2)` and the half-width is `aspect`
        // times that.
        let mut xs = Vec::with_capacity(view.len());
        let mut ys = Vec::with_capacity(view.len());
        for p in view.positions {
            let rel = *p - self.eye;
            let depth = rel.dot(self.forward);
            if depth <= 1.0 {
                continue;
            }
            xs.push(rel.dot(self.right) / depth);
            ys.push(rel.dot(self.up) / depth);
        }
        let target_fov = if xs.is_empty() {
            self.fov
        } else {
            let (x0, x1) = percentiles(&mut xs);
            let (y0, y1) = percentiles(&mut ys);
            let x_extent = x0.abs().max(x1.abs());
            let y_extent = y0.abs().max(y1.abs());
            // Fill `FILL` of the binding (smaller) screen dimension.
            let t = (y_extent.max(x_extent / aspect.max(0.1))) / FILL;
            2.0 * (t * self.zoom_factor).atan()
        };
        let min_fov = (FOV_MIN * self.zoom_factor).clamp(6.0_f32.to_radians(), FOV_MAX);
        let target_fov = target_fov.clamp(min_fov, FOV_MAX);
        // Ease the zoom (time constant ~3 s).
        if initializing {
            self.fov = target_fov;
        } else {
            // Open the lens promptly on an approaching flock; tighten slowly.
            // A symmetric slow zoom clips the flock during overhead passes.
            let tau = if target_fov > self.fov { 0.6 } else { 3.0 };
            self.fov += (target_fov - self.fov) * (1.0 - (-dt / tau).exp());
        }
        self.fov = self.fov.clamp(min_fov, FOV_MAX);
    }

    /// Focal length in pixels for a pixel grid `pixel_h` dots tall.
    pub fn focal_px(&self, pixel_h: usize) -> f32 {
        (pixel_h as f32 / 2.0) / (self.fov / 2.0).tan()
    }

    /// Projects `p` to pixel coordinates `(x, y, depth)`, or `None` if it is
    /// behind the camera.
    pub fn project(&self, p: Vec3, pixel_w: usize, pixel_h: usize) -> Option<(f32, f32, f32)> {
        let rel = p - self.eye;
        let depth = rel.dot(self.forward);
        if depth <= 1.0 {
            return None;
        }
        let f = self.focal_px(pixel_h);
        let x = rel.dot(self.right);
        let y = rel.dot(self.up);
        Some((
            pixel_w as f32 / 2.0 + x / depth * f,
            pixel_h as f32 / 2.0 - y / depth * f,
            depth,
        ))
    }

    /// The point a cursor pixel is aimed at, on the plane through `plane_point`
    /// perpendicular to the view. Mirrors the reference's
    /// `camera.cursor_to_plane`, in window pixels instead of terminal cells.
    pub fn cursor_to_plane(
        &self,
        x: f32,
        y: f32,
        pixel_w: f32,
        pixel_h: f32,
        plane_point: Vec3,
    ) -> Vec3 {
        let f = self.focal_px(pixel_h as usize);
        let ndc_x = (x - pixel_w / 2.0) / f;
        let ndc_y = (y - pixel_h / 2.0) / f;
        let dir = (self.forward + self.right * ndc_x - self.up * ndc_y).normalize();
        let normal = self.forward;
        let denom = dir.dot(normal);
        if denom.abs() < 1e-6 {
            return plane_point;
        }
        let t = (plane_point - self.eye).dot(normal) / denom;
        self.eye + dir * t
    }
}

/// Exact 5th and 95th percentiles, selected in linear time. No full sort is
/// needed just to frame the flock; captures can contain millions of positions.
fn percentiles(v: &mut [f32]) -> (f32, f32) {
    let lo = ((v.len() as f32 * 0.05) as usize).min(v.len() - 1);
    let hi = ((v.len() as f32 * 0.95) as usize).min(v.len() - 1);
    let (_, high, _) = v.select_nth_unstable_by(hi, f32::total_cmp);
    let high = *high;
    let (_, low, _) = v[..=hi].select_nth_unstable_by(lo, f32::total_cmp);
    (*low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_capture_scales_focal_length_without_a_180_degree_projection() {
        for degrees in [12.0f32, 45.0, 70.0] {
            let fov = degrees.to_radians();
            let wide = scaled_fov(fov, 4.0);
            assert!(wide > fov && wide < std::f32::consts::PI);
            assert!(((wide * 0.5).tan() / (fov * 0.5).tan() - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn selected_percentiles_match_a_full_sort() {
        for n in [1, 2, 19, 20, 4096] {
            let mut values: Vec<_> = (0..n).map(|i| ((i * 7919) % 1009) as f32 - 500.0).collect();
            let mut sorted = values.clone();
            sorted.sort_by(f32::total_cmp);
            let expected = (
                sorted[((n as f32 * 0.05) as usize).min(n - 1)],
                sorted[((n as f32 * 0.95) as usize).min(n - 1)],
            );
            assert_eq!(percentiles(&mut values), expected);
        }
    }

    /// A straight line of birds flying east, as in the reference's camera tests.
    fn line_of_birds(n: usize) -> (Vec<Vec3>, Vec<Vec3>) {
        let positions = (0..n)
            .map(|i| Vec3::new(i as f32 - n as f32 / 2.0, 0.0, 100.0 + (i % 5) as f32))
            .collect();
        let velocities = vec![Vec3::X * 10.0; n];
        (positions, velocities)
    }

    #[test]
    fn wide_flock_is_fitted_in_both_dimensions() {
        for (w, h) in [(240usize, 144usize), (160, 96), (80, 160)] {
            let (positions, velocities) = line_of_birds(200);
            let view = FlockView {
                positions: &positions,
                velocities: &velocities,
                time: 0.0,
            };
            let mut camera = Camera::new();
            for _ in 0..600 {
                camera.update(&view, 1.0 / 30.0, w as f32 / h as f32);
            }
            let visible = positions
                .iter()
                .filter(|&&p| {
                    camera
                        .project(p, w, h)
                        .is_some_and(|(x, y, _)| x > 0.0 && x < w as f32 && y > 0.0 && y < h as f32)
                })
                .count();
            assert!(
                visible >= 190,
                "clipped wide flock: {visible}/200 at {w}x{h}"
            );
        }
    }

    #[test]
    fn tracking_does_not_lag_behind_a_translating_flock() {
        let (mut positions, velocities) = line_of_birds(200);
        let mut camera = Camera::new();
        let mut time = 0.0;
        {
            let view = FlockView {
                positions: &positions,
                velocities: &velocities,
                time,
            };
            camera.update(&view, 1.0 / 30.0, 240.0 / 144.0);
        }
        for _ in 0..150 {
            for p in &mut positions {
                *p += Vec3::X / 3.0;
            }
            time += 1.0 / 30.0;
            let view = FlockView {
                positions: &positions,
                velocities: &velocities,
                time,
            };
            camera.update(&view, 1.0 / 30.0, 240.0 / 144.0);
        }
        let centroid = positions.iter().copied().sum::<Vec3>() / positions.len() as f32;
        let (x, y, _) = camera.project(centroid, 240, 144).unwrap();
        assert!(
            (x - 120.0).abs() < 2.0 && (y - 72.0).abs() < 2.0,
            "camera watches where the flock used to be: {x}, {y}"
        );
    }
}
