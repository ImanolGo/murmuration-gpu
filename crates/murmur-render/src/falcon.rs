//! The falcon's flight attitude for drawing: heading and bank.
//!
//! Rendering only. The simulation's falcon is a point with a velocity; the
//! silhouette in `shaders/falcon.wgsl` needs a body frame. Heading is the
//! (smoothed) velocity, and the bank is the coordinated-turn angle the starlings
//! use too (`flight::step_bird`: `atan(a_lat / g)`), taken from how fast the
//! smoothed velocity turns. Telemetry arrives every few frames, so both are
//! eased with short time constants instead of differenced raw.

use glam::Vec3;

/// Gravity \[m/s²\], as in the model.
const G: f32 = 9.81;
/// Steepest drawn bank \[rad\]; a peregrine rolls hard, but not past 70°.
const MAX_BANK: f32 = 1.2;
/// Smoothing of the heading \[s\].
const TAU_VEL: f32 = 0.12;
/// Smoothing of the bank \[s\].
const TAU_BANK: f32 = 0.25;

/// Eases the falcon's drawn heading and bank from frame to frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FalconMotion {
    vel: Option<Vec3>,
    bank: f32,
}

impl FalconMotion {
    /// Forgets the previous attitude (a new falcon, or a camera cut).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Advances by `dt` seconds with the falcon's current velocity and returns
    /// the velocity and bank angle to draw, positive with the right wing down.
    pub fn update(&mut self, velocity: Vec3, dt: f32) -> (Vec3, f32) {
        let dt = dt.clamp(0.0, 0.25);
        let Some(prev) = self.vel else {
            self.vel = Some(velocity);
            return (velocity, 0.0);
        };
        let k = 1.0 - (-dt / TAU_VEL).exp();
        let vel = prev + (velocity - prev) * k;
        self.vel = Some(vel);
        if dt > 1e-5 && vel.length_squared() > 1e-4 {
            let f = vel.normalize();
            let side = f.cross(Vec3::Z).normalize_or(Vec3::X);
            let accel = (vel - prev) / dt;
            let target = (accel.dot(side)).atan2(G).clamp(-MAX_BANK, MAX_BANK);
            self.bank += (target - self.bank) * (1.0 - (-dt / TAU_BANK).exp());
        }
        (vel, self.bank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_flight_does_not_bank() {
        let mut m = FalconMotion::default();
        for _ in 0..120 {
            let (v, bank) = m.update(Vec3::new(15.0, 0.0, 0.0), 1.0 / 60.0);
            assert!((v - Vec3::new(15.0, 0.0, 0.0)).length() < 1e-3);
            assert!(bank.abs() < 1e-4);
        }
    }

    #[test]
    fn a_right_turn_banks_right_wing_down() {
        // Heading +x, turning toward `side = f × z = -y`: a right turn.
        let mut m = FalconMotion::default();
        let mut bank = 0.0;
        for i in 0..240 {
            let a = -(i as f32) * (1.0 / 60.0) * 0.8;
            let v = Vec3::new(a.cos(), a.sin(), 0.0) * 15.0;
            bank = m.update(v, 1.0 / 60.0).1;
        }
        // ω = 0.8 rad/s at 15 m/s is 12 m/s² lateral: atan(12/9.81) ≈ 0.88.
        assert!(bank > 0.7 && bank < 0.95, "bank {bank}");
    }
}
