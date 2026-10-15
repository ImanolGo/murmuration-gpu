//! Extra flock statistics PLAN.md phase 1 asks for, on top of the reference
//! metrics in [`crate::analysis`]: nearest-neighbour distance over all birds,
//! polarisation, flock extent and aspect ratios, border-vs-centre density and a
//! speed distribution.
//!
//! These do not change the model; they only measure it. The reference `Stats`
//! stays byte-for-byte comparable with the terminal `--analyze` output.

use glam::Vec3;

use crate::analysis::CENTRALITY_INTERIOR;
use crate::flight::forward;
use crate::neighbours::Grid;
use crate::params::GRID_CELL;
use crate::world::World;

/// Radius within which the nearest neighbour is searched for \[m\], matching
/// the reference analysis.
const NND_RADIUS: f32 = 15.0;

/// A snapshot of the extended statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlockStats {
    /// Number of birds.
    pub birds: usize,
    /// Heading coherence: length of the mean unit forward vector, 0–1.
    pub polarization: f32,
    /// Mean topological neighbour count `|N_i|` (from each bird's decision).
    pub mean_neighbours: f32,
    /// Mean nearest-neighbour distance over *all* birds \[m\].
    pub mean_nnd_all: f32,
    /// Bounding-box size `max-min` on each axis \[m\].
    pub extent: Vec3,
    /// Aspect ratio `extent.y / extent.x`.
    pub aspect_yx: f32,
    /// Aspect ratio `extent.z / extent.x`.
    pub aspect_zx: f32,
    /// Radius of gyration (RMS distance from the centroid) \[m\].
    pub radius_of_gyration: f32,
    /// Fraction of birds the centrality classifies as interior.
    pub interior_fraction: f32,
    /// Mean bird density (neighbours per m³ inside the adaptive radius) for
    /// interior birds.
    pub centre_density: f32,
    /// The same for border birds.
    pub border_density: f32,
    /// Mean speed [m/s].
    pub speed_mean: f32,
    /// Population standard deviation of speed [m/s].
    pub speed_std: f32,
    /// 10th percentile of speed [m/s].
    pub speed_p10: f32,
    /// Median speed [m/s].
    pub speed_p50: f32,
    /// 90th percentile of speed [m/s].
    pub speed_p90: f32,
}

impl FlockStats {
    /// A one-line human-readable summary.
    pub fn line(&self) -> String {
        format!(
            "birds={:5} pol={:4.2} |N|={:4.2} NNDall={:4.2}m extent={:5.1}x{:5.1}x{:5.1}m eig_yx={:4.2} eig_zx={:4.2} rgyr={:5.1}m interior={:5.1}% rho_c={:5.2} rho_b={:5.2} v={:4.1}+-{:4.2}m/s p10/p50/p90={:4.1}/{:.1}/{:.1}",
            self.birds,
            self.polarization,
            self.mean_neighbours,
            self.mean_nnd_all,
            self.extent.x,
            self.extent.y,
            self.extent.z,
            self.aspect_yx,
            self.aspect_zx,
            self.radius_of_gyration,
            self.interior_fraction * 100.0,
            self.centre_density,
            self.border_density,
            self.speed_mean,
            self.speed_std,
            self.speed_p10,
            self.speed_p50,
            self.speed_p90,
        )
    }
}

/// Computes the extended statistics for `world`.
pub fn compute(world: &World) -> FlockStats {
    let pos = &world.position;
    let n = pos.len();
    if n == 0 {
        return FlockStats::default();
    }
    let mut grid = Grid::new(GRID_CELL);
    grid.rebuild(pos);

    let polarization = polarization(&world.velocity);

    // Nearest-neighbour distance over all birds.
    let mut nnd_sum = 0.0;
    let mut nnd_count = 0u32;
    for i in 0..n {
        let mut best = f32::MAX;
        grid.for_each_within(pos[i], NND_RADIUS, pos, |j, d| {
            if j != i && d < best {
                best = d;
            }
        });
        if best < f32::MAX {
            nnd_sum += best;
            nnd_count += 1;
        }
    }
    let mean_nnd_all = if nnd_count > 0 {
        nnd_sum / nnd_count as f32
    } else {
        f32::NAN
    };

    // Extent and aspect ratios from the bounding box.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let centroid = pos.iter().copied().sum::<Vec3>() / n as f32;
    let mut gyration = 0.0;
    for p in pos {
        lo = lo.min(*p);
        hi = hi.max(*p);
        gyration += (*p - centroid).length_squared();
    }
    let extent = hi - lo;
    let aspect_yx = if extent.x > 1e-6 {
        extent.y / extent.x
    } else {
        f32::NAN
    };
    let aspect_zx = if extent.x > 1e-6 {
        extent.z / extent.x
    } else {
        f32::NAN
    };
    let radius_of_gyration = (gyration / n as f32).sqrt();

    // Border vs centre density: neighbours per m³ inside each bird's adaptive
    // radius. Border birds carry a larger radius, so the volume normalisation
    // keeps the comparison fair.
    let mut centre_sum = 0.0;
    let mut centre_n = 0u32;
    let mut border_sum = 0.0;
    let mut border_n = 0u32;
    for i in 0..n {
        let r = world.radii[i].max(1e-3);
        let volume = (4.0 / 3.0) * std::f32::consts::PI * r * r * r;
        let density = world.neighbour_count[i] as f32 / volume;
        if world.centrality[i] < CENTRALITY_INTERIOR {
            centre_sum += density;
            centre_n += 1;
        } else {
            border_sum += density;
            border_n += 1;
        }
    }
    let centre_density = if centre_n > 0 {
        centre_sum / centre_n as f32
    } else {
        f32::NAN
    };
    let border_density = if border_n > 0 {
        border_sum / border_n as f32
    } else {
        f32::NAN
    };
    let interior_fraction = centre_n as f32 / n as f32;

    // Speed distribution.
    let mut speeds: Vec<f32> = world.velocity.iter().map(|v| v.length()).collect();
    let speed_mean = speeds.iter().sum::<f32>() / n as f32;
    let speed_std = (speeds.iter().map(|s| (s - speed_mean).powi(2)).sum::<f32>() / n as f32)
        .max(0.0)
        .sqrt();
    speeds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let speed_p10 = quantile(&speeds, 0.10);
    let speed_p50 = quantile(&speeds, 0.50);
    let speed_p90 = quantile(&speeds, 0.90);

    let mean_neighbours = world.neighbour_count.iter().map(|&c| c as f32).sum::<f32>() / n as f32;

    FlockStats {
        birds: n,
        polarization,
        mean_neighbours,
        mean_nnd_all,
        extent,
        aspect_yx,
        aspect_zx,
        radius_of_gyration,
        interior_fraction,
        centre_density,
        border_density,
        speed_mean,
        speed_std,
        speed_p10,
        speed_p50,
        speed_p90,
    }
}

/// Heading coherence of a set of velocities: the length of the mean unit
/// forward vector, from 0 (no order) to 1 (perfectly aligned).
pub fn polarization(velocities: &[Vec3]) -> f32 {
    if velocities.is_empty() {
        return 0.0;
    }
    velocities
        .iter()
        .map(|v| forward(*v))
        .sum::<Vec3>()
        .length()
        / velocities.len() as f32
}

/// Linear-interpolation quantile of a sorted slice, `q` in 0..=1.
fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f32;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_interpolate() {
        let v = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(quantile(&v, 0.0), 0.0);
        assert_eq!(quantile(&v, 1.0), 3.0);
        assert!((quantile(&v, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn polarization_is_one_for_a_common_heading() {
        let v = vec![Vec3::X, Vec3::X * 2.0, Vec3::new(3.0, 0.0, 0.0)];
        assert!((polarization(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn polarization_is_low_for_opposed_headings() {
        let v = vec![Vec3::X, -Vec3::X];
        assert!(polarization(&v) < 1e-6);
    }
}
