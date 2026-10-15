//! Flock statistics for the oracle run, matching the reference `--analyze`.
//!
//! Mirrors `murmuration::sim::analysis` so a headless run of this crate can be
//! diffed against the recorded terminal output. Extra statistics that PLAN.md
//! phase 1 asks for (polarisation, extent, border-vs-centre density, speed
//! distribution) live in [`crate::stats`] and are added on top.

use glam::Vec3;

use crate::flight::forward;
use crate::neighbours::Grid;
use crate::params::{GRID_CELL, MAX_BANK};
use crate::world::World;

/// Centrality below which a bird counts as interior, from Table 1 (`C_c`).
pub const CENTRALITY_INTERIOR: f32 = 0.35;

/// A snapshot of flock statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Number of birds.
    pub birds: usize,
    /// Mean topological neighbour count `|N_i|`.
    pub mean_neighbours: f32,
    /// Fraction of birds in the largest single-link cluster at 5 m.
    pub largest_cluster_fraction: f32,
    /// Mean nearest-neighbour distance over interior birds \[m\].
    pub mean_nnd: f32,
    /// Middle-to-shortest inertia ratio `I2/I1`.
    pub i2_i1: f32,
    /// Longest-to-shortest inertia ratio `I3/I1`.
    pub i3_i1: f32,
    /// `|I1·ẑ|`: how vertical the flock's thin axis is.
    pub flatness: f32,
    /// Fraction of interior birds whose nearest neighbour is ahead or behind
    /// (`|cos θ| > 0.9`); isotropic would be 0.1.
    pub nn_aligned_fraction: f32,
    /// Largest horizontal distance from the roost centre \[m\].
    pub max_radius_h: f32,
    /// Largest 3-D distance from the roost centre \[m\].
    pub max_radius_3d: f32,
    /// Mean speed [m/s].
    pub mean_speed: f32,
    /// Mean adaptive interaction radius \[m\].
    pub mean_radius: f32,
    /// Mean distance of birds from the flock centroid \[m\].
    pub spread: f32,
    /// Mean `|z - z0|` \[m\].
    pub mean_dz: f32,
    /// Fraction of birds alarmed by the falcon.
    pub alarmed_fraction: f32,
    /// Mean absolute bank angle \[rad\].
    pub mean_roll: f32,
    /// Fraction of birds pinned at the ±`MAX_BANK` clamp.
    pub clamp_fraction: f32,
    /// Mean signed tangential velocity about the roost centre [m/s]; positive
    /// is counter-clockwise seen from above. A persistent sign means the flock
    /// is milling around the boundary instead of crossing the roost.
    pub tangential: f32,
    /// Standard deviation of bird height \[m\].
    pub height_std: f32,
    /// Horizontal distance of the flock centroid from the roost centre \[m\].
    pub centroid_radius: f32,
}

impl Stats {
    /// A one-line human-readable summary, identical to the reference format.
    pub fn line(&self) -> String {
        format!(
            "birds={:5} |N|={:4.2} R={:5.2}m cluster={:5.1}% NND={:4.2}m spread={:5.1}m Rh={:5.1}m dz={:4.1}m I2/I1={:4.2} I3/I1={:4.2} flat={:4.2} nn_ab={:4.2} v={:4.1}m/s alarm={:4.1}% roll={:4.2} clamp={:4.1}% tang={:5.2}m/s hstd={:4.1}m rc={:5.1}m",
            self.birds,
            self.mean_neighbours,
            self.mean_radius,
            self.largest_cluster_fraction * 100.0,
            self.mean_nnd,
            self.spread,
            self.max_radius_h,
            self.mean_dz,
            self.i2_i1,
            self.i3_i1,
            self.flatness,
            self.nn_aligned_fraction,
            self.mean_speed,
            self.alarmed_fraction * 100.0,
            self.mean_roll,
            self.clamp_fraction * 100.0,
            self.tangential,
            self.height_std,
            self.centroid_radius,
        )
    }
}

/// Computes all statistics for `world`.
pub fn compute(world: &World) -> Stats {
    let pos = &world.position;
    let n = pos.len();
    if n == 0 {
        return Stats::default();
    }
    let mut grid = Grid::new(GRID_CELL);
    grid.rebuild(pos);

    let mean_neighbours = mean(world.neighbour_count.iter().map(|&c| c as f32));
    let mean_speed = mean(world.velocity.iter().map(|v| v.length()));
    let mean_radius = mean(world.radii.iter().copied());
    let centroid = pos.iter().copied().sum::<Vec3>() / n as f32;
    let spread = mean(pos.iter().map(|p| (*p - centroid).length()));
    let mean_dz = mean(pos.iter().map(|p| (p.z - world.z0).abs()));
    let alarmed_fraction = world.escape.iter().filter(|e| e.alarmed).count() as f32 / n as f32;
    let mean_roll = mean(world.roll.iter().map(|r| r.abs()));
    let clamp_fraction = world
        .roll
        .iter()
        .filter(|r| r.abs() >= MAX_BANK - 1e-3)
        .count() as f32
        / n as f32;
    let mean_height = mean(pos.iter().map(|p| p.z));
    let height_std = mean(pos.iter().map(|p| (p.z - mean_height).powi(2)))
        .max(0.0)
        .sqrt();
    let centroid_radius = (centroid.x * centroid.x + centroid.y * centroid.y).sqrt();
    // Signed tangential velocity about the roost centre: positive is
    // counter-clockwise seen from above.
    let mut tangential_sum = 0.0;
    let mut tangential_count = 0u32;
    for (p, v) in pos.iter().zip(&world.velocity) {
        let r = Vec3::new(p.x, p.y, 0.0);
        let d = r.length();
        if d > 1e-3 {
            tangential_sum += v.dot(Vec3::Z.cross(r / d));
            tangential_count += 1;
        }
    }
    let tangential = if tangential_count > 0 {
        tangential_sum / tangential_count as f32
    } else {
        0.0
    };
    let max_radius_h = pos
        .iter()
        .map(|p| (p.x * p.x + p.y * p.y).sqrt())
        .fold(0.0, f32::max);
    let max_radius_3d = pos.iter().map(|p| p.length()).fold(0.0, f32::max);

    let mut nnd_sum = 0.0;
    let mut nnd_count = 0u32;
    let mut aligned = 0u32;
    let mut nn_total = 0u32;
    for i in 0..n {
        let mut best = f32::MAX;
        let mut best_j = usize::MAX;
        grid.for_each_within(pos[i], 15.0, pos, |j, d| {
            if j != i && d < best {
                best = d;
                best_j = j;
            }
        });
        if best_j == usize::MAX || world.centrality[i] >= CENTRALITY_INTERIOR {
            continue;
        }
        nnd_sum += best;
        nnd_count += 1;
        let f = forward(world.velocity[i]);
        let dhat = (pos[best_j] - pos[i]) / best.max(1e-6);
        if f.dot(dhat).abs() > 0.9 {
            aligned += 1;
        }
        nn_total += 1;
    }

    let (i2_i1, i3_i1, flatness) = inertia(pos);

    Stats {
        birds: n,
        mean_neighbours,
        largest_cluster_fraction: largest_cluster(pos, &grid, 5.0),
        mean_nnd: if nnd_count > 0 {
            nnd_sum / nnd_count as f32
        } else {
            f32::NAN
        },
        i2_i1,
        i3_i1,
        flatness,
        nn_aligned_fraction: if nn_total > 0 {
            aligned as f32 / nn_total as f32
        } else {
            f32::NAN
        },
        max_radius_h,
        max_radius_3d,
        mean_speed,
        mean_radius,
        spread,
        mean_dz,
        alarmed_fraction,
        mean_roll,
        clamp_fraction,
        tangential,
        height_std,
        centroid_radius,
    }
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut sum = 0.0;
    let mut count = 0u32;
    for v in values {
        sum += v;
        count += 1;
    }
    if count > 0 {
        sum / count as f32
    } else {
        f32::NAN
    }
}

/// Fraction of birds in the largest connected component when two birds are
/// linked if they are within `r` metres.
fn largest_cluster(pos: &[Vec3], grid: &Grid, r: f32) -> f32 {
    let n = pos.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        grid.for_each_within(pos[i], r, pos, |j, _| {
            if j > i {
                let ri = find(&mut parent, i as u32);
                let rj = find(&mut parent, j as u32);
                if ri != rj {
                    parent[ri as usize] = rj;
                }
            }
        });
    }
    let mut counts = vec![0u32; n];
    for i in 0..n {
        let root = find(&mut parent, i as u32);
        counts[root as usize] += 1;
    }
    counts.iter().copied().max().unwrap_or(0) as f32 / n as f32
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Inertia ratios and the verticality of the thin axis.
pub(crate) fn inertia(pos: &[Vec3]) -> (f32, f32, f32) {
    let n = pos.len() as f32;
    let c = pos.iter().copied().sum::<Vec3>() / n;
    let mut cov = [[0.0f32; 3]; 3];
    for p in pos {
        let d = p - c;
        let a = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += a[i] * a[j];
            }
        }
    }
    for row in &mut cov {
        for v in row {
            *v /= n;
        }
    }
    let (vals, vecs) = jacobi_eigen(cov);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| {
        vals[a]
            .partial_cmp(&vals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let shortest = vals[idx[0]].max(1e-9);
    let i2_i1 = vals[idx[1]] / shortest;
    let i3_i1 = vals[idx[2]] / shortest;
    let thin = vecs[idx[0]];
    (i2_i1, i3_i1, thin[2].abs())
}

/// Jacobi eigen-decomposition of a symmetric 3×3 matrix.
///
/// Returns eigenvalues and, in `vecs[k]`, the eigenvector for eigenvalue `k`.
#[allow(clippy::needless_range_loop)] // fixed 3×3 rotations are clearest indexed
fn jacobi_eigen(mut a: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let (mut p, mut q) = (0, 1);
        let mut max = a[0][1].abs();
        if a[0][2].abs() > max {
            max = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > max {
            max = a[1][2].abs();
            p = 1;
            q = 2;
        }
        if max < 1e-10 {
            break;
        }
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        // Angle that zeroes a[p][q]; reduce to the small root so nearly
        // degenerate matrices converge instead of rotating by ~90°.
        let mut theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        if theta > std::f32::consts::FRAC_PI_4 {
            theta -= std::f32::consts::FRAC_PI_2;
        } else if theta < -std::f32::consts::FRAC_PI_4 {
            theta += std::f32::consts::FRAC_PI_2;
        }
        let (s, c) = theta.sin_cos();
        for i in 0..3 {
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[i][q] = s * aip + c * aiq;
        }
        for i in 0..3 {
            let api = a[p][i];
            let aqi = a[q][i];
            a[p][i] = c * api - s * aqi;
            a[q][i] = s * api + c * aqi;
        }
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    let mut vecs = [[0.0f32; 3]; 3];
    for k in 0..3 {
        for i in 0..3 {
            vecs[k][i] = v[i][k];
        }
    }
    ([a[0][0], a[1][1], a[2][2]], vecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_finds_known_eigenvalues() {
        let (vals, _) = jacobi_eigen([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]]);
        let mut s = vals;
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((s[0] - 1.0).abs() < 1e-4);
        assert!((s[1] - 3.0).abs() < 1e-4);
        assert!((s[2] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn flat_cloud_has_vertical_thin_axis() {
        let mut pos = Vec::new();
        for i in 0..100 {
            for j in 0..100 {
                pos.push(Vec3::new(i as f32, j as f32, 0.01 * ((i + j) % 3) as f32));
            }
        }
        let (_, _, flatness) = inertia(&pos);
        assert!(flatness > 0.99, "thin axis not vertical: {flatness}");
    }

    #[test]
    fn a_line_of_birds_is_one_cluster() {
        let pos: Vec<Vec3> = (0..50)
            .map(|i| Vec3::new(i as f32 * 2.0, 0.0, 0.0))
            .collect();
        let mut grid = Grid::new(GRID_CELL);
        grid.rebuild(&pos);
        assert!((largest_cluster(&pos, &grid, 5.0) - 1.0).abs() < 1e-6);
    }
}
