//! A spatial hash grid for neighbour queries.
//!
//! Integer cell coordinates are hashed into a fixed-size table, so the grid is
//! unbounded: birds far outside the roost still find each other. Each bucket
//! holds a linked list of bird indices; a query walks the buckets for the cells
//! its radius can touch and keeps the exact distance check.
//!
//! Mirrors `murmuration::sim::neighbours`. The GPU side replaces this with a
//! grid of its own (PLAN.md phase 3); the CPU version stays the oracle.

use glam::Vec3;

/// Number of hash buckets (a power of two). Must exceed the number of occupied
/// cells by a good margin so collisions stay rare.
const BUCKETS: usize = 1 << 16;

/// An unbounded uniform spatial hash with cells of size `cell`.
pub struct Grid {
    inv_cell: f32,
    /// First bird index per bucket, or −1.
    head: Vec<i32>,
    /// Next bird in the same bucket, or −1. Length N.
    next: Vec<i32>,
    /// Cell of each bird from the current build. Length N.
    cells: Vec<[i32; 3]>,
    /// Buckets occupied in the current build (so rebuild clears only those).
    touched: Vec<u32>,
}

impl Grid {
    /// Builds an empty grid with the given cell size.
    pub fn new(cell: f32) -> Self {
        Self {
            inv_cell: 1.0 / cell,
            head: vec![-1; BUCKETS],
            next: Vec::new(),
            cells: Vec::new(),
            touched: Vec::new(),
        }
    }

    /// Rebuilds the buckets from `positions`.
    pub fn rebuild(&mut self, positions: &[Vec3]) {
        for &b in &self.touched {
            self.head[b as usize] = -1;
        }
        self.touched.clear();
        self.next.resize(positions.len(), -1);
        self.cells.resize(positions.len(), [0; 3]);
        for (i, &p) in positions.iter().enumerate() {
            let cell = self.coord(p);
            self.cells[i] = cell;
            let b = self.hash(cell);
            if self.head[b] == -1 {
                self.touched.push(b as u32);
            }
            self.next[i] = self.head[b];
            self.head[b] = i as i32;
        }
    }

    fn coord(&self, p: Vec3) -> [i32; 3] {
        let f = p * self.inv_cell;
        [f.x.floor() as i32, f.y.floor() as i32, f.z.floor() as i32]
    }

    fn hash(&self, c: [i32; 3]) -> usize {
        let h = (c[0] as u32).wrapping_mul(73_856_093)
            ^ (c[1] as u32).wrapping_mul(19_349_663)
            ^ (c[2] as u32).wrapping_mul(83_492_791);
        (h as usize) & (BUCKETS - 1)
    }

    /// Calls `f(j, distance)` for every `j` whose position is within `radius`
    /// of `p`. Includes `j == i` when querying a bird against itself.
    pub fn for_each_within(
        &self,
        p: Vec3,
        radius: f32,
        positions: &[Vec3],
        mut f: impl FnMut(usize, f32),
    ) {
        let lo = self.coord(p - Vec3::splat(radius));
        let hi = self.coord(p + Vec3::splat(radius));
        let r2 = radius * radius;
        for cz in lo[2]..=hi[2] {
            for cy in lo[1]..=hi[1] {
                for cx in lo[0]..=hi[0] {
                    let cell = [cx, cy, cz];
                    let mut j = self.head[self.hash(cell)];
                    while j >= 0 {
                        let jj = j as usize;
                        // A bucket can hold birds from several cells; only the
                        // one matching `cell` belongs here, which also stops a
                        // bird being visited twice when two query cells collide.
                        if self.cells[jj] == cell {
                            let d2 = (positions[jj] - p).length_squared();
                            if d2 <= r2 {
                                f(jj, d2.sqrt());
                            }
                        }
                        j = self.next[jj];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_only_birds_within_radius() {
        let pos = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        ];
        let mut grid = Grid::new(1.0);
        grid.rebuild(&pos);
        let mut found = Vec::new();
        grid.for_each_within(Vec3::ZERO, 2.5, &pos, |j, _| found.push(j));
        found.sort_unstable();
        assert_eq!(found, vec![0, 1]);
    }

    #[test]
    fn rebuild_clears_old_cells() {
        let first = vec![Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0)];
        let second = vec![Vec3::new(50.0, 0.0, 0.0)];
        let mut grid = Grid::new(1.0);
        grid.rebuild(&first);
        grid.rebuild(&second);
        let mut found = Vec::new();
        grid.for_each_within(Vec3::ZERO, 5.0, &second, |j, _| found.push(j));
        assert!(found.is_empty());
    }

    #[test]
    fn grid_matches_bruteforce() {
        let mut state = 12345u64;
        let mut rnd = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let pos: Vec<Vec3> = (0..800)
            .map(|_| Vec3::new(rnd() * 30.0, rnd() * 30.0, rnd() * 10.0))
            .collect();
        let mut grid = Grid::new(4.0);
        grid.rebuild(&pos);
        for i in 0..pos.len() {
            for &r in &[1.0f32, 3.0, 8.0, 20.0] {
                let mut got = Vec::new();
                grid.for_each_within(pos[i], r, &pos, |j, _| got.push(j));
                got.sort_unstable();
                let mut want: Vec<usize> = (0..pos.len())
                    .filter(|&j| (pos[j] - pos[i]).length() <= r)
                    .collect();
                want.sort_unstable();
                assert_eq!(got, want, "i={i} r={r}");
            }
        }
    }

    /// Birds far outside the old ±210 m box must still find each other.
    #[test]
    fn finds_neighbours_far_from_the_origin() {
        let pos: Vec<Vec3> = (0..40)
            .map(|i| Vec3::new(300.0 + i as f32 * 0.5, 0.0, 100.0))
            .collect();
        let mut grid = Grid::new(4.0);
        grid.rebuild(&pos);
        let mut found = Vec::new();
        grid.for_each_within(pos[0], 3.0, &pos, |j, _| found.push(j));
        assert!(found.len() > 3, "only {} neighbours at 300 m", found.len());
    }
}
