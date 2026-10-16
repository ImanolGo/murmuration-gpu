//! The phase 3 acceptance tests: the scan and the bucket sort against CPU
//! results, and the grid's neighbour sets against the brute-force walk.
//!
//! Skipped with a clear message where no adapter is found, like the other GPU
//! tests.

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::sim::Sim;
use murmur_ref::Snapshot;
use murmur_ref::World;
use murmur_ref::params::DT;

fn gpu_or_skip() -> Option<Gpu> {
    match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => Some(gpu),
        Err(DeviceError::NoAdapter(err)) => {
            eprintln!("SKIP: no suitable GPU adapter was found ({err}); grid tests not run");
            None
        }
        Err(err) => panic!("an adapter was found but opening its device failed: {err}"),
    }
}

/// The exclusive prefix scan and the bucket sort against a CPU computation on
/// the same random positions (PLAN.md phase 3: "unit tests for scan and sort
/// with random inputs, compared to CPU results").
#[test]
fn scan_and_sort_match_the_cpu() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    for (birds, seed) in [(2000usize, 9u64), (500, 17), (5000, 3)] {
        let world = World::new(birds, seed);
        let sim = Sim::from_snapshot(&gpu, &world.snapshot());
        sim.rebuild_grid(&gpu);

        let buckets = sim.grid.buckets() as usize;
        let offsets = sim.grid.read_offsets(&gpu);
        let sorted = sim.grid.read_sorted(&gpu, birds);
        let bucket = sim.grid.read_bucket(&gpu, birds);

        assert_eq!(offsets.len(), buckets + 1);

        // The bucket histogram and its exclusive prefix sum, on the CPU.
        let mut want = vec![0u32; buckets + 1];
        for &b in &bucket {
            assert!((b as usize) < buckets, "bucket {b} out of range");
            want[b as usize + 1] += 1;
        }
        for i in 0..buckets {
            want[i + 1] += want[i];
        }
        assert_eq!(
            offsets, want,
            "the GPU scan differs from the CPU prefix sum at {birds} birds"
        );

        // The scatter is a permutation and every bird sits in its bucket's
        // range.
        let mut seen = vec![false; birds];
        for (slot, &j) in sorted.iter().enumerate() {
            let j = j as usize;
            assert!(j < birds, "sorted slot {slot} holds bird {j}");
            assert!(!seen[j], "bird {j} appears twice in the sorted order");
            seen[j] = true;
            let b = bucket[j] as usize;
            let (start, end) = (want[b] as usize, want[b + 1] as usize);
            assert!(
                (start..end).contains(&slot),
                "bird {j} (bucket {b}) in slot {slot}, outside [{start}, {end})"
            );
        }
        assert!(seen.iter().all(|&s| s), "the sort dropped a bird");
        eprintln!("# scan and sort match the CPU at {birds} birds (seed {seed})");
    }
}

/// PLAN.md phase 3 acceptance: at 2,000 birds the grid's neighbour sets match
/// the brute-force walk for at least 99 % of birds.
#[test]
fn grid_neighbours_match_bruteforce() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let birds = 2000;
    let mut world = World::new(birds, 42);
    for _ in 0..1000 {
        world.step(DT);
    }
    let mut snapshot = world.snapshot();
    // Force every bird to decide on the next step, so all 2,000 sets are
    // compared instead of only the tenth due this step.
    let time = snapshot.time;
    for bird in &mut snapshot.birds {
        bird.next_decision = time;
    }

    // The noise is off on both engines, so the only difference left is the
    // neighbour search. The crossover is `R > 5.8 m` at 2,000 birds, above the
    // settled flock's ~2 m mean radius: measured, 1 query of 2,000 crossed it
    // (a bird in the tail of the radius distribution, which is the same effect
    // that blew the 100,000-bird run up), so this still tests the grid walk.
    let wide = compare_walks(&gpu, &snapshot, birds, "grid");
    assert!(
        wide < (birds / 100) as u32,
        "the grid test no longer exercises the grid walk: {wide} queries fell back"
    );
}

/// The bounded query: a bird whose radius covers more cells than the flock has
/// birds takes the brute-force walk instead of the grid walk, and must get the
/// same neighbours.
///
/// The model lets a straggler's radius reach `params::R_MAX = 100 m`
/// (`radius_step` clamps there), and at the grid's 1 m cell that is ~8.1e6 cells
/// for a single bird — as much as ~300,000 ordinary ones. Measured, that is what
/// made the 100,000-bird acceptance run explode at one particular simulated
/// second instead of slowing smoothly. `gather` in `sim.wgsl` therefore falls
/// back to the brute-force walk once the box outgrows the flock; the fallback is
/// a performance choice, so it is only valid if it changes nothing. This forces
/// the radii past the crossover — `(2R + 1)^3 > n` — for every bird.
#[test]
fn a_wide_radius_query_matches_bruteforce() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let birds = 2000;
    let mut world = World::new(birds, 42);
    for _ in 0..1000 {
        world.step(DT);
    }
    // 20 m radii cover 41 cells per axis, 68,921 cells, against 2,000 birds.
    world.radii.fill(20.0);
    let mut snapshot = world.snapshot();
    let time = snapshot.time;
    for bird in &mut snapshot.birds {
        bird.next_decision = time;
    }

    let wide = compare_walks(&gpu, &snapshot, birds, "bounded");
    assert!(
        wide > 0,
        "no query took the bounded fallback, so this test compared nothing new"
    );
}

/// Runs both walks from one snapshot (noise off, every bird deciding), checks
/// the grid walk reproduces the brute-force one, and returns how many queries
/// took the bounded fallback.
///
/// The set itself is compared through what `decide` derives from it, so a shared
/// neighbour count cannot hide a different set.
fn compare_walks(gpu: &Gpu, snapshot: &Snapshot, birds: usize, label: &str) -> u32 {
    let mut grid = Sim::from_snapshot(gpu, snapshot);
    grid.set_w_xi(gpu, 0.0);
    grid.advance(gpu, 1);
    let from_grid = grid.read_birds(gpu).expect("reading the grid flock back");
    let wide = grid
        .wide_queries(gpu)
        .expect("reading the step's fallback count");

    let mut brute = Sim::from_snapshot(gpu, snapshot);
    brute.set_w_xi(gpu, 0.0);
    brute.advance_bruteforce(gpu, 1);
    let from_brute = brute.read_birds(gpu).expect("reading the brute flock back");

    let mut matching = 0usize;
    let mut count_mismatches = 0usize;
    let mut worst_rel = 0.0f32;
    for (a, b) in from_grid.iter().zip(&from_brute) {
        let count_ok = a.neighbour_count == b.neighbour_count;
        if !count_ok {
            count_mismatches += 1;
        }
        let central_ok = (a.centrality - b.centrality).abs() <= 1e-4;
        let rel = rel_error(a.steering, b.steering);
        worst_rel = worst_rel.max(rel);
        if count_ok && central_ok && rel < 1e-3 {
            matching += 1;
        }
    }
    let fraction = matching as f64 / birds as f64;
    eprintln!(
        "# {label} vs brute force: {matching}/{birds} birds match ({:.2} %), \
         {count_mismatches} neighbour-count mismatches, worst steering rel error {worst_rel:.2e}, \
         {wide} queries on the fallback walk",
        fraction * 100.0
    );
    assert!(
        fraction >= 0.99,
        "only {:.2} % of neighbour sets matched brute force (the {label} walk), \
         under the 99 % target",
        fraction * 100.0
    );
    wide
}

/// `|a - b| / |b|`, with a floor so a near-zero force is not a pure ratio.
fn rel_error(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut diff = 0.0f32;
    let mut value = 0.0f32;
    for k in 0..3 {
        diff += (a[k] - b[k]) * (a[k] - b[k]);
        value += b[k] * b[k];
    }
    diff.sqrt() / value.sqrt().max(1e-3)
}
