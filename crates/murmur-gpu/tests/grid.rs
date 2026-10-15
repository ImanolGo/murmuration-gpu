//! The phase 3 acceptance tests: the scan and the bucket sort against CPU
//! results, and the grid's neighbour sets against the brute-force walk.
//!
//! Skipped with a clear message where no adapter is found, like the other GPU
//! tests.

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::sim::Sim;
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
///
/// The set itself is compared through what `decide` derives from it: the
/// neighbour count, the centrality (a second walk over 2R_i) and the steering
/// force (a sum over the set). Two different sets that happened to share a
/// count would still differ in the last two.
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
    // neighbour search.
    let mut grid = Sim::from_snapshot(&gpu, &snapshot);
    grid.set_w_xi(&gpu, 0.0);
    grid.advance(&gpu, 1);
    let from_grid = grid.read_birds(&gpu);

    let mut brute = Sim::from_snapshot(&gpu, &snapshot);
    brute.set_w_xi(&gpu, 0.0);
    brute.advance_bruteforce(&gpu, 1);
    let from_brute = brute.read_birds(&gpu);

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
        "# grid vs brute force: {matching}/{birds} birds match ({:.2} %), \
         {count_mismatches} neighbour-count mismatches, worst steering rel error {worst_rel:.2e}",
        fraction * 100.0
    );
    assert!(
        fraction >= 0.99,
        "only {:.2} % of neighbour sets matched brute force, under the 99 % target",
        fraction * 100.0
    );
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
