//! The phase 4 `--analyze` and `--coherence` smoke tests.
//!
//! The light coherence sampler must agree field-for-field with the reference
//! metric set (the CPU-only test below), and both GPU runs must produce one
//! sample per whole second. The GPU tests are skipped with a clear message where
//! no adapter is found. The long 100,000-bird coherence run is not a test: it is
//! the acceptance run recorded in `docs/`.

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::{Turning, analyze, coherence};
use murmur_ref::{World, analysis};

fn gpu_or_skip() -> Option<Gpu> {
    match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => Some(gpu),
        Err(DeviceError::NoAdapter(err)) => {
            eprintln!("SKIP: no suitable GPU adapter was found ({err}); analyze test not run");
            None
        }
        Err(err) => panic!("an adapter was found but opening its device failed: {err}"),
    }
}

/// The light sampler must not drift from the reference: every field it shares
/// with `analysis::compute` is computed by the same formula in the same order,
/// so the two must be bit-identical. This runs on the CPU, no GPU needed.
#[test]
fn light_sample_matches_the_reference_metrics() {
    let world = World::new(2000, 42);
    let full = analysis::compute(&world);
    let light = coherence::sample(&world, 0.0);

    assert_eq!(light.birds, full.birds);
    assert_eq!(light.mean_neighbours, full.mean_neighbours);
    assert_eq!(light.mean_radius, full.mean_radius);
    assert_eq!(light.mean_speed, full.mean_speed);
    assert_eq!(light.spread, full.spread);
    assert_eq!(light.mean_dz, full.mean_dz);
    assert_eq!(light.height_std, full.height_std);
    assert_eq!(light.clamp_fraction, full.clamp_fraction);
    assert_eq!(light.centroid_radius, full.centroid_radius);
    assert_eq!(light.max_radius_h, full.max_radius_h);
    assert_eq!(
        light.largest_cluster_fraction,
        full.largest_cluster_fraction
    );
}

#[test]
fn analyze_reports_one_line_per_second() {
    let Some(mut gpu) = gpu_or_skip() else {
        return;
    };
    let report = analyze::run(&mut gpu, 2000, 42, 3.0, Turning::Base, true)
        .expect("the analyze run could not reopen the device");
    assert_eq!(report.per_second.len(), 3, "three whole seconds expected");
    for (i, sample) in report.per_second.iter().enumerate() {
        let t = i as f32 + 1.0;
        assert!(
            (sample.time - t).abs() < 1e-3,
            "sample {i} at t={} s, expected {t}",
            sample.time
        );
        // The GPU flock is a real flock: the reference fields are sane.
        assert!(
            (0.0..=1.0).contains(&sample.stats.largest_cluster_fraction),
            "cluster {} out of range",
            sample.stats.largest_cluster_fraction
        );
        assert!(
            sample.stats.mean_neighbours > 0.0,
            "no neighbours at t={} s",
            sample.time
        );
        assert!(
            sample.extended.is_some(),
            "--extended must produce the extra line"
        );
    }
    assert_eq!(report.summary().birds, 2000);
    assert_eq!(report.initial.birds, 2000);
    // The initial cloud has not decided yet, so it has no neighbour counts;
    // the reference keeps the same value only as a sub-second fallback.
    assert_eq!(report.initial.mean_neighbours, 0.0);
}

#[test]
fn coherence_run_samples_every_second() {
    let Some(mut gpu) = gpu_or_skip() else {
        return;
    };
    let report = coherence::run(&mut gpu, 2000, 42, 3.0, Turning::Base)
        .expect("the coherence run could not reopen the device");
    assert_eq!(report.per_second.len(), 3, "three whole seconds expected");
    for (i, sample) in report.per_second.iter().enumerate() {
        let t = i as f32 + 1.0;
        assert!(
            (sample.time - t).abs() < 1e-3,
            "sample {i} at t={} s, expected {t}",
            sample.time
        );
        assert!(
            (0.0..=1.0).contains(&sample.largest_cluster_fraction),
            "cluster {} out of range",
            sample.largest_cluster_fraction
        );
        assert!(
            sample.mean_neighbours > 0.0,
            "no neighbours at t={} s",
            sample.time
        );
    }
    // Short runs are not required to pass the long-run criteria; the checker
    // must simply read the same samples without panicking.
    let verdict = coherence::evaluate(&report);
    assert_eq!(verdict.checks.len(), 13);
}

/// A repair must not change the run.
///
/// The claim in [`murmur_gpu::analyze::each_second_resilient`] is that a flock
/// rebuilt from [`murmur_gpu::sim::RawState`] continues the *same* trajectory,
/// because the bird records carry their own random streams and the grid is
/// derived from the positions. Stepping through a repair (read back, rebuild,
/// carry on) must therefore land on exactly the state a straight run reaches.
#[test]
fn a_resumed_run_takes_the_same_steps() {
    use murmur_gpu::params::Params;
    use murmur_gpu::sim::Sim;

    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let world = World::new_density_scaled(2_000, 7);
    let base = world.snapshot();

    // Straight through: twenty steps.
    let mut direct = Sim::from_snapshot(&gpu, &base);
    direct.advance(&gpu, 20);
    let want = direct
        .raw_state(&gpu)
        .expect("reading the direct run")
        .birds;

    // The same run with a repair after seven steps.
    let mut part = Sim::from_snapshot(&gpu, &base);
    part.advance(&gpu, 7);
    let raw = part.raw_state(&gpu).expect("reading the partial run");
    let mut resumed = Sim::from_raw(&gpu, Params::from_snapshot(&base), base.seed, &raw);
    resumed.advance(&gpu, 13);
    let got = resumed
        .raw_state(&gpu)
        .expect("reading the resumed run")
        .birds;

    // Compare the whole record, byte for byte (including each bird's random
    // stream): the strongest form of "the same trajectory".
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&got),
        bytemuck::cast_slice::<_, u8>(&want),
        "a repair must land on the uninterrupted state"
    );
}
