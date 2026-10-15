//! The phase 2 acceptance tests: single-step agreement, short-horizon lock-in
//! and multi-seed statistical parity.
//!
//! Skipped with a clear message where no adapter is found, like the smoke test.
//! The full 60-second, five-seed run is `#[ignore]`d (minutes of CPU flocking)
//! and is also what `murmuration-gpu --parity` drives.

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::parity;
use murmur_ref::World;
use murmur_ref::params::DT;

fn gpu_or_skip() -> Option<Gpu> {
    match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => Some(gpu),
        Err(DeviceError::NoAdapter(err)) => {
            eprintln!("SKIP: no suitable GPU adapter was found ({err}); parity tests not run");
            None
        }
        Err(err) => panic!("an adapter was found but opening its device failed: {err}"),
    }
}

/// A developed flock, so one step exercises the whole decision path.
fn developed_world(birds: usize, seed: u64) -> World {
    let mut world = World::new(birds, seed);
    for _ in 0..1000 {
        world.step(DT);
    }
    world
}

#[test]
fn single_step_matches_cpu_to_one_part_in_ten_thousand() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let world = developed_world(2000, 42);
    let error = parity::single_step_max_rel_error(&gpu, &world);
    eprintln!("single step (noise off) max relative error per bird: {error:.3e}");
    assert!(
        error < 1e-4,
        "one step from the same state differed by {error:.3e} relative, over 1e-4"
    );
}

#[test]
fn single_step_noise_is_the_expected_floor() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let world = developed_world(2000, 42);
    let error = parity::single_step_noise_floor(&gpu, &world);
    eprintln!("single step (noise on) max relative error per bird: {error:.3e}");
    // With the noise on, the two engines draw from independent generators (a
    // documented adaptation), so the residual is the noise floor; relative to a
    // bird's position it is around 1e-7, far below any use of the statistic.
    assert!(
        error < 1e-3,
        "noise-on single step differed by {error:.3e}, unexpectedly large"
    );
}

/// Both engines stepped together stay on the same trajectory: every reference
/// metric within its tolerance for the first five seconds. This is the check
/// that the shader mirrors the dynamics, not merely the statistics; the 60 s
/// statistical run below cannot see it, because the trajectories diverge.
#[test]
fn both_engines_lock_in_for_the_first_seconds() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut all_ok = true;
    for seed in [1u64, 2, 3, 4, 5] {
        let metrics = parity::lock_in(&gpu, seed, 2000, 5);
        all_ok &= metrics.iter().all(|m| m.ok);
        eprintln!("# seed {seed} lock-in (worst per second, t = 1..5 s)");
        eprint!("{}", parity::table("lock-in", &metrics));
    }
    assert!(
        all_ok,
        "an engine left the CPU trajectory within five seconds"
    );
}

/// PLAN.md phase 2 acceptance: 60 s, five seeds, GPU within tolerance of CPU.
#[test]
#[ignore = "60 s x five seeds of CPU flocking; run with --ignored"]
fn parity_sixty_seconds_over_five_seeds() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut all_ok = true;
    for seed in [1u64, 2, 3, 4, 5] {
        let report = parity::run_seed(&gpu, seed, 2000, 60);
        eprintln!("# seed {seed} settled (t = 30..60 s)");
        eprint!("{}", parity::table("settled", &report.settled));
        all_ok &= report.passed();
    }
    assert!(all_ok, "a settled metric was outside its tolerance");
}
