//! The phase 1 acceptance test: the CPU oracle reproduces the recorded output
//! of the terminal version's `--analyze`.
//!
//! `docs/reference-analyze-2000-42.txt` is the baseline captured from
//! `murmuration --analyze 120 --birds 2000 --seed 42`. Because both engines use
//! the same `glam` and `rand` versions and the same order of operations, the
//! statistics agree digit for digit, so the test compares whole lines exactly.

use murmur_ref::params::DT;
use murmur_ref::{World, analysis, stats};

const ORACLE: &str = include_str!("../../../docs/reference-analyze-2000-42.txt");

/// The recorded `t= …s …` line for a given whole second.
fn oracle_line(seconds: f32) -> &'static str {
    let prefix = format!("t={seconds:6.1}s");
    ORACLE
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no oracle line for t={seconds}"))
}

fn line_at(world: &World) -> String {
    format!("t={:6.1}s {}", world.time, analysis::compute(world).line())
}

#[test]
fn reproduces_the_oracle_early_and_midway() {
    let mut world = World::new(2000, 42);
    for _ in 0..(20.0 / DT) as usize {
        world.step(DT);
    }
    assert_eq!(line_at(&world), oracle_line(20.0), "t=20 s diverged");
    for _ in 0..(40.0 / DT) as usize {
        world.step(DT);
    }
    assert_eq!(line_at(&world), oracle_line(60.0), "t=60 s diverged");
}

#[test]
fn extended_statistics_are_sane_once_settled() {
    let mut world = World::new(2000, 42);
    for _ in 0..(60.0 / DT) as usize {
        world.step(DT);
    }
    let s = stats::compute(&world);
    assert_eq!(s.birds, 2000);
    assert!(
        (0.9..=1.0).contains(&s.polarization),
        "polarisation {}",
        s.polarization
    );
    assert!(
        s.mean_nnd_all > 0.0 && s.mean_nnd_all < 3.0,
        "NND {}",
        s.mean_nnd_all
    );
    assert!(
        s.centre_density > s.border_density,
        "density: centre {} border {}",
        s.centre_density,
        s.border_density
    );
    assert!(s.speed_p10 <= s.speed_p50 && s.speed_p50 <= s.speed_p90);
    assert!(s.extent.z < s.extent.x, "flock should be flatter than wide");
}

/// The full acceptance run: every recorded second, exactly. Ignored by default
/// because it simulates the whole 120 s in a test build (tens of seconds).
#[test]
#[ignore = "full 120 s acceptance run; enable with --ignored"]
fn reproduces_the_full_oracle() {
    let mut world = World::new(2000, 42);
    let per_second = (1.0 / DT).round() as usize;
    let steps = (120.0 / DT) as usize;
    for s in 1..=steps {
        world.step(DT);
        if s % per_second == 0 {
            assert_eq!(
                line_at(&world),
                oracle_line(world.time),
                "t={} s diverged",
                world.time
            );
        }
    }
}
