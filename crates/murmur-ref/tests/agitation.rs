//! The phase 7 acceptance test for the agitation wave, on the CPU oracle.
//!
//! `murmuration --agitation --birds 2000 --seed 42 --zigzag-k 2` reports
//! `wave speed = 7.56 m/s, R2 = 0.952, samples = 2000, reached = 100%,
//! expansion = 22%` for the sibling terminal version. The oracle mirrors its
//! step and its seeding, so it reproduces that run; the GPU experiment
//! (`murmuration-gpu --agitation`) is compared against this range.

use murmur_ref::World;
use murmur_ref::escape::measure_agitation;
use murmur_ref::params::DT;

#[test]
fn the_agitation_wave_matches_the_terminal_run() {
    let mut world = World::new(2000, 42);
    world.zigzag_k = 2;
    let wave = measure_agitation(&mut world, 4.0);
    println!(
        "# agitation cpu 2000 birds seed 42 k=2: speed {:.2} m/s, R2 {:.3}, samples {}, \
         reached {:.0}%, expansion {:.0}%",
        wave.speed,
        wave.r2,
        wave.samples,
        wave.peak_alarmed * 100.0,
        wave.peak_expansion * 100.0,
    );
    assert!(
        (wave.speed - 7.56).abs() < 0.05,
        "wave speed {} differs from the terminal's 7.56 m/s",
        wave.speed
    );
    assert!(wave.peak_alarmed > 0.95, "the wave did not reach the flock");
}

/// The measurement must stay inside the range the terminal version reports for
/// its own agitation runs at this bird count (the model's own calibration).
#[test]
fn the_wave_speed_is_inside_the_reference_range() {
    for (seed, expected) in [(42u64, 7.56f32), (1, 6.85), (7, 6.81)] {
        let mut world = World::new(2000, seed);
        world.zigzag_k = 2;
        let wave = measure_agitation(&mut world, 4.0);
        assert!(
            (wave.speed - expected).abs() < 0.1,
            "seed {seed}: {} vs the terminal's {expected} m/s",
            wave.speed
        );
        assert!(
            (3.0..=25.0).contains(&wave.speed),
            "seed {seed}: {} m/s is outside the reference's 3-25 m/s range",
            wave.speed
        );
    }
}

/// The wave is fitted in seconds, not steps: the propagation speed must land in
/// the metres-per-second range the model reports, not in a rate per step.
#[test]
fn the_wave_is_measured_in_seconds() {
    let mut world = World::new(2000, 42);
    world.zigzag_k = 2;
    let wave = measure_agitation(&mut world, 4.0);
    // A 15 m/s wave crosses the ~40 m flock in ~3 s; a step-counted wave would
    // report ~0.003 m/s, and the fit would still be good.
    assert!(wave.r2 > 0.5, "the fit is not a line: R2 = {}", wave.r2);
    let flock_metres = 40.0;
    let crossing_seconds = flock_metres / wave.speed;
    assert!(
        (0.5..20.0).contains(&crossing_seconds),
        "the wave takes {crossing_seconds:.2} s to cross {flock_metres} m"
    );
    let _ = DT;
}
