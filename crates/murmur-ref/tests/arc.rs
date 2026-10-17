//! The evening, on the CPU oracle (PLAN.md phase 8).
//!
//! pinned so the GPU's evening (`crates/murmur-gpu/tests/arc.rs`) can be
//! compared against the same phases and the same shape of flock.

use murmur_ref::World;
use murmur_ref::params::DT;

/// The RMS spread of a flock about its centroid \[m\].
fn spread(world: &World) -> f32 {
    let n = world.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let c = world.position.iter().copied().sum::<glam::Vec3>() / n;
    (world
        .position
        .iter()
        .map(|p| (*p - c).length_squared())
        .sum::<f32>()
        / n)
        .sqrt()
}

#[test]
fn an_evening_arrives_and_displays() {
    let mut world = World::new(2000, 3);
    world.enable_arc(3.0);
    let mut trace = Vec::new();
    for second in 1..=180u32 {
        for _ in 0..(1.0 / DT) as u32 {
            world.step(DT);
        }
        if second % 20 == 0 {
            let phase = world.arc.as_ref().map(|a| a.phase());
            trace.push((
                second,
                world.len(),
                spread(&world),
                world.z0,
                world.evening(),
                phase,
            ));
        }
    }
    for (t, n, s, z0, e, phase) in &trace {
        eprintln!(
            "t={t:4}s live {n:5} spread {s:6.1} m z0 {z0:6.1} evening {e:.2} phase {phase:?}"
        );
    }

    // Arrival fills the flock; the display holds one flock near the roost.
    let display = trace
        .iter()
        .find(|(t, ..)| *t == 120)
        .expect("a display sample");
    assert_eq!(display.1, 2000, "the flock is not full at the display");
    assert!(
        display.2 < 120.0,
        "the display flock is spread over {} m",
        display.2
    );
    // Night clears it. The last sample of the evening is at t=160; by t=180 the
    // next evening has already begun to fill the flock again.
    let night = trace
        .iter()
        .find(|(t, ..)| *t == 160)
        .expect("a roosting sample");
    assert!(
        matches!(night.5, Some(murmur_ref::arc::Phase::Roosting)),
        "t=160 is not in the roosting phase: {:?}",
        night.5
    );
    let cleared = trace
        .iter()
        .find(|(t, ..)| *t == 180)
        .expect("a last sample");
    // At t=180 the evening has wrapped, so the flock is whatever the next
    // arrival has spawned; the night's clear happened inside that second.
    assert!(
        cleared.1 < 2000,
        "the flock was not cleared: {} birds",
        cleared.1
    );
}
