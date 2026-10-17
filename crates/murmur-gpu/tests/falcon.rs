//! The falcon and the alarm it raises (PLAN.md phase 7).
//!
//! Skipped with a clear message when the machine has no adapter. The first test
//! flies an autonomous falcon through a settled flock and checks the response
//! the model predicts; the second isolates the alarm rule itself and compares it
//! with the CPU oracle, bird for bird, on the same frozen flock.

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::falcon::{Falcon, Mode};
use murmur_gpu::params::Params;
use murmur_gpu::sim::Sim;
use murmur_ref::params::DT;
use murmur_ref::{World, state};

fn open() -> Option<Gpu> {
    match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => Some(gpu),
        Err(DeviceError::NoAdapter(err)) => {
            eprintln!("SKIP: no suitable GPU adapter was found ({err}); falcon test not run");
            None
        }
        Err(err) => panic!("an adapter was found but opening its device failed: {err}"),
    }
}

/// A settled GPU flock, so the falcon has somewhere to hunt and the alarm rule
/// sees a realistic neighbour number.
fn settled(gpu: &Gpu, birds: usize, seed: u64) -> Sim {
    let world = World::new(birds, seed);
    let mut sim = Sim::from_snapshot(gpu, &world.snapshot());
    sim.advance_seconds(gpu, 20.0).expect("settling the flock");
    sim
}

#[test]
fn a_falcon_alarms_the_birds_it_swoops_past() {
    let Some(gpu) = open() else {
        return;
    };
    eprintln!(
        "falcon test on {} ({:?}), driver {}",
        gpu.info().name,
        gpu.info().backend,
        gpu.info().driver
    );
    let mut sim = settled(&gpu, 2000, 42);
    let z0 = Params::from_snapshot(&World::new(2000, 42).snapshot()).z0;
    sim.spawn_falcon(&gpu, 42, z0);
    assert!(sim.falcon_active());

    let mut peak_alarmed = 0.0f32;
    let mut peak_rolling = 0.0f32;
    let mut states = std::collections::BTreeSet::new();
    // The falcon spawns 220 m out and hunts its way in, so this runs half a
    // minute in one-second batches: the first attack, the wave it raises and the
    // retreat all happen inside it.
    for _ in 0..30 {
        sim.advance(&gpu, 200);
        let falcon = sim.falcon(&gpu).expect("reading the falcon");
        states.insert(falcon.state);
        let birds = sim.read_birds(&gpu).expect("reading the flock");
        let alarmed = birds.iter().filter(|b| b.alarmed != 0).count() as f32 / birds.len() as f32;
        let rolling =
            birds.iter().filter(|b| b.zigzag_active != 0).count() as f32 / birds.len() as f32;
        peak_alarmed = peak_alarmed.max(alarmed);
        peak_rolling = peak_rolling.max(rolling);
    }
    eprintln!(
        "peak alarmed {:.1}%, peak zigzagging {:.1}%, falcon states {:?}",
        peak_alarmed * 100.0,
        peak_rolling * 100.0,
        states
    );
    // The falcon patrols near the flock, so birds sit inside `R_THREAT` and the
    // roll manoeuvre starts. A falcon that never alarms anything is the failure
    // this test exists for.
    assert!(
        peak_alarmed > 0.001,
        "the falcon never alarmed a bird (peak {peak_alarmed})"
    );
    assert!(peak_rolling > 0.0, "the alarm never started a zigzag");
}

/// The alarm rule, bird for bird, against the CPU oracle.
///
/// The GPU's random stream differs from the oracle's (`docs/model.md`), and so
/// do the trajectories after one step, so this compares the *alarm* alone on a
/// frozen flock: the same positions, the same falcon. The falcon is
/// mouse-steered at its own position, where both engines run the identical
/// update (no centroid to disagree about) and the same flash-expansion test.
#[test]
fn the_alarm_matches_the_oracle_on_a_frozen_flock() {
    let Some(gpu) = open() else {
        return;
    };
    let mut sim = settled(&gpu, 2000, 7);
    let flock = sim.snapshot(&gpu);
    let centroid = flock.birds.iter().map(|b| b.pos).sum::<glam::Vec3>() / flock.birds.len() as f32;

    // A fast falcon just below the flock centre: inside `R_FLASH` for the
    // nearest birds, so the flash term is exercised too.
    let mut falcon = Falcon::spawn(7, flock.z0);
    falcon.pos = (centroid - glam::Vec3::Z * 5.0).to_array();
    falcon.vel = [0.0, 40.0, 0.0];
    falcon.speed = 40.0;
    falcon.mode = Mode::Mouse.code();
    falcon.aim = falcon.pos;
    sim.write_falcon(&gpu, &falcon);
    sim.advance(&gpu, 1);
    let gpu_birds = sim.read_birds(&gpu).expect("reading the flock");

    // The oracle, one step of the same flock and the same falcon.
    let mut world = state::import(&flock);
    let mut cpu_falcon = murmur_ref::falcon::Falcon::new(7, flock.z0);
    cpu_falcon.mode = murmur_ref::falcon::Mode::Mouse;
    cpu_falcon.position = glam::Vec3::from_array(falcon.pos);
    cpu_falcon.velocity = glam::Vec3::from_array(falcon.vel);
    cpu_falcon.target = cpu_falcon.position;
    world.falcon = Some(cpu_falcon);
    world.step(DT);

    let alarmed_gpu = gpu_birds.iter().filter(|b| b.alarmed != 0).count();
    let alarmed_cpu = world.escape.iter().filter(|e| e.alarmed).count();
    let flash_gpu = gpu_birds
        .iter()
        .filter(|b| b.flash_until > flock.time)
        .count();
    let flash_cpu = world
        .escape
        .iter()
        .filter(|e| e.flash_until > flock.time)
        .count();
    eprintln!(
        "falcon test: alarmed gpu {alarmed_gpu} cpu {alarmed_cpu}; flash gpu {flash_gpu} cpu {flash_cpu}"
    );
    // The falcon moves a fraction of a millimetre differently (the two `exp`
    // implementations), so a bird on the exact `R_THREAT` boundary may fall on
    // either side; the rule, not the boundary, is what is compared.
    assert!(
        (alarmed_gpu as i64 - alarmed_cpu as i64).abs() <= 2,
        "different birds alarmed: gpu {alarmed_gpu}, cpu {alarmed_cpu}"
    );
    assert_eq!(
        flash_gpu, flash_cpu,
        "the flash expansion fired on different birds"
    );

    // The roll direction is taken from the falcon's side of each bird; both
    // engines must choose the same sign.
    let mut compared = 0;
    let mut agree = 0;
    for (i, b) in gpu_birds.iter().enumerate() {
        let e = world.escape[i];
        if b.alarmed != 0 && e.alarmed && b.zigzag_active != 0 && e.zigzag_active {
            compared += 1;
            if (b.zigzag_sign - e.zigzag_sign).abs() < 1e-6 {
                agree += 1;
            }
        }
    }
    eprintln!("roll direction agrees on {agree}/{compared} alarmed birds");
    assert!(
        compared == 0 || agree as f32 / compared as f32 > 0.95,
        "the roll direction differs too often: {agree}/{compared}"
    );
}

/// A falcon attack must alarm a growing share of the flock as it sweeps through,
/// and the zigzag must then copy through the rest: the "dark wave" of
/// Hemelrijk et al. (2015) and the phase 7 acceptance. The timeline is printed
/// so the wave's shape (rise, peak, decay over a few seconds) is visible and not
/// just a threshold.
#[test]
fn the_attack_sweeps_the_flock() {
    let Some(gpu) = open() else {
        return;
    };
    // Spawned at t = 0, as the app and the terminal version do, so `until` is an
    // absolute time the falcon can actually reach.
    let world = World::new(2000, 42);
    let mut sim = Sim::from_snapshot(&gpu, &world.snapshot());
    let z0 = Params::from_snapshot(&World::new(2000, 42).snapshot()).z0;
    sim.spawn_falcon(&gpu, 42, z0);

    let mut timeline = Vec::new();
    let mut peak_rolling = 0.0f32;
    // Three seconds per printed sample would hide the wave; 0.25 s shows it.
    for sample in 0..160 {
        sim.advance(&gpu, 50);
        let birds = sim.read_birds(&gpu).expect("reading the flock");
        let rolling =
            birds.iter().filter(|b| b.zigzag_active != 0).count() as f32 / birds.len() as f32;
        peak_rolling = peak_rolling.max(rolling);
        if sample % 4 == 0 {
            let f = sim.falcon(&gpu).expect("falcon");
            timeline.push((sample as f32 * 0.25, rolling, f.state, f.pos));
        }
    }
    let line: String = timeline
        .iter()
        .map(|(t, r, st, p)| format!("{t:.0}:{:.0}%/s{st}/{:.0} ", r * 100.0, p[1]))
        .collect();
    eprintln!("time:share/state/falcon.y {line}");
    assert!(
        peak_rolling > 0.2,
        "the wave never covered a fifth of the flock (peak {peak_rolling})"
    );
}
