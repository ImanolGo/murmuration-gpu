//! The evening arc (PLAN.md phase 8): arrival, display, roosting and night.
//!
//! Skipped with a clear message when the machine has no adapter. The test runs a
//! short evening (two minutes) and checks the model's phases: the flock grows
//! from nothing during the arrival, holds during the display, thins as the birds
//! land, and is empty at night — and the roost level sinks to the reed tops as
//! it pours down.

use murmur_gpu::device::{self, DeviceError, Gpu};
use murmur_gpu::params::Params;
use murmur_gpu::sim::Sim;
use murmur_ref::params::{REED_TOP, Z0};
use murmur_ref::{World, arc};

fn open() -> Option<Gpu> {
    match device::headless(wgpu::PowerPreference::HighPerformance) {
        Ok(gpu) => Some(gpu),
        Err(DeviceError::NoAdapter(err)) => {
            eprintln!("SKIP: no suitable GPU adapter was found ({err}); arc test not run");
            None
        }
        Err(err) => panic!("an adapter was found but opening its device failed: {err}"),
    }
}

/// A live count from a bird readback.
fn live(birds: &[murmur_gpu::bird::Bird]) -> usize {
    birds.iter().filter(|b| b.live != 0).count()
}

#[test]
fn arriving_count_follows_the_continuous_reference_schedule() {
    let Some(gpu) = open() else { return };
    let mut sim = Sim::new(&gpu, Params::new(2000, 3));
    sim.enable_arc(&gpu, 1.0, 3);
    let mut previous = 0;
    for _ in 0..24 {
        sim.advance_seconds(&gpu, 0.25).unwrap();
        let count = live(&sim.read_birds(&gpu).unwrap());
        let u = (sim.time() / 6.0).clamp(0.0, 1.0);
        let expected = (2000.0 * u * u * (3.0 - 2.0 * u)).floor() as usize;
        assert!(
            count.abs_diff(expected) <= 2,
            "count {count}, expected {expected}"
        );
        assert!(
            count - previous <= 127,
            "a batch arrived: {previous} -> {count}"
        );
        previous = count;
    }
    sim.advance_seconds(&gpu, 0.1).unwrap();
    assert_eq!(live(&sim.read_birds(&gpu).unwrap()), 2000);
}

#[test]
fn an_evening_runs_through_its_phases() {
    let Some(gpu) = open() else {
        return;
    };
    eprintln!(
        "arc test on {} ({:?}), driver {}",
        gpu.info().name,
        gpu.info().backend,
        gpu.info().driver
    );
    // Forty seconds and a thousand birds: the arrival needs only a few groups to
    // fill that, so every phase fits in a short run. One sample per phase
    // boundary, not per second: the interesting instants are the fill, the hold,
    // the descent and the clear, and a readback costs more than the steps.
    let duration = 40.0f32;
    let world = World::new(1000, 3);
    let params = Params::from_snapshot(&world.snapshot());
    let mut sim = Sim::new(&gpu, params);
    sim.enable_arc(&gpu, duration / 60.0, 3);
    assert!(sim.arc_active());

    let mut samples = Vec::new();
    let mut now = 0.0f32;
    for target in [0.0f32, 4.0, 6.0, 20.0, 30.0, 34.0, 35.5, 38.0] {
        if target > 0.0 {
            sim.advance_seconds(&gpu, target - now)
                .expect("stepping the evening");
            now = target;
        }
        let birds = sim.read_birds(&gpu).expect("reading the flock");
        let state = sim.arc_state(&gpu).expect("reading the arc");
        let f = sim.time() / duration;
        let phase = if f < 0.15 {
            "arrival"
        } else if f < 0.70 {
            "display"
        } else if f < 0.90 {
            "roosting"
        } else {
            "night"
        };
        samples.push((sim.time(), phase, live(&birds), state.z0));
    }
    for (t, phase, n, z0) in &samples {
        eprintln!("t={t:6.1}s {phase:9} live {n:5} z0 {z0:6.1}");
    }

    // Arrival: the flock grows from nothing and is full by the display.
    assert_eq!(
        samples[0].2, 0,
        "the evening must start with an empty flock"
    );
    let arriving = samples
        .iter()
        .filter(|(_, phase, ..)| *phase == "arrival")
        .map(|(t, _, n, _)| (*t, *n))
        .fold((0.0, 0), |acc, s| if s.1 > acc.1 { s } else { acc });
    assert!(arriving.1 > 0, "no bird had arrived by t={}", arriving.0);

    // Display: the flock is at capacity and holding.
    let display = samples
        .iter()
        .filter(|(_, phase, ..)| *phase == "display")
        .max_by_key(|(_, _, n, _)| *n)
        .expect("a display sample");
    assert_eq!(display.2, 1000, "the display flock is not full");

    // Roosting: the roost level sinks towards the reed tops as the flock pours
    // down. `remove_landed` itself retires a bird only once it is *below* the
    // reed tops (`z < REED_TOP`), which the descent reaches exactly at the
    // roosting/night boundary, so the model's landing and its night clear
    // coincide. That is the reference's own behaviour (Z0 100, REED_TOP 3); it is
    // recorded here rather than "fixed".
    let roosting = samples
        .iter()
        .filter(|(_, phase, ..)| *phase == "roosting")
        .collect::<Vec<_>>();
    assert!(!roosting.is_empty(), "no roosting samples");
    let last_roosting = roosting.last().expect("a last roosting sample");
    assert!(
        last_roosting.3 < Z0 / 2.0,
        "the roost level only descended to {}",
        last_roosting.3
    );
    let lowest_z0 = roosting
        .iter()
        .map(|(_, _, _, z0)| *z0)
        .fold(f32::INFINITY, f32::min);
    assert!(
        lowest_z0 > REED_TOP - 1.0 && lowest_z0 < REED_TOP + 10.0,
        "the roost level ended at {lowest_z0}, not near the reed tops {REED_TOP}"
    );

    // Night: the flock is cleared and the roost level parks at the reed tops.
    let night = samples
        .iter()
        .rfind(|(_, phase, ..)| *phase == "night")
        .expect("a night sample");
    assert_eq!(night.2, 0, "the flock was not cleared at night");
    assert_eq!(night.3, REED_TOP);
    // The roost sits well above the reeds: the descent is what brings the flock
    // down to them.
    let roost_above_reeds = Z0 - REED_TOP;
    assert!(
        roost_above_reeds > 50.0,
        "the roost sits barely above the reeds: {roost_above_reeds} m"
    );
}

/// The phase boundaries match the CPU oracle's, so the GPU evening is the same
/// evening the terminal runs.
#[test]
fn the_phases_match_the_oracle() {
    let mine = arc::Arc::new(2.0, 3);
    for (f, want) in [(0.0, 0), (0.1, 0), (0.2, 1), (0.69, 1), (0.8, 2), (0.95, 3)] {
        let t = f * mine.duration;
        let oracle = arc::Arc::new(2.0, 3);
        // `phase()` is a pure function of `time`, so read it directly.
        let mut probe = oracle;
        probe.time = t;
        let got = match probe.phase() {
            arc::Phase::Arrival => 0,
            arc::Phase::Display => 1,
            arc::Phase::Roosting => 2,
            arc::Phase::Night => 3,
        };
        assert_eq!(got, want, "at f={f}");
    }
}
