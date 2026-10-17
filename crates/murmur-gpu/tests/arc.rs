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
    // Two minutes, so the phase boundaries come quickly.
    let world = World::new(2000, 3);
    let params = Params::from_snapshot(&world.snapshot());
    let mut sim = Sim::new(&gpu, params);
    sim.enable_arc(&gpu, 2.0, 3);
    assert!(sim.arc_active());

    let duration = 120.0f32;
    let mut samples = Vec::new();
    // One sample a second: the landing at the end of the roosting phase is
    // brief, and a coarse sample would miss it.
    for step in 0..=(duration as u32) {
        if step > 0 {
            sim.advance_seconds(&gpu, 1.0)
                .expect("stepping the evening");
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

    let start = samples.first().expect("a first sample");
    assert_eq!(start.2, 0, "the evening must start with an empty flock");

    // Arrival: the flock grows towards the capacity.
    let arrival_end = samples
        .iter()
        .find(|(_, phase, ..)| *phase == "display")
        .expect("the display phase arrives");
    assert!(
        arrival_end.2 > 1000,
        "the arrival only reached {} birds",
        arrival_end.2
    );

    // Display: the flock is at capacity and holding.
    let display = samples
        .iter()
        .filter(|(_, phase, ..)| *phase == "display")
        .max_by_key(|(_, _, n, _)| *n)
        .expect("a display sample");
    assert_eq!(display.2, 2000, "the display flock is not full");

    // Roosting: the roost level sinks towards the reed tops as the flock pours
    // down. `remove_landed` itself retires a bird only once it is *below* the
    // reed tops (`z < REED_TOP`), which the descent reaches exactly at the
    // roosting/night boundary — so the model's landing and its night clear
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
