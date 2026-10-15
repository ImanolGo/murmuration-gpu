//! Headless `--analyze` for the CPU oracle.
//!
//! Prints the same per-second line and summary as the terminal version's
//! `--analyze`, so the output can be diffed against
//! `docs/reference-analyze-2000-42.txt`. With `--extended` it also prints the
//! extra [`murmur_ref::stats`] line each second.
//!
//! Usage:
//!   cargo run -p murmur-ref --bin analyze --release -- \
//!       --birds 2000 --seed 42 --seconds 120 [--falcon] [--extended]

use murmur_ref::params::DT;
use murmur_ref::{World, analysis, flight, stats};

struct Args {
    birds: usize,
    seed: u64,
    seconds: f32,
    falcon: bool,
    extended: bool,
    turning: flight::Turning,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        birds: 2000,
        seed: 42,
        seconds: 120.0,
        falcon: false,
        extended: false,
        turning: flight::Turning::Base,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--birds" => {
                args.birds = next(&mut it, "--birds")?
                    .parse()
                    .map_err(|e| format!("--birds: {e}"))?;
            }
            "--seed" => {
                args.seed = next(&mut it, "--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?;
            }
            "--seconds" => {
                args.seconds = next(&mut it, "--seconds")?
                    .parse()
                    .map_err(|e| format!("--seconds: {e}"))?;
            }
            "--falcon" => args.falcon = true,
            "--extended" => args.extended = true,
            "--turning" => {
                let value = next(&mut it, "--turning")?;
                args.turning = match value.as_str() {
                    "base" => flight::Turning::Base,
                    "inertial" => flight::Turning::Inertial,
                    other => return Err(format!("--turning: unknown value {other:?}")),
                };
            }
            "-h" | "--help" => {
                println!(
                    "analyze --birds N --seed S --seconds T [--falcon] [--extended] \
                     [--turning base|inertial]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(args)
}

fn next(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let mut world = World::new(args.birds, args.seed);
    world.turning = args.turning;
    if args.falcon {
        world.spawn_falcon();
    }

    let steps = (args.seconds / DT).max(0.0) as usize;
    let per_second = (1.0 / DT).round() as usize;
    println!(
        "# murmuration-gpu-ref --analyze birds={} seed={} seconds={} falcon={} turning={:?}",
        args.birds, args.seed, args.seconds, args.falcon, args.turning
    );

    let mut last = analysis::compute(&world);
    for s in 1..=steps {
        world.step(DT);
        if s % per_second == 0 {
            last = analysis::compute(&world);
            println!("t={:6.1}s {}", world.time, last.line());
            if args.extended {
                println!("        {}", stats::compute(&world).line());
            }
        }
    }
    println!("# summary  {}", last.line());
}
