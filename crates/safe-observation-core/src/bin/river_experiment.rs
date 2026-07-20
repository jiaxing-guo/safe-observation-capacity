fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::env::set_var("SAFE_OBSERVATION_HIGHS_THREADS", "1");

    match args.get(1).map(String::as_str) {
        Some("zoo") if args.len() == 6 => {
            std::fs::create_dir_all(&args[5]).expect("create out_dir");
            safe_observation_core::river_pipeline::run_population_river_experiment(
                args[2].as_str(),
                args[3].parse().expect("param"),
                args[4].parse().expect("salt"),
                args[5].as_str(),
            );
            return;
        }
        Some("drift") if args.len() == 7 => {
            std::fs::create_dir_all(&args[6]).expect("create out_dir");
            safe_observation_core::river_pipeline::run_drift_river_experiment(
                args[2].as_str(),
                args[3].parse().expect("n_hands"),
                args[4].parse().expect("quantiles"),
                args[5].parse().expect("seed"),
                args[6].as_str(),
            );
            return;
        }
        _ => {}
    }
    if args.len() != 6 && args.len() != 7 {
        eprintln!(
            "usage: river_experiment <leak> <n_hands> <quantiles> <seed> <out_dir> [passive]\n\
             \x20      river_experiment zoo <class> <param> <salt> <out_dir>\n\
             \x20      river_experiment drift <pair> <n_hands> <quantiles> <seed> <out_dir>"
        );
        std::process::exit(2);
    }
    let leak = args[1].as_str();
    let n_hands: u64 = args[2].parse().expect("n_hands");
    let quantiles: usize = args[3].parse().expect("quantiles");
    let seed: u64 = args[4].parse().expect("seed");
    let out_dir = args[5].as_str();

    std::fs::create_dir_all(out_dir).expect("create out_dir");
    match args.get(6).map(String::as_str) {
        None => safe_observation_core::river_pipeline::run_river_experiment(
            leak, n_hands, quantiles, seed, out_dir,
        ),
        Some("passive") => safe_observation_core::river_pipeline::run_passive_river_experiment(
            leak, n_hands, quantiles, seed, out_dir,
        ),
        Some(other) => {
            eprintln!("unknown mode {other:?} (expected `passive`)");
            std::process::exit(2);
        }
    }
}
