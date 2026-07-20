use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use safe_observation_core::best_response as tree_br;
use safe_observation_core::censored_chain;
use safe_observation_core::goofspiel::Goofspiel;
use safe_observation_core::holdem::canonical_holdem;
use safe_observation_core::kuhn::Kuhn;
use safe_observation_core::leduc::Leduc;
use safe_observation_core::sequence_form as seq_form;
use safe_observation_core::{
    confidence, goofspiel, holdem, kuhn, leduc, lp, payoff, probe, sim, solver,
    version as core_version,
};

fn unknown_game(game: &str) -> PyErr {
    PyValueError::new_err(format!(
        "unknown game {game:?}; expected 'kuhn', 'leduc', or 'goofspiel'"
    ))
}

fn check_player(player: usize) -> PyResult<()> {
    if player > 1 {
        return Err(PyValueError::new_err(
            "player must be 0 (player 1) or 1 (player 2)",
        ));
    }
    Ok(())
}

struct NativeGameData {
    sf0: seq_form::SequenceForm,
    sf1: seq_form::SequenceForm,
    payoff: payoff::PayoffMatrix,
}

static GAME_DATA_CACHE: OnceLock<Mutex<HashMap<String, Arc<NativeGameData>>>> = OnceLock::new();

fn build_game_data(game: &str) -> PyResult<NativeGameData> {
    match game {
        "kuhn" => Ok(NativeGameData {
            sf0: seq_form::compile_kuhn(0),
            sf1: seq_form::compile_kuhn(1),
            payoff: payoff::build_kuhn(),
        }),
        "leduc" => Ok(NativeGameData {
            sf0: leduc::compile_leduc(0),
            sf1: leduc::compile_leduc(1),
            payoff: leduc::build_leduc(),
        }),
        "goofspiel" => Ok(NativeGameData {
            sf0: goofspiel::compile_goofspiel(0),
            sf1: goofspiel::compile_goofspiel(1),
            payoff: goofspiel::build_goofspiel(),
        }),
        "holdem" => {
            let game = canonical_holdem();
            let sf0 = seq_form::compile(&game, 0);
            let sf1 = seq_form::compile(&game, 1);
            let payoff = payoff::build(&game, &sf0, &sf1);
            Ok(NativeGameData { sf0, sf1, payoff })
        }
        g if g.starts_with("holdem") => {
            let game = holdem_game(g).ok_or_else(|| unknown_game(g))?;
            let sf0 = seq_form::compile(&game, 0);
            let sf1 = seq_form::compile(&game, 1);
            let payoff = payoff::build(&game, &sf0, &sf1);
            Ok(NativeGameData { sf0, sf1, payoff })
        }
        g if g.starts_with("cchain") => {
            let game = chain_game(g).ok_or_else(|| unknown_game(g))?;
            let sf0 = seq_form::compile(&game, 0);
            let sf1 = seq_form::compile(&game, 1);
            let payoff = payoff::build(&game, &sf0, &sf1);
            Ok(NativeGameData { sf0, sf1, payoff })
        }
        _ => Err(unknown_game(game)),
    }
}

fn game_data(game: &str) -> PyResult<Arc<NativeGameData>> {
    let cache = GAME_DATA_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(data) = cache
        .lock()
        .expect("game cache poisoned")
        .get(game)
        .cloned()
    {
        return Ok(data);
    }
    let data = Arc::new(build_game_data(game)?);
    cache
        .lock()
        .expect("game cache poisoned")
        .insert(game.to_string(), data.clone());
    Ok(data)
}

fn timing_enabled() -> bool {
    std::env::var("SAFE_OBSERVATION_TIMERS")
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(false)
}

fn timing_start() -> Option<Instant> {
    timing_enabled().then(Instant::now)
}

fn timing_log(timer: Option<Instant>, scope: &str, stage: &str, details: impl AsRef<str>) {
    if let Some(start) = timer {
        let elapsed = start.elapsed().as_secs_f64();
        let details = details.as_ref();
        let mut stderr = std::io::stderr().lock();
        if details.is_empty() {
            let _ = writeln!(stderr, "[timer rust +{elapsed:8.3}s] {scope}:{stage}");
        } else {
            let _ = writeln!(
                stderr,
                "[timer rust +{elapsed:8.3}s] {scope}:{stage} {details}"
            );
        }
        let _ = stderr.flush();
    }
}

fn holdem_game(game: &str) -> Option<holdem::HoldemGame> {
    use holdem::HoldemGame;
    if let Some(suffix) = game.strip_prefix("holdem_tr") {
        return holdem::turn_river_game(suffix).map(HoldemGame::TurnRiver);
    }
    match game {
        "holdem" => Some(HoldemGame::River(canonical_holdem())),
        _ => game
            .strip_prefix("holdem_")
            .and_then(holdem::holdem_variant)
            .map(HoldemGame::River),
    }
}

fn chain_game(game: &str) -> Option<censored_chain::ChainGame> {
    game.strip_prefix("cchain")
        .and_then(censored_chain::chain_game)
}

fn compile(game: &str, player: usize) -> PyResult<seq_form::SequenceForm> {
    match game {
        "kuhn" => Ok(seq_form::compile_kuhn(player)),
        "leduc" => Ok(leduc::compile_leduc(player)),
        "goofspiel" => Ok(goofspiel::compile_goofspiel(player)),
        "holdem" => Ok(holdem::compile_holdem(player)),
        g if g.starts_with("holdem") => holdem_game(g)
            .map(|game| seq_form::compile(&game, player))
            .ok_or_else(|| unknown_game(g)),
        g if g.starts_with("cchain") => chain_game(g)
            .map(|game| seq_form::compile(&game, player))
            .ok_or_else(|| unknown_game(g)),
        _ => Err(unknown_game(game)),
    }
}

fn payoff_of(game: &str) -> PyResult<payoff::PayoffMatrix> {
    match game {
        "kuhn" => Ok(payoff::build_kuhn()),
        "leduc" => Ok(leduc::build_leduc()),
        "goofspiel" => Ok(goofspiel::build_goofspiel()),
        "holdem" => Ok(holdem::build_holdem()),
        g if g.starts_with("holdem") => {
            let game = holdem_game(g).ok_or_else(|| unknown_game(g))?;
            let sf0 = seq_form::compile(&game, 0);
            let sf1 = seq_form::compile(&game, 1);
            Ok(payoff::build(&game, &sf0, &sf1))
        }
        g if g.starts_with("cchain") => {
            let game = chain_game(g).ok_or_else(|| unknown_game(g))?;
            let sf0 = seq_form::compile(&game, 0);
            let sf1 = seq_form::compile(&game, 1);
            Ok(payoff::build(&game, &sf0, &sf1))
        }
        _ => Err(unknown_game(game)),
    }
}

fn probe_coeffs_of(
    game: &str,
    sf0: &seq_form::SequenceForm,
    opp_behavior: &HashMap<String, Vec<f64>>,
    weights: &HashMap<String, f64>,
) -> PyResult<Vec<f64>> {
    match game {
        "kuhn" => Ok(probe::probe_reach_coeffs(&Kuhn, sf0, opp_behavior, weights)),
        "leduc" => Ok(probe::probe_reach_coeffs(
            &Leduc,
            sf0,
            opp_behavior,
            weights,
        )),
        "goofspiel" => Ok(probe::probe_reach_coeffs(
            &Goofspiel,
            sf0,
            opp_behavior,
            weights,
        )),
        "holdem" => Ok(probe::probe_reach_coeffs(
            &canonical_holdem(),
            sf0,
            opp_behavior,
            weights,
        )),
        g if g.starts_with("holdem") => Ok(probe::probe_reach_coeffs(
            &holdem_game(g).ok_or_else(|| unknown_game(g))?,
            sf0,
            opp_behavior,
            weights,
        )),
        _ => Err(unknown_game(game)),
    }
}

fn reach_weights_of(
    game: &str,
    sf0: &seq_form::SequenceForm,
    x_agent: &[f64],
) -> PyResult<HashMap<String, f64>> {
    match game {
        "kuhn" => Ok(confidence::opponent_reach_weights(&Kuhn, sf0, x_agent)),
        "leduc" => Ok(confidence::opponent_reach_weights(&Leduc, sf0, x_agent)),
        "goofspiel" => Ok(confidence::opponent_reach_weights(&Goofspiel, sf0, x_agent)),
        "holdem" => Ok(confidence::opponent_reach_weights(
            &canonical_holdem(),
            sf0,
            x_agent,
        )),
        g if g.starts_with("holdem") => Ok(confidence::opponent_reach_weights(
            &holdem_game(g).ok_or_else(|| unknown_game(g))?,
            sf0,
            x_agent,
        )),
        g if g.starts_with("cchain") => Ok(confidence::opponent_reach_weights(
            &chain_game(g).ok_or_else(|| unknown_game(g))?,
            sf0,
            x_agent,
        )),
        _ => Err(unknown_game(game)),
    }
}

fn showdown_reach_of(
    game: &str,
    sf0: &seq_form::SequenceForm,
    x_agent: &[f64],
) -> PyResult<HashMap<String, Vec<(f64, bool)>>> {
    match game {
        "kuhn" => Ok(confidence::agent_showdown_reach(&Kuhn, sf0, x_agent)),
        "leduc" => Ok(confidence::agent_showdown_reach(&Leduc, sf0, x_agent)),
        "goofspiel" => Ok(confidence::agent_showdown_reach(&Goofspiel, sf0, x_agent)),
        "holdem" => Ok(confidence::agent_showdown_reach(
            &canonical_holdem(),
            sf0,
            x_agent,
        )),
        g if g.starts_with("holdem") => Ok(confidence::agent_showdown_reach(
            &holdem_game(g).ok_or_else(|| unknown_game(g))?,
            sf0,
            x_agent,
        )),
        g if g.starts_with("cchain") => Ok(confidence::agent_showdown_reach(
            &chain_game(g).ok_or_else(|| unknown_game(g))?,
            sf0,
            x_agent,
        )),
        _ => Err(unknown_game(game)),
    }
}

#[pyfunction]
fn version() -> String {
    core_version().to_string()
}

#[pyfunction]
#[pyo3(signature = (iterations = 100_000))]
fn solve_kuhn(iterations: u64) -> (f64, BTreeMap<String, Vec<f64>>) {
    let solution = kuhn::solve(iterations);
    let strategy = solution
        .strategy
        .into_iter()
        .map(|(key, probs)| (key, probs.to_vec()))
        .collect();
    (solution.value, strategy)
}

#[pyfunction]
#[pyo3(signature = (game, iterations = 100_000, seed = 2026))]
fn blueprint_mccfr(
    game: &str,
    iterations: u64,
    seed: u64,
) -> PyResult<(f64, HashMap<String, Vec<f64>>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.blueprint_mccfr",
        "start",
        format!("game={game} iterations={iterations} seed={seed}"),
    );
    let sf0 = compile(game, 0)?;
    timing_log(
        timer,
        "native.blueprint_mccfr",
        "compile_sf0",
        format!(
            "seq={} infosets={}",
            sf0.num_sequences(),
            sf0.num_infosets()
        ),
    );
    let sf1 = compile(game, 1)?;
    timing_log(
        timer,
        "native.blueprint_mccfr",
        "compile_sf1",
        format!(
            "seq={} infosets={}",
            sf1.num_sequences(),
            sf1.num_infosets()
        ),
    );
    let bp = match game {
        "kuhn" => solver::solve_blueprint_mccfr(&Kuhn, &sf0, &sf1, iterations, seed),
        "leduc" => solver::solve_blueprint_mccfr(&Leduc, &sf0, &sf1, iterations, seed),
        "goofspiel" => solver::solve_blueprint_mccfr(&Goofspiel, &sf0, &sf1, iterations, seed),
        g if g.starts_with("holdem") => {
            let game = holdem_game(g).ok_or_else(|| unknown_game(g))?;
            solver::solve_blueprint_mccfr(&game, &sf0, &sf1, iterations, seed)
        }
        _ => return Err(unknown_game(game)),
    };
    timing_log(
        timer,
        "native.blueprint_mccfr",
        "solve_done",
        format!("value={:.9}", bp.value),
    );
    Ok((bp.value, bp.avg_behavior0))
}

type SequenceFormPy = (
    Vec<String>,
    Vec<(String, usize, Vec<(String, usize)>)>,
    Vec<(usize, usize, f64)>,
    Vec<f64>,
);

fn serialize_sf(sf: &seq_form::SequenceForm) -> SequenceFormPy {
    let info_sets = sf
        .info_sets
        .iter()
        .map(|info| {
            let children: Vec<(String, usize)> = info
                .children
                .iter()
                .map(|&(sym, child)| (sym.to_string(), child))
                .collect();
            (info.label.clone(), info.parent_seq, children)
        })
        .collect();
    (
        sf.sequences.clone(),
        info_sets,
        sf.constraint_entries(),
        sf.constraint_rhs(),
    )
}

#[pyfunction]
fn sequence_form(game: &str, player: usize) -> PyResult<SequenceFormPy> {
    check_player(player)?;
    Ok(serialize_sf(&compile(game, player)?))
}

#[pyfunction]
fn sequence_form_sizes(game: &str) -> PyResult<(usize, usize, usize, usize)> {
    let s0 = compile(game, 0)?;
    let s1 = compile(game, 1)?;
    Ok((
        s0.num_sequences(),
        s0.num_infosets(),
        s1.num_sequences(),
        s1.num_infosets(),
    ))
}

#[pyfunction]
fn blueprint_lp(game: &str) -> PyResult<(f64, Vec<f64>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.blueprint_lp",
        "start",
        format!("game={game}"),
    );
    let sf0 = compile(game, 0)?;
    timing_log(
        timer,
        "native.blueprint_lp",
        "compile_sf0",
        format!(
            "seq={} infosets={}",
            sf0.num_sequences(),
            sf0.num_infosets()
        ),
    );
    let sf1 = compile(game, 1)?;
    timing_log(
        timer,
        "native.blueprint_lp",
        "compile_sf1",
        format!(
            "seq={} infosets={}",
            sf1.num_sequences(),
            sf1.num_infosets()
        ),
    );
    let payoff = payoff_of(game)?;
    timing_log(
        timer,
        "native.blueprint_lp",
        "payoff",
        format!(
            "rows={} cols={} nnz={}",
            payoff.nrows,
            payoff.ncols,
            payoff.nnz()
        ),
    );
    let sol = lp::solve_blueprint(&sf0, &sf1, &payoff);
    timing_log(
        timer,
        "native.blueprint_lp",
        "solve_done",
        format!("value={:.9}", sol.value),
    );
    Ok((sol.value, sol.realization))
}

#[pyfunction]
fn blueprint_realization(game: &str, player: usize) -> PyResult<Vec<f64>> {
    check_player(player)?;
    let sf0 = compile(game, 0)?;
    let sf1 = compile(game, 1)?;
    let a = payoff_of(game)?;
    if player == 0 {
        Ok(lp::solve_blueprint(&sf0, &sf1, &a).realization)
    } else {
        let b = payoff::PayoffMatrix {
            nrows: a.ncols,
            ncols: a.nrows,
            entries: a.entries.iter().map(|&(r, c, v)| (c, r, -v)).collect(),
        };
        Ok(lp::solve_blueprint(&sf1, &sf0, &b).realization)
    }
}

#[pyfunction]
fn safety_verify(game: &str, x: Vec<f64>) -> PyResult<(f64, Vec<f64>)> {
    let sf0 = compile(game, 0)?;
    if x.len() != sf0.num_sequences() {
        return Err(PyValueError::new_err(format!(
            "x has length {}, expected {}",
            x.len(),
            sf0.num_sequences()
        )));
    }
    let r = tree_br::safety_verify_from_matrix(&compile(game, 1)?, &payoff_of(game)?, &x);
    Ok((r.value, r.realization))
}

#[pyfunction]
fn best_response(game: &str, y: Vec<f64>) -> PyResult<(f64, Vec<f64>)> {
    let sf1 = compile(game, 1)?;
    if y.len() != sf1.num_sequences() {
        return Err(PyValueError::new_err(format!(
            "y has length {}, expected {}",
            y.len(),
            sf1.num_sequences()
        )));
    }
    let r = tree_br::best_response_p1_from_matrix(&compile(game, 0)?, &payoff_of(game)?, &y);
    Ok((r.value, r.realization))
}

#[pyfunction]
fn safety_constrained_best_response(
    game: &str,
    opponent_behavior: HashMap<String, Vec<f64>>,
    v_ref: f64,
    eps_safe: f64,
) -> PyResult<(f64, Vec<f64>)> {
    let data = game_data(game)?;
    let y_fix = data.sf1.realization_from_behavior(&opponent_behavior);
    let r = lp::safety_constrained_best_response_p1(
        &data.sf0,
        &data.sf1,
        &data.payoff,
        &y_fix,
        v_ref,
        eps_safe,
    );
    Ok((r.value, r.realization))
}

#[pyfunction]
fn restricted_nash_response(game: &str, y_fix: Vec<f64>, p: f64) -> PyResult<(f64, Vec<f64>)> {
    let sf1 = compile(game, 1)?;
    if y_fix.len() != sf1.num_sequences() {
        return Err(PyValueError::new_err(format!(
            "y_fix has length {}, expected {}",
            y_fix.len(),
            sf1.num_sequences()
        )));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err("p must be in [0, 1]"));
    }
    let r = lp::restricted_nash_response(&compile(game, 0)?, &sf1, &payoff_of(game)?, &y_fix, p);
    Ok((r.value, r.realization))
}

#[pyfunction]
fn robust_safe_response(
    game: &str,
    intervals: HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
) -> PyResult<(f64, Vec<f64>)> {
    let data = game_data(game)?;
    let cs = confidence::build(&data.sf1, &intervals);
    let sol = lp::robust_safe_response(&data.sf0, &data.sf1, &data.payoff, &cs, v_ref, eps_safe);
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[pyo3(signature = (game, groups, intervals, v_ref, eps_safe, weights = None))]
fn robust_safe_response_public(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    intervals: HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.robust_safe_response_public",
        "start",
        format!(
            "game={game} groups={} interval_keys={}",
            groups.len(),
            intervals.len()
        ),
    );
    let data = game_data(game)?;
    timing_log(
        timer,
        "native.robust_safe_response_public",
        "compile_sf0",
        format!(
            "seq={} infosets={}",
            data.sf0.num_sequences(),
            data.sf0.num_infosets()
        ),
    );
    timing_log(
        timer,
        "native.robust_safe_response_public",
        "compile_sf1",
        format!(
            "seq={} infosets={}",
            data.sf1.num_sequences(),
            data.sf1.num_infosets()
        ),
    );
    timing_log(
        timer,
        "native.robust_safe_response_public",
        "payoff",
        format!(
            "rows={} cols={} nnz={}",
            data.payoff.nrows,
            data.payoff.ncols,
            data.payoff.nnz()
        ),
    );
    let cs = confidence::build_public(&data.sf1, &groups, &intervals, &weights.unwrap_or_default());
    timing_log(
        timer,
        "native.robust_safe_response_public",
        "confidence",
        format!("rows={} nnz={}", cs.nrows, cs.g_entries.len()),
    );
    let sol = lp::robust_safe_response(&data.sf0, &data.sf1, &data.payoff, &cs, v_ref, eps_safe);
    timing_log(
        timer,
        "native.robust_safe_response_public",
        "solve_done",
        format!("robust_value={:.9}", sol.robust_value),
    );
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[pyo3(signature = (game, groups, intervals, v_ref, eps_safe, weights = None, max_iters = 64, tol = 1e-7))]
fn robust_safe_response_public_cutting_plane(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    intervals: HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
    weights: Option<HashMap<String, f64>>,
    max_iters: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.robust_safe_response_public_cutting_plane",
        "start",
        format!(
            "game={game} groups={} interval_keys={} max_iters={max_iters} tol={tol:.3e}",
            groups.len(),
            intervals.len()
        ),
    );
    let data = game_data(game)?;
    let cs = confidence::build_public(&data.sf1, &groups, &intervals, &weights.unwrap_or_default());
    let sol = lp::try_robust_safe_response_cutting_plane(
        &data.sf0,
        &data.sf1,
        &data.payoff,
        &cs,
        v_ref,
        eps_safe,
        max_iters,
        tol,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    timing_log(
        timer,
        "native.robust_safe_response_public_cutting_plane",
        "solve_done",
        format!("robust_value={:.9}", sol.robust_value),
    );
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[pyo3(signature = (game, groups, public_intervals, box_intervals, v_ref, eps_safe, weights = None))]
fn robust_safe_response_envelope(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    public_intervals: HashMap<String, Vec<(f64, f64)>>,
    box_intervals: HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let data = game_data(game)?;
    let c_pub = confidence::build_public(
        &data.sf1,
        &groups,
        &public_intervals,
        &weights.unwrap_or_default(),
    );
    let c_box = confidence::build(&data.sf1, &box_intervals);
    let cs = c_pub.intersect(c_box);
    let sol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lp::robust_safe_response(&data.sf0, &data.sf1, &data.payoff, &cs, v_ref, eps_safe)
    }))
    .map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "envelope LP infeasible: C_pub ∩ B(q, r) is empty (radius below λ_min(q))",
        )
    })?;
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[pyo3(signature = (game, groups, public_intervals, box_intervals, v_ref, eps_safe, weights = None))]
fn robust_safe_response_obs(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    public_intervals: HashMap<String, Vec<(f64, f64)>>,
    box_intervals: HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "start",
        format!("game={game}"),
    );
    let data = game_data(game)?;
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "compile_sf0",
        format!(
            "seq={} infosets={}",
            data.sf0.num_sequences(),
            data.sf0.num_infosets()
        ),
    );
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "compile_sf1",
        format!(
            "seq={} infosets={}",
            data.sf1.num_sequences(),
            data.sf1.num_infosets()
        ),
    );
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "payoff",
        format!(
            "rows={} cols={} nnz={}",
            data.payoff.nrows,
            data.payoff.ncols,
            data.payoff.nnz()
        ),
    );
    let c_pub = confidence::build_public(
        &data.sf1,
        &groups,
        &public_intervals,
        &weights.unwrap_or_default(),
    );
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "confidence_public",
        format!("rows={} nnz={}", c_pub.nrows, c_pub.g_entries.len()),
    );
    let c_box = confidence::build_boxes(&data.sf1, &box_intervals);
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "confidence_box",
        format!("rows={} nnz={}", c_box.nrows, c_box.g_entries.len()),
    );
    let cs = c_pub.intersect(c_box);
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "confidence_intersect",
        format!("rows={} nnz={}", cs.nrows, cs.g_entries.len()),
    );
    let sol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lp::robust_safe_response(&data.sf0, &data.sf1, &data.payoff, &cs, v_ref, eps_safe)
    }))
    .map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "evidence-maximal LP infeasible: C_t^obs is empty (inconsistent estimates)",
        )
    })?;
    timing_log(
        timer,
        "native.robust_safe_response_obs",
        "solve_done",
        format!("robust_value={:.9}", sol.robust_value),
    );
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[pyo3(signature = (game, groups, public_intervals, event_entries, event_h, v_ref, eps_safe, weights = None, row_meta = None))]
fn robust_safe_response_linear(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    public_intervals: HashMap<String, Vec<(f64, f64)>>,
    event_entries: Vec<(usize, usize, f64)>,
    event_h: Vec<f64>,
    v_ref: f64,
    eps_safe: f64,
    weights: Option<HashMap<String, f64>>,
    row_meta: Option<Vec<(String, usize)>>,
) -> PyResult<(f64, Vec<f64>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "start",
        format!(
            "game={game} groups={} interval_keys={} event_rows={} event_nnz={}",
            groups.len(),
            public_intervals.len(),
            event_h.len(),
            event_entries.len()
        ),
    );
    let data = game_data(game)?;
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "compile_sf0",
        format!(
            "seq={} infosets={}",
            data.sf0.num_sequences(),
            data.sf0.num_infosets()
        ),
    );
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "compile_sf1",
        format!(
            "seq={} infosets={}",
            data.sf1.num_sequences(),
            data.sf1.num_infosets()
        ),
    );
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "payoff",
        format!(
            "rows={} cols={} nnz={}",
            data.payoff.nrows,
            data.payoff.ncols,
            data.payoff.nnz()
        ),
    );
    let c_pub = confidence::build_public(
        &data.sf1,
        &groups,
        &public_intervals,
        &weights.unwrap_or_default(),
    );
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "confidence_public",
        format!("rows={} nnz={}", c_pub.nrows, c_pub.g_entries.len()),
    );
    let meta = row_meta.unwrap_or_else(|| {
        (0..event_h.len())
            .map(|row| (format!("event:{row}"), 0usize))
            .collect()
    });
    let c_event = confidence::build_linear(&data.sf1, event_entries, event_h, meta);
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "confidence_event",
        format!("rows={} nnz={}", c_event.nrows, c_event.g_entries.len()),
    );
    let cs = c_pub.intersect(c_event);
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "confidence_intersect",
        format!("rows={} nnz={}", cs.nrows, cs.g_entries.len()),
    );
    let sol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lp::robust_safe_response(&data.sf0, &data.sf1, &data.payoff, &cs, v_ref, eps_safe)
    }))
    .map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "linear-observation LP infeasible: C_pub ∩ C_event is empty",
        )
    })?;
    timing_log(
        timer,
        "native.robust_safe_response_linear",
        "solve_done",
        format!("robust_value={:.9}", sol.robust_value),
    );
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[pyo3(signature = (game, groups, public_intervals, event_entries, event_h, v_ref, eps_safe, weights = None, row_meta = None, max_iters = 64, tol = 1e-7))]
fn robust_safe_response_linear_cutting_plane(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    public_intervals: HashMap<String, Vec<(f64, f64)>>,
    event_entries: Vec<(usize, usize, f64)>,
    event_h: Vec<f64>,
    v_ref: f64,
    eps_safe: f64,
    weights: Option<HashMap<String, f64>>,
    row_meta: Option<Vec<(String, usize)>>,
    max_iters: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.robust_safe_response_linear_cutting_plane",
        "start",
        format!(
            "game={game} groups={} interval_keys={} event_rows={} event_nnz={} max_iters={max_iters} tol={tol:.3e}",
            groups.len(),
            public_intervals.len(),
            event_h.len(),
            event_entries.len()
        ),
    );
    let data = game_data(game)?;
    let c_pub = confidence::build_public(
        &data.sf1,
        &groups,
        &public_intervals,
        &weights.unwrap_or_default(),
    );
    let meta = row_meta.unwrap_or_else(|| {
        (0..event_h.len())
            .map(|row| (format!("event:{row}"), 0usize))
            .collect()
    });
    let c_event = confidence::build_linear(&data.sf1, event_entries, event_h, meta);
    let cs = c_pub.intersect(c_event);
    let sol = lp::try_robust_safe_response_cutting_plane(
        &data.sf0,
        &data.sf1,
        &data.payoff,
        &cs,
        v_ref,
        eps_safe,
        max_iters,
        tol,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    timing_log(
        timer,
        "native.robust_safe_response_linear_cutting_plane",
        "solve_done",
        format!("robust_value={:.9}", sol.robust_value),
    );
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
fn probe_coefficients(
    game: &str,
    opp_behavior: HashMap<String, Vec<f64>>,
    weights: HashMap<String, f64>,
) -> PyResult<Vec<f64>> {
    let sf0 = compile(game, 0)?;
    probe_coeffs_of(game, &sf0, &opp_behavior, &weights)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn robust_safe_response_probe(
    game: &str,
    intervals: HashMap<String, Vec<(f64, f64)>>,
    opp_behavior: HashMap<String, Vec<f64>>,
    weights: HashMap<String, f64>,
    v_ref: f64,
    eps_safe: f64,
    beta: f64,
    rho: f64,
) -> PyResult<(f64, Vec<f64>)> {
    let sf0 = compile(game, 0)?;
    let sf1 = compile(game, 1)?;
    let a = payoff_of(game)?;
    let cs = confidence::build(&sf1, &intervals);
    let coeffs = probe_coeffs_of(game, &sf0, &opp_behavior, &weights)?;
    let sol =
        lp::robust_safe_response_probe(&sf0, &sf1, &a, &cs, v_ref, eps_safe, &coeffs, beta, rho);
    Ok((sol.robust_value, sol.realization))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (game, groups, intervals, y_hat, v_ref, eps_safe, rho, guard_rhs, weights = None))]
fn confidence_guarded_point_probe(
    game: &str,
    groups: HashMap<String, Vec<String>>,
    intervals: HashMap<String, Vec<(f64, f64)>>,
    y_hat: Vec<f64>,
    v_ref: f64,
    eps_safe: f64,
    rho: f64,
    guard_rhs: f64,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let sf0 = compile(game, 0)?;
    let sf1 = compile(game, 1)?;
    let a = payoff_of(game)?;
    let cs = confidence::build_public(&sf1, &groups, &intervals, &weights.unwrap_or_default());
    let sol = lp::confidence_guarded_point_probe(
        &sf0, &sf1, &a, &cs, &y_hat, v_ref, eps_safe, rho, guard_rhs,
    );
    Ok((sol.point_value, sol.realization))
}

#[pyfunction]
fn opponent_reach_weights(game: &str, x_agent: Vec<f64>) -> PyResult<HashMap<String, f64>> {
    let sf0 = compile(game, 0)?;
    Ok(reach_weights_of(game, &sf0, &x_agent)?)
}

#[pyfunction]
fn agent_showdown_reach(
    game: &str,
    x_agent: Vec<f64>,
) -> PyResult<HashMap<String, Vec<(f64, bool)>>> {
    let sf0 = compile(game, 0)?;
    Ok(showdown_reach_of(game, &sf0, &x_agent)?)
}

#[pyfunction]
fn confidence_sensitivity(
    game: &str,
    intervals: HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let sf0 = compile(game, 0)?;
    let sf1 = compile(game, 1)?;
    let a = payoff_of(game)?;
    let cs = confidence::build(&sf1, &intervals);
    let sol = lp::robust_safe_response(&sf0, &sf1, &a, &cs, v_ref, eps_safe);
    let mut importance: HashMap<String, Vec<f64>> = sf1
        .info_sets
        .iter()
        .map(|i| (i.label.clone(), vec![0.0; i.children.len()]))
        .collect();
    for (r, (label, action)) in cs.row_meta.iter().enumerate() {
        if let Some(row) = importance.get_mut(label) {
            row[*action] += sol.confidence_duals[r];
        }
    }
    Ok(importance)
}

#[pyfunction]
fn simulate(
    game: &str,
    strat0: HashMap<String, Vec<f64>>,
    strat1: HashMap<String, Vec<f64>>,
    episodes: u64,
    seed: u64,
) -> PyResult<(f64, HashMap<String, Vec<u64>>)> {
    let r = match game {
        "kuhn" => sim::simulate(&Kuhn, &strat0, &strat1, episodes, seed),
        "leduc" => sim::simulate(&Leduc, &strat0, &strat1, episodes, seed),
        "goofspiel" => sim::simulate(&Goofspiel, &strat0, &strat1, episodes, seed),
        "holdem" => sim::simulate(&canonical_holdem(), &strat0, &strat1, episodes, seed),
        g if g.starts_with("holdem") => sim::simulate(
            &holdem_game(g).ok_or_else(|| unknown_game(g))?,
            &strat0,
            &strat1,
            episodes,
            seed,
        ),
        g if g.starts_with("cchain") => sim::simulate(
            &chain_game(g).ok_or_else(|| unknown_game(g))?,
            &strat0,
            &strat1,
            episodes,
            seed,
        ),
        _ => return Err(unknown_game(game)),
    };
    Ok((r.total_payoff, r.p2_counts.into_iter().collect()))
}

#[pyfunction]
fn simulate_showdown(
    game: &str,
    strat0: HashMap<String, Vec<f64>>,
    strat1: HashMap<String, Vec<f64>>,
    episodes: u64,
    seed: u64,
) -> PyResult<(f64, HashMap<String, Vec<u64>>, HashMap<String, Vec<u64>>)> {
    let timer = timing_start();
    timing_log(
        timer,
        "native.simulate_showdown",
        "start",
        format!("game={game} episodes={episodes} seed={seed}"),
    );
    let r = match game {
        "kuhn" => sim::simulate_showdown(&Kuhn, &strat0, &strat1, episodes, seed),
        "leduc" => sim::simulate_showdown(&Leduc, &strat0, &strat1, episodes, seed),
        "goofspiel" => sim::simulate_showdown(&Goofspiel, &strat0, &strat1, episodes, seed),
        "holdem" => sim::simulate_showdown(&canonical_holdem(), &strat0, &strat1, episodes, seed),
        g if g.starts_with("holdem") => sim::simulate_showdown(
            &holdem_game(g).ok_or_else(|| unknown_game(g))?,
            &strat0,
            &strat1,
            episodes,
            seed,
        ),
        g if g.starts_with("cchain") => sim::simulate_showdown(
            &chain_game(g).ok_or_else(|| unknown_game(g))?,
            &strat0,
            &strat1,
            episodes,
            seed,
        ),
        _ => return Err(unknown_game(game)),
    };
    timing_log(
        timer,
        "native.simulate_showdown",
        "done",
        format!(
            "showdown_infosets={} fold_infosets={} payoff={:.6}",
            r.p2_counts_showdown.len(),
            r.p2_counts_folded.len(),
            r.total_payoff
        ),
    );
    Ok((
        r.total_payoff,
        r.p2_counts_showdown.into_iter().collect(),
        r.p2_counts_folded.into_iter().collect(),
    ))
}

#[pyclass(name = "PayoffMatrix")]
struct PyPayoffMatrix {
    inner: payoff::PayoffMatrix,
}

#[pymethods]
impl PyPayoffMatrix {
    #[new]
    fn new(game: &str) -> PyResult<Self> {
        Ok(Self {
            inner: payoff_of(game)?,
        })
    }

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols
    }

    #[getter]
    fn entries(&self) -> Vec<(usize, usize, f64)> {
        self.inner.entries.clone()
    }

    fn bilinear(&self, x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
        if x.len() != self.inner.nrows || y.len() != self.inner.ncols {
            return Err(PyValueError::new_err(format!(
                "expected x len {} and y len {}, got {} and {}",
                self.inner.nrows,
                self.inner.ncols,
                x.len(),
                y.len()
            )));
        }
        Ok(self.inner.bilinear(&x, &y))
    }

    fn matvec_a_y(&self, y: Vec<f64>) -> PyResult<Vec<f64>> {
        if y.len() != self.inner.ncols {
            return Err(PyValueError::new_err(format!(
                "y has length {}, expected {}",
                y.len(),
                self.inner.ncols
            )));
        }
        Ok(self.inner.matvec_a_y(&y))
    }

    fn matvec_at_x(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        if x.len() != self.inner.nrows {
            return Err(PyValueError::new_err(format!(
                "x has length {}, expected {}",
                x.len(),
                self.inner.nrows
            )));
        }
        Ok(self.inner.matvec_at_x(&x))
    }
}

#[pyclass(name = "ConfidenceSet")]
struct PyConfidenceSet {
    inner: confidence::ConfidenceSet,
}

#[pymethods]
impl PyConfidenceSet {
    #[new]
    fn new(game: &str, intervals: HashMap<String, Vec<(f64, f64)>>) -> PyResult<Self> {
        Ok(Self {
            inner: confidence::build(&compile(game, 1)?, &intervals),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (game, groups, intervals, weights = None))]
    fn from_public(
        game: &str,
        groups: HashMap<String, Vec<String>>,
        intervals: HashMap<String, Vec<(f64, f64)>>,
        weights: Option<HashMap<String, f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: confidence::build_public(
                &compile(game, 1)?,
                &groups,
                &intervals,
                &weights.unwrap_or_default(),
            ),
        })
    }

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols
    }

    #[getter]
    fn g_entries(&self) -> Vec<(usize, usize, f64)> {
        self.inner.g_entries.clone()
    }

    #[getter]
    fn h(&self) -> Vec<f64> {
        self.inner.h.clone()
    }

    fn max_violation(&self, y: Vec<f64>) -> PyResult<f64> {
        if y.len() != self.inner.ncols {
            return Err(PyValueError::new_err(format!(
                "y has length {}, expected {}",
                y.len(),
                self.inner.ncols
            )));
        }
        Ok(self.inner.max_violation(&y))
    }
}

#[pymodule]
fn safe_observation_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Native Rust core for Safe Observation Capacity (heavy compute).",
    )?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(solve_kuhn, m)?)?;
    m.add_function(wrap_pyfunction!(blueprint_mccfr, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_form, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_form_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(blueprint_lp, m)?)?;
    m.add_function(wrap_pyfunction!(blueprint_realization, m)?)?;
    m.add_function(wrap_pyfunction!(safety_verify, m)?)?;
    m.add_function(wrap_pyfunction!(best_response, m)?)?;
    m.add_function(wrap_pyfunction!(safety_constrained_best_response, m)?)?;
    m.add_function(wrap_pyfunction!(restricted_nash_response, m)?)?;
    m.add_function(wrap_pyfunction!(robust_safe_response, m)?)?;
    m.add_function(wrap_pyfunction!(robust_safe_response_public, m)?)?;
    m.add_function(wrap_pyfunction!(
        robust_safe_response_public_cutting_plane,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(robust_safe_response_envelope, m)?)?;
    m.add_function(wrap_pyfunction!(robust_safe_response_obs, m)?)?;
    m.add_function(wrap_pyfunction!(robust_safe_response_linear, m)?)?;
    m.add_function(wrap_pyfunction!(
        robust_safe_response_linear_cutting_plane,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(opponent_reach_weights, m)?)?;
    m.add_function(wrap_pyfunction!(agent_showdown_reach, m)?)?;
    m.add_function(wrap_pyfunction!(probe_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(robust_safe_response_probe, m)?)?;
    m.add_function(wrap_pyfunction!(confidence_guarded_point_probe, m)?)?;
    m.add_function(wrap_pyfunction!(confidence_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_showdown, m)?)?;
    m.add_class::<PyPayoffMatrix>()?;
    m.add_class::<PyConfidenceSet>()?;
    Ok(())
}
