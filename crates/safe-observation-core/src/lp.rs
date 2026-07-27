//! Linear program algorithms for safe observation. See The Safe Observation-Capacity Frontier, Certified Value Recovery, and supplementary Certification at the Unbucketed River.

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use highs::{Col, HighsModelStatus, Model, RowProblem, Sense};

use crate::best_response;
use crate::confidence::{build_kuhn as build_kuhn_confidence, ConfidenceSet};
use crate::payoff::{build_kuhn as build_kuhn_payoff, PayoffMatrix};
use crate::sequence_form::{compile_kuhn, SequenceForm};

/// Stores state for blueprint solution.
pub struct BlueprintSolution {
    pub value: f64,

    pub realization: Vec<f64>,
}

/// Stores state for safety result.
pub struct SafetyResult {
    pub value: f64,

    pub best_response: Vec<f64>,
}

/// Stores state for robust safe solution.
pub struct RobustSafeSolution {
    pub robust_value: f64,

    pub realization: Vec<f64>,

    pub confidence_duals: Vec<f64>,
}

/// Stores state for confidence response result.
pub struct ConfidenceResponseResult {
    pub value: f64,

    pub realization: Vec<f64>,
}

/// Computes require optimal.
pub(crate) fn require_optimal(status: HighsModelStatus, what: &str) {
    assert_eq!(
        status,
        HighsModelStatus::Optimal,
        "{what} LP was not optimal"
    );
}

/// Computes timing enabled.
fn timing_enabled() -> bool {
    std::env::var("SAFE_OBSERVATION_TIMERS")
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(false)
}

/// Computes timing start.
fn timing_start() -> Option<Instant> {
    timing_enabled().then(Instant::now)
}

/// Computes timing log.
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

/// Apply highs options.
pub(crate) fn apply_highs_options(model: &mut Model) {
    model.set_option("output_flag", false);
    model.set_option("log_to_console", false);
    if let Ok(solver) = std::env::var("SAFE_OBSERVATION_HIGHS_SOLVER") {
        if !solver.is_empty() {
            model.set_option("solver", solver.as_str());
        }
    }
    if let Ok(presolve) = std::env::var("SAFE_OBSERVATION_HIGHS_PRESOLVE") {
        if !presolve.is_empty() {
            model.set_option("presolve", presolve.as_str());
        }
    }
    if let Ok(parallel) = std::env::var("SAFE_OBSERVATION_HIGHS_PARALLEL") {
        if !parallel.is_empty() {
            model.set_option("parallel", parallel.as_str());
        }
    }
    if let Ok(threads) = std::env::var("SAFE_OBSERVATION_HIGHS_THREADS") {
        if let Ok(threads) = threads.parse::<i32>() {
            model.set_option("threads", threads);
        }
    }
    if let Ok(time_limit) = std::env::var("SAFE_OBSERVATION_HIGHS_TIME_LIMIT") {
        if let Ok(time_limit) = time_limit.parse::<f64>() {
            model.set_option("time_limit", time_limit);
        }
    }
}

/// Solve blueprint.
pub fn solve_blueprint(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
) -> BlueprintSolution {
    let timer = timing_start();
    let n_x = sf0.num_sequences();
    let n_q = sf1.num_constraints();
    let n_y = sf1.num_sequences();
    let e1 = sf0.constraint_rhs();
    let e2 = sf1.constraint_rhs();
    timing_log(
        timer,
        "lp.solve_blueprint",
        "start",
        format!(
            "n_x={n_x} n_q={n_q} n_y={n_y} payoff_nnz={}",
            payoff.entries.len()
        ),
    );

    let mut pb = RowProblem::default();

    // x is the player-one realization plan; q is the dual certificate for the
    // opponent's sequence-form flow problem.
    let xs: Vec<Col> = (0..n_x).map(|_| pb.add_column(0.0, 0.0..)).collect();
    let qs: Vec<Col> = (0..n_q)
        .map(|i| pb.add_column(e2[i], f64::NEG_INFINITY..))
        .collect();
    timing_log(
        timer,
        "lp.solve_blueprint",
        "columns",
        format!("cols={}", n_x + n_q),
    );

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, k, v) in sf0.constraint_entries() {
        e1_rows[i].push((xs[k], v));
    }
    for (i, row) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }
    timing_log(
        timer,
        "lp.solve_blueprint",
        "e1_rows",
        format!("rows={}", sf0.num_constraints()),
    );

    let mut j_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); n_y];
    // These rows impose E_2^T q <= A^T x, so maximizing e_2^T q computes the
    // maximin blueprint without enumerating opponent pure strategies.
    for (i, j, v) in sf1.constraint_entries() {
        j_rows[j].push((qs[i], v));
    }
    for &(k, j, v) in &payoff.entries {
        j_rows[j].push((xs[k], -v));
    }
    for row in j_rows {
        pb.add_row(..=0.0, &row);
    }
    timing_log(
        timer,
        "lp.solve_blueprint",
        "dual_rows",
        format!("rows={n_y}"),
    );

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    timing_log(timer, "lp.solve_blueprint", "optimise_model", "");
    let solved = model.solve();
    timing_log(
        timer,
        "lp.solve_blueprint",
        "solve_done",
        format!("status={:?}", solved.status()),
    );
    require_optimal(solved.status(), "blueprint");

    let cols = solved.get_solution().columns().to_vec();
    let realization = cols[0..n_x].to_vec();
    let value = (0..n_q).map(|i| e2[i] * cols[n_x + i]).sum();
    timing_log(
        timer,
        "lp.solve_blueprint",
        "extract",
        format!("value={value:.9}"),
    );
    BlueprintSolution { value, realization }
}

/// Computes safety verify.
pub fn safety_verify(sf1: &SequenceForm, payoff: &PayoffMatrix, x: &[f64]) -> SafetyResult {
    let n_y = sf1.num_sequences();
    let e2 = sf1.constraint_rhs();
    let c = payoff.matvec_at_x(x);

    let mut pb = RowProblem::default();
    let ys: Vec<Col> = (0..n_y).map(|j| pb.add_column(c[j], 0.0..)).collect();

    let mut e2_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf1.num_constraints()];
    for (i, j, v) in sf1.constraint_entries() {
        e2_rows[i].push((ys[j], v));
    }
    for (i, row) in e2_rows.into_iter().enumerate() {
        pb.add_row(e2[i]..=e2[i], &row);
    }

    let mut model = pb.optimise(Sense::Minimise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "safety");

    let best_response = solved.get_solution().columns().to_vec();
    let value = (0..n_y).map(|j| c[j] * best_response[j]).sum();
    SafetyResult {
        value,
        best_response,
    }
}

/// Stores state for best response result.
pub struct BestResponseResult {
    pub value: f64,

    pub realization: Vec<f64>,
}

/// Computes best response player-one.
pub fn best_response_p1(
    sf0: &SequenceForm,
    payoff: &PayoffMatrix,
    y: &[f64],
) -> BestResponseResult {
    let n_x = sf0.num_sequences();
    let e1 = sf0.constraint_rhs();
    let c = payoff.matvec_a_y(y);

    let mut pb = RowProblem::default();
    let xs: Vec<Col> = (0..n_x).map(|k| pb.add_column(c[k], 0.0..)).collect();

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, k, v) in sf0.constraint_entries() {
        e1_rows[i].push((xs[k], v));
    }
    for (i, row) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "best_response");

    let realization = solved.get_solution().columns().to_vec();
    let value = (0..n_x).map(|k| c[k] * realization[k]).sum();
    BestResponseResult { value, realization }
}

/// Computes safety constrained best response player-one.
pub fn safety_constrained_best_response_p1(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    y_fix: &[f64],
    v_ref: f64,
    eps_safe: f64,
) -> BestResponseResult {
    let n_x = sf0.num_sequences();
    let n_q2 = sf1.num_constraints();
    let n_y = sf1.num_sequences();
    assert_eq!(
        y_fix.len(),
        n_y,
        "y_fix has length {}, expected {n_y}",
        y_fix.len()
    );
    let e1 = sf0.constraint_rhs();
    let e2 = sf1.constraint_rhs();
    let ay = payoff.matvec_a_y(y_fix);

    let mut pb = RowProblem::default();
    let xs: Vec<Col> = (0..n_x).map(|k| pb.add_column(ay[k], 0.0..)).collect();
    let nus: Vec<Col> = (0..n_q2)
        .map(|_| pb.add_column(0.0, f64::NEG_INFINITY..))
        .collect();

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, k, v) in sf0.constraint_entries() {
        e1_rows[i].push((xs[k], v));
    }
    for (i, row) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }

    let mut safety_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); n_y];
    for (i, j, v) in sf1.constraint_entries() {
        safety_rows[j].push((nus[i], v));
    }
    for &(k, j, v) in &payoff.entries {
        safety_rows[j].push((xs[k], -v));
    }
    for row in safety_rows {
        pb.add_row(..=0.0, &row);
    }

    let floor_row: Vec<(Col, f64)> = (0..n_q2).map(|i| (nus[i], e2[i])).collect();
    pb.add_row((v_ref - eps_safe).., &floor_row);

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "safety_constrained_best_response");
    let cols = solved.get_solution().columns().to_vec();
    let realization = cols[0..n_x].to_vec();
    let value = (0..n_x).map(|k| ay[k] * realization[k]).sum();
    BestResponseResult { value, realization }
}

/// Computes confidence min response.
pub fn confidence_min_response(
    sf1: &SequenceForm,
    confidence: &ConfidenceSet,
    objective: &[f64],
) -> ConfidenceResponseResult {
    let n_y = sf1.num_sequences();
    assert_eq!(
        objective.len(),
        n_y,
        "objective has length {}, expected {n_y}",
        objective.len()
    );
    assert_eq!(
        confidence.ncols, n_y,
        "confidence set has {} columns, expected {n_y}",
        confidence.ncols
    );
    let e2 = sf1.constraint_rhs();

    let mut pb = RowProblem::default();
    let ys: Vec<Col> = (0..n_y)
        .map(|j| pb.add_column(objective[j], 0.0..))
        .collect();

    let mut e2_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf1.num_constraints()];
    for (i, j, v) in sf1.constraint_entries() {
        e2_rows[i].push((ys[j], v));
    }
    for (i, row) in e2_rows.into_iter().enumerate() {
        pb.add_row(e2[i]..=e2[i], &row);
    }

    let mut confidence_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); confidence.nrows];
    for &(row, col, value) in &confidence.g_entries {
        confidence_rows[row].push((ys[col], value));
    }
    for (row, cells) in confidence_rows.into_iter().enumerate() {
        pb.add_row(..=confidence.h[row], &cells);
    }

    let mut model = pb.optimise(Sense::Minimise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "confidence_min_response");
    let realization = solved.get_solution().columns().to_vec();
    let value = (0..n_y).map(|j| objective[j] * realization[j]).sum();
    ConfidenceResponseResult { value, realization }
}

/// Stores state for restricted Nash result.
pub struct RestrictedNashResult {
    pub value: f64,

    pub realization: Vec<f64>,
}

/// Computes restricted Nash response.
pub fn restricted_nash_response(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    y_fix: &[f64],
    p: f64,
) -> RestrictedNashResult {
    let n_x = sf0.num_sequences();
    let n_q2 = sf1.num_constraints();
    let e1 = sf0.constraint_rhs();
    let e2 = sf1.constraint_rhs();
    let c = payoff.matvec_a_y(y_fix);

    let mut pb = RowProblem::default();

    let xs: Vec<Col> = (0..n_x).map(|k| pb.add_column(p * c[k], 0.0..)).collect();
    let nus: Vec<Col> = (0..n_q2)
        .map(|i| pb.add_column((1.0 - p) * e2[i], f64::NEG_INFINITY..))
        .collect();

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, k, v) in sf0.constraint_entries() {
        e1_rows[i].push((xs[k], v));
    }
    for (i, row) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }

    let mut safety_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf1.num_sequences()];
    for (i, j, v) in sf1.constraint_entries() {
        safety_rows[j].push((nus[i], v));
    }
    for &(k, j, v) in &payoff.entries {
        safety_rows[j].push((xs[k], -v));
    }
    for row in safety_rows {
        pb.add_row(..=0.0, &row);
    }

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "restricted_nash");

    let cols = solved.get_solution().columns().to_vec();
    let realization = cols[0..n_x].to_vec();
    let value = p * (0..n_x).map(|k| c[k] * realization[k]).sum::<f64>()
        + (1.0 - p) * (0..n_q2).map(|i| e2[i] * cols[n_x + i]).sum::<f64>();
    RestrictedNashResult { value, realization }
}

/// Solve blueprint Kuhn.
pub fn solve_blueprint_kuhn() -> BlueprintSolution {
    let sf0 = compile_kuhn(0);
    let sf1 = compile_kuhn(1);
    let payoff = build_kuhn_payoff();
    solve_blueprint(&sf0, &sf1, &payoff)
}

/// Computes safety verify Kuhn.
pub fn safety_verify_kuhn(x: &[f64]) -> SafetyResult {
    let sf1 = compile_kuhn(1);
    let payoff = build_kuhn_payoff();
    safety_verify(&sf1, &payoff, x)
}

/// Computes best response player-one Kuhn.
pub fn best_response_p1_kuhn(y: &[f64]) -> BestResponseResult {
    let sf0 = compile_kuhn(0);
    let payoff = build_kuhn_payoff();
    best_response_p1(&sf0, &payoff, y)
}

/// Computes robust safe response.
pub fn robust_safe_response(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    confidence: &ConfidenceSet,
    v_ref: f64,
    eps_safe: f64,
) -> RobustSafeSolution {
    let zero = vec![0.0; sf0.num_sequences()];
    robust_safe_response_inner(
        sf0, sf1, payoff, confidence, v_ref, eps_safe, &zero, 0.0, 0.0,
    )
}

#[allow(clippy::too_many_arguments)]
/// Computes robust safe response probe.
pub fn robust_safe_response_probe(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    confidence: &ConfidenceSet,
    v_ref: f64,
    eps_safe: f64,
    coeffs: &[f64],
    beta: f64,
    rho: f64,
) -> RobustSafeSolution {
    assert_eq!(
        coeffs.len(),
        sf0.num_sequences(),
        "probe coeffs have length {}, expected {}",
        coeffs.len(),
        sf0.num_sequences()
    );
    robust_safe_response_inner(
        sf0, sf1, payoff, confidence, v_ref, eps_safe, coeffs, beta, rho,
    )
}

/// Stores state for cut master solution.
struct CutMasterSolution {
    value_bound: f64,
    realization: Vec<f64>,
}

/// Solve cut master.
fn solve_cut_master(
    sf0: &SequenceForm,
    robust_cuts: &[Vec<f64>],
    safety_cuts: &[Vec<f64>],
    floor: f64,
) -> CutMasterSolution {
    let n_x = sf0.num_sequences();
    let e1 = sf0.constraint_rhs();

    let mut pb = RowProblem::default();
    let xs: Vec<Col> = (0..n_x).map(|_| pb.add_column(0.0, 0.0..)).collect();
    let t_col = pb.add_column(1.0, f64::NEG_INFINITY..);

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (row, col, value) in sf0.constraint_entries() {
        e1_rows[row].push((xs[col], value));
    }
    for (row, cells) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[row]..=e1[row], &cells);
    }

    for cut in robust_cuts {
        assert_eq!(
            cut.len(),
            n_x,
            "robust cut has length {}, expected {n_x}",
            cut.len()
        );
        let mut cells = Vec::with_capacity(1 + n_x);
        cells.push((t_col, 1.0));
        for (col, &value) in cut.iter().enumerate() {
            if value != 0.0 {
                cells.push((xs[col], -value));
            }
        }
        pb.add_row(..=0.0, &cells);
    }

    for cut in safety_cuts {
        assert_eq!(
            cut.len(),
            n_x,
            "safety cut has length {}, expected {n_x}",
            cut.len()
        );
        let cells: Vec<(Col, f64)> = cut
            .iter()
            .enumerate()
            .filter_map(|(col, &value)| (value != 0.0).then_some((xs[col], value)))
            .collect();
        pb.add_row(floor.., &cells);
    }

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "cut_master");
    let cols = solved.get_solution().columns().to_vec();
    CutMasterSolution {
        value_bound: cols[n_x],
        realization: cols[0..n_x].to_vec(),
    }
}

/// Computes try robust safe response cutting plane.
pub fn try_robust_safe_response_cutting_plane(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    confidence: &ConfidenceSet,
    v_ref: f64,
    eps_safe: f64,
    max_iters: usize,
    tol: f64,
) -> Result<RobustSafeSolution, String> {
    let timer = timing_start();
    let n_y = sf1.num_sequences();
    assert_eq!(
        confidence.ncols, n_y,
        "confidence set has {} columns, expected {n_y}",
        confidence.ncols
    );
    let floor = v_ref - eps_safe;
    let max_iters = max_iters.max(1);
    let tol = tol.max(0.0);
    timing_log(
        timer,
        "lp.robust_safe_response_cutting_plane",
        "start",
        format!(
            "n_x={} n_y={n_y} n_g={} max_iters={max_iters} tol={tol:.3e}",
            sf0.num_sequences(),
            confidence.nrows
        ),
    );

    let feasible_confidence = confidence_min_response(sf1, confidence, &vec![0.0; n_y]);
    let mut robust_cuts = vec![payoff.matvec_a_y(&feasible_confidence.realization)];
    let mut safety_cuts: Vec<Vec<f64>> = Vec::new();
    let mut last_master = CutMasterSolution {
        value_bound: f64::NEG_INFINITY,
        realization: vec![0.0; sf0.num_sequences()],
    };
    let mut last_robust = feasible_confidence;

    for iter in 0..max_iters {
        let master = solve_cut_master(sf0, &robust_cuts, &safety_cuts, floor);
        let safety = best_response::safety_verify_from_matrix(sf1, payoff, &master.realization);
        let robust =
            confidence_min_response(sf1, confidence, &payoff.matvec_at_x(&master.realization));
        let mut added_cut = false;

        if safety.value < floor - tol {
            safety_cuts.push(payoff.matvec_a_y(&safety.realization));
            added_cut = true;
        }
        if robust.value < master.value_bound - tol {
            robust_cuts.push(payoff.matvec_a_y(&robust.realization));
            added_cut = true;
        }
        timing_log(
            timer,
            "lp.robust_safe_response_cutting_plane",
            "iter",
            format!(
                "iter={iter} master={:.9} robust={:.9} safety={:.9} robust_cuts={} safety_cuts={} added={added_cut}",
                master.value_bound,
                robust.value,
                safety.value,
                robust_cuts.len(),
                safety_cuts.len()
            ),
        );

        last_master = master;
        last_robust = robust;
        if !added_cut {
            return Ok(RobustSafeSolution {
                robust_value: last_robust.value,
                realization: last_master.realization,
                confidence_duals: vec![0.0; confidence.nrows],
            });
        }
    }

    Err(format!(
        "cutting-plane robust safe LP did not converge in {max_iters} iterations; \
         last master {}, last robust {}",
        last_master.value_bound, last_robust.value
    ))
}

/// Computes robust safe response cutting plane.
pub fn robust_safe_response_cutting_plane(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    confidence: &ConfidenceSet,
    v_ref: f64,
    eps_safe: f64,
    max_iters: usize,
    tol: f64,
) -> RobustSafeSolution {
    try_robust_safe_response_cutting_plane(
        sf0, sf1, payoff, confidence, v_ref, eps_safe, max_iters, tol,
    )
    .unwrap_or_else(|message| panic!("{message}"))
}

#[allow(clippy::too_many_arguments)]
/// Computes robust safe response inner.
fn robust_safe_response_inner(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    confidence: &ConfidenceSet,
    v_ref: f64,
    eps_safe: f64,
    coeffs: &[f64],
    beta: f64,
    rho: f64,
) -> RobustSafeSolution {
    let timer = timing_start();
    let n_x = sf0.num_sequences();
    let n_q2 = sf1.num_constraints();
    let n_y = sf1.num_sequences();
    let n_g = confidence.nrows;
    assert_eq!(
        confidence.ncols, n_y,
        "confidence set has {} columns, expected {n_y}",
        confidence.ncols
    );
    let e1 = sf0.constraint_rhs();
    let e2 = sf1.constraint_rhs();
    timing_log(
        timer,
        "lp.robust_safe_response",
        "start",
        format!(
            "n_x={n_x} n_q2={n_q2} n_y={n_y} n_g={n_g} g_nnz={} payoff_nnz={} eps_safe={eps_safe:.6} rho={rho:.6} beta={beta:.6}",
            confidence.g_entries.len(),
            payoff.entries.len()
        ),
    );

    let mut pb = RowProblem::default();

    let xs: Vec<Col> = (0..n_x)
        .map(|k| pb.add_column(beta * coeffs[k], 0.0..))
        .collect();
    let etas: Vec<Col> = (0..n_q2)
        .map(|i| pb.add_column(e2[i], f64::NEG_INFINITY..))
        .collect();
    let lambdas: Vec<Col> = (0..n_g)
        .map(|r| pb.add_column(-confidence.h[r], 0.0..))
        .collect();
    let nus: Vec<Col> = (0..n_q2)
        .map(|_| pb.add_column(0.0, f64::NEG_INFINITY..))
        .collect();
    timing_log(
        timer,
        "lp.robust_safe_response",
        "columns",
        format!("cols={}", n_x + n_q2 + n_g + n_q2),
    );

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, k, v) in sf0.constraint_entries() {
        e1_rows[i].push((xs[k], v));
    }
    for (i, row) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }
    timing_log(
        timer,
        "lp.robust_safe_response",
        "e1_rows",
        format!("rows={}", sf0.num_constraints()),
    );

    let mut robust_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); n_y];
    for (i, j, v) in sf1.constraint_entries() {
        robust_rows[j].push((etas[i], v));
    }
    for &(r, j, v) in &confidence.g_entries {
        robust_rows[j].push((lambdas[r], -v));
    }
    for &(k, j, v) in &payoff.entries {
        robust_rows[j].push((xs[k], -v));
    }
    for row in robust_rows {
        pb.add_row(..=0.0, &row);
    }
    timing_log(
        timer,
        "lp.robust_safe_response",
        "robust_rows",
        format!("rows={n_y}"),
    );

    let mut safety_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); n_y];
    for (i, j, v) in sf1.constraint_entries() {
        safety_rows[j].push((nus[i], v));
    }
    for &(k, j, v) in &payoff.entries {
        safety_rows[j].push((xs[k], -v));
    }
    for row in safety_rows {
        pb.add_row(..=0.0, &row);
    }
    timing_log(
        timer,
        "lp.robust_safe_response",
        "safety_rows",
        format!("rows={n_y}"),
    );

    let floor_row: Vec<(Col, f64)> = (0..n_q2).map(|i| (nus[i], e2[i])).collect();
    pb.add_row((v_ref - eps_safe - rho).., &floor_row);
    timing_log(timer, "lp.robust_safe_response", "floor_row", "rows=1");

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    timing_log(timer, "lp.robust_safe_response", "optimise_model", "");
    let solved = model.solve();
    timing_log(
        timer,
        "lp.robust_safe_response",
        "solve_done",
        format!("status={:?}", solved.status()),
    );
    assert_eq!(
        solved.status(),
        HighsModelStatus::Optimal,
        "robust safe LP was not optimal (an unbounded result means C_t is empty / \
         the intervals are inconsistent)"
    );

    let cols = solved.get_solution().columns().to_vec();
    let realization = cols[0..n_x].to_vec();
    let robust_value: f64 = (0..n_q2).map(|i| e2[i] * cols[n_x + i]).sum::<f64>()
        - (0..n_g)
            .map(|r| confidence.h[r] * cols[n_x + n_q2 + r])
            .sum::<f64>();

    let confidence_duals: Vec<f64> = (0..n_g).map(|r| cols[n_x + n_q2 + r]).collect();
    timing_log(
        timer,
        "lp.robust_safe_response",
        "extract",
        format!("robust_value={robust_value:.9}"),
    );
    RobustSafeSolution {
        robust_value,
        realization,
        confidence_duals,
    }
}

/// Stores state for guarded solution.
pub struct GuardedSolution {
    pub realization: Vec<f64>,

    pub point_value: f64,
}

#[allow(clippy::too_many_arguments)]
/// Computes confidence guarded point probe.
pub fn confidence_guarded_point_probe(
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    confidence: &ConfidenceSet,
    y_hat: &[f64],
    v_ref: f64,
    eps_safe: f64,
    rho: f64,
    guard_rhs: f64,
) -> GuardedSolution {
    let n_x = sf0.num_sequences();
    let n_q2 = sf1.num_constraints();
    let n_y = sf1.num_sequences();
    let n_g = confidence.nrows;
    assert_eq!(
        confidence.ncols, n_y,
        "confidence set has {} columns, expected {n_y}",
        confidence.ncols
    );
    assert_eq!(
        y_hat.len(),
        n_y,
        "y_hat has length {}, expected {n_y}",
        y_hat.len()
    );
    let e1 = sf0.constraint_rhs();
    let e2 = sf1.constraint_rhs();
    let ay = payoff.matvec_a_y(y_hat);

    let mut pb = RowProblem::default();

    let xs: Vec<Col> = (0..n_x).map(|k| pb.add_column(ay[k], 0.0..)).collect();
    let etas: Vec<Col> = (0..n_q2)
        .map(|_| pb.add_column(0.0, f64::NEG_INFINITY..))
        .collect();
    let lambdas: Vec<Col> = (0..n_g).map(|_| pb.add_column(0.0, 0.0..)).collect();
    let nus: Vec<Col> = (0..n_q2)
        .map(|_| pb.add_column(0.0, f64::NEG_INFINITY..))
        .collect();

    let mut e1_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, k, v) in sf0.constraint_entries() {
        e1_rows[i].push((xs[k], v));
    }
    for (i, row) in e1_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }

    let mut guard_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); n_y];
    for (i, j, v) in sf1.constraint_entries() {
        guard_rows[j].push((etas[i], v));
    }
    for &(r, j, v) in &confidence.g_entries {
        guard_rows[j].push((lambdas[r], -v));
    }
    for &(k, j, v) in &payoff.entries {
        guard_rows[j].push((xs[k], -v));
    }
    for row in guard_rows {
        pb.add_row(..=0.0, &row);
    }

    let mut guard_row: Vec<(Col, f64)> = (0..n_q2).map(|i| (etas[i], e2[i])).collect();
    for (&lambda, &h) in lambdas.iter().zip(&confidence.h) {
        guard_row.push((lambda, -h));
    }
    pb.add_row(guard_rhs.., &guard_row);

    let mut safety_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); n_y];
    for (i, j, v) in sf1.constraint_entries() {
        safety_rows[j].push((nus[i], v));
    }
    for &(k, j, v) in &payoff.entries {
        safety_rows[j].push((xs[k], -v));
    }
    for row in safety_rows {
        pb.add_row(..=0.0, &row);
    }

    let floor_row: Vec<(Col, f64)> = (0..n_q2).map(|i| (nus[i], e2[i])).collect();
    pb.add_row((v_ref - eps_safe - rho).., &floor_row);

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    assert_eq!(
        solved.status(),
        HighsModelStatus::Optimal,
        "confidence-guarded LP was not optimal (guard_rhs above J_t(rho) is infeasible)"
    );
    let cols = solved.get_solution().columns().to_vec();
    let realization = cols[0..n_x].to_vec();
    let point_value: f64 = (0..n_x).map(|k| ay[k] * cols[k]).sum();
    GuardedSolution {
        realization,
        point_value,
    }
}

/// Computes robust safe response Kuhn.
pub fn robust_safe_response_kuhn(
    intervals: &HashMap<String, Vec<(f64, f64)>>,
    v_ref: f64,
    eps_safe: f64,
) -> RobustSafeSolution {
    let sf0 = compile_kuhn(0);
    let sf1 = compile_kuhn(1);
    let payoff = build_kuhn_payoff();
    let confidence = build_kuhn_confidence(intervals);
    robust_safe_response(&sf0, &sf1, &payoff, &confidence, v_ref, eps_safe)
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Defines the Kuhn value constant.
    const KUHN_VALUE: f64 = -1.0 / 18.0;

    #[test]
    /// Verifies that blueprint value matches known Kuhn value.
    fn blueprint_value_matches_known_kuhn_value() {
        let sol = solve_blueprint_kuhn();
        assert!(
            (sol.value - KUHN_VALUE).abs() < 1e-9,
            "value = {} (expected {KUHN_VALUE})",
            sol.value
        );

        let sf0 = compile_kuhn(0);
        assert!(sf0.constraint_residual(&sol.realization) < 1e-9);
    }

    #[test]
    /// Verifies that blueprint is exactly safe against best response.
    fn blueprint_is_exactly_safe_against_best_response() {
        let sol = solve_blueprint_kuhn();
        let safety = safety_verify_kuhn(&sol.realization);
        assert!((safety.value - sol.value).abs() < 1e-9);
        let sf1 = compile_kuhn(1);
        assert!(sf1.constraint_residual(&safety.best_response) < 1e-9);
    }

    #[test]
    /// Verifies that always pass is maximally exploited.
    fn always_pass_is_maximally_exploited() {
        let sf0 = compile_kuhn(0);
        let mut behavior = HashMap::new();
        for info in &sf0.info_sets {
            behavior.insert(info.label.clone(), vec![1.0, 0.0]);
        }
        let x = sf0.realization_from_behavior(&behavior);
        let safety = safety_verify_kuhn(&x);
        assert!(
            (safety.value + 1.0).abs() < 1e-9,
            "value = {} (expected -1)",
            safety.value
        );
    }

    /// Computes always fold intervals.
    fn always_fold_intervals() -> HashMap<String, Vec<(f64, f64)>> {
        let sf1 = compile_kuhn(1);
        let mut m = HashMap::new();
        for info in &sf1.info_sets {
            m.insert(info.label.clone(), vec![(1.0, 1.0), (0.0, 0.0)]);
        }
        m
    }

    #[test]
    /// Verifies that robust value equals game value for full polytope.
    fn robust_value_equals_game_value_for_full_polytope() {
        let sol = robust_safe_response_kuhn(&HashMap::new(), KUHN_VALUE, 0.0);
        assert!(
            (sol.robust_value - KUHN_VALUE).abs() < 1e-9,
            "robust_value = {} (expected {KUHN_VALUE})",
            sol.robust_value
        );
        let sf0 = compile_kuhn(0);
        assert!(sf0.constraint_residual(&sol.realization) < 1e-9);
    }

    #[test]
    /// Verifies that robust response is always safe.
    fn robust_response_is_always_safe() {
        let sol = robust_safe_response_kuhn(&always_fold_intervals(), KUHN_VALUE, 0.0);
        let safety = safety_verify_kuhn(&sol.realization);
        assert!(
            safety.value >= KUHN_VALUE - 1e-9,
            "safety value = {} < v_ref = {KUHN_VALUE}",
            safety.value
        );
    }

    #[test]
    /// Verifies that robust exploits always fold when safety relaxed.
    fn robust_exploits_always_fold_when_safety_relaxed() {
        let sol = robust_safe_response_kuhn(&always_fold_intervals(), KUHN_VALUE, 10.0);
        assert!(
            (sol.robust_value - 1.0).abs() < 1e-9,
            "robust_value = {} (expected 1.0)",
            sol.robust_value
        );
    }

    #[test]
    /// Verifies that safe exploit beats equilibrium but trails unconstrained.
    fn safe_exploit_beats_equilibrium_but_trails_unconstrained() {
        let safe = robust_safe_response_kuhn(&always_fold_intervals(), KUHN_VALUE, 0.0);
        let unconstrained = robust_safe_response_kuhn(&always_fold_intervals(), KUHN_VALUE, 10.0);
        assert!(
            safe.robust_value > KUHN_VALUE + 1e-6,
            "safe robust_value = {} not above game value",
            safe.robust_value
        );
        assert!(safe.robust_value <= unconstrained.robust_value + 1e-9);
    }

    #[test]
    /// Verifies that cutting plane matches monolithic robust Kuhn.
    fn cutting_plane_matches_monolithic_robust_kuhn() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let payoff = build_kuhn_payoff();
        for intervals in [HashMap::new(), always_fold_intervals()] {
            let confidence = build_kuhn_confidence(&intervals);
            let monolithic =
                robust_safe_response(&sf0, &sf1, &payoff, &confidence, KUHN_VALUE, 0.0);
            let cutting = robust_safe_response_cutting_plane(
                &sf0,
                &sf1,
                &payoff,
                &confidence,
                KUHN_VALUE,
                0.0,
                64,
                1e-8,
            );
            assert!(
                (monolithic.robust_value - cutting.robust_value).abs() < 1e-7,
                "monolithic {} cutting {}",
                monolithic.robust_value,
                cutting.robust_value
            );
            let safety = safety_verify_kuhn(&cutting.realization);
            assert!(safety.value >= KUHN_VALUE - 1e-7);
        }
    }

    #[test]
    /// Verifies that smaller scbr matches singleton robust Kuhn.
    fn smaller_scbr_matches_singleton_robust_kuhn() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let payoff = build_kuhn_payoff();
        let y = always_fold_realization();
        let singleton = robust_safe_response(
            &sf0,
            &sf1,
            &payoff,
            &build_kuhn_confidence(&always_fold_intervals()),
            KUHN_VALUE,
            0.0,
        );
        let smaller = safety_constrained_best_response_p1(&sf0, &sf1, &payoff, &y, KUHN_VALUE, 0.0);
        assert!(
            (singleton.robust_value - smaller.value).abs() < 1e-9,
            "singleton {} smaller {}",
            singleton.robust_value,
            smaller.value
        );
        let safety = safety_verify_kuhn(&smaller.realization);
        assert!(safety.value >= KUHN_VALUE - 1e-9);
    }

    #[test]
    /// Verifies that best response lower bounds game value.
    fn best_response_lower_bounds_game_value() {
        let sf1 = compile_kuhn(1);
        let cases = [vec![0.5, 0.5], vec![1.0, 0.0], vec![0.2, 0.8]];
        for probs in cases {
            let behavior: HashMap<String, Vec<f64>> = sf1
                .info_sets
                .iter()
                .map(|info| (info.label.clone(), probs.clone()))
                .collect();
            let y = sf1.realization_from_behavior(&behavior);
            let br = best_response_p1_kuhn(&y);
            assert!(
                br.value >= KUHN_VALUE - 1e-9,
                "BR value {} below game value for probs {probs:?}",
                br.value
            );
            let sf0 = compile_kuhn(0);
            assert!(sf0.constraint_residual(&br.realization) < 1e-9);
        }
    }

    #[test]
    /// Verifies that best response to equilibrium near game value.
    fn best_response_to_equilibrium_near_game_value() {
        let sol = crate::kuhn::solve(100_000);
        let sf1 = compile_kuhn(1);
        let behavior: HashMap<String, Vec<f64>> = sf1
            .info_sets
            .iter()
            .map(|info| {
                let key: String = info.label.split(':').collect();
                (info.label.clone(), sol.strategy[&key].to_vec())
            })
            .collect();
        let y_eq = sf1.realization_from_behavior(&behavior);
        let br = best_response_p1_kuhn(&y_eq);
        assert!(
            (br.value - KUHN_VALUE).abs() < 5e-3,
            "BR value = {} (expected ~ {KUHN_VALUE})",
            br.value
        );
    }

    #[test]
    /// Verifies that best response to always fold is plus one.
    fn best_response_to_always_fold_is_plus_one() {
        let sf1 = compile_kuhn(1);
        let mut behavior = HashMap::new();
        for info in &sf1.info_sets {
            behavior.insert(info.label.clone(), vec![1.0, 0.0]);
        }
        let y = sf1.realization_from_behavior(&behavior);
        let br = best_response_p1_kuhn(&y);
        assert!(
            (br.value - 1.0).abs() < 1e-9,
            "BR value = {} (expected 1.0)",
            br.value
        );
    }

    /// Computes always fold realization.
    fn always_fold_realization() -> Vec<f64> {
        let sf1 = compile_kuhn(1);
        let mut behavior = HashMap::new();
        for info in &sf1.info_sets {
            behavior.insert(info.label.clone(), vec![1.0, 0.0]);
        }
        sf1.realization_from_behavior(&behavior)
    }

    /// Computes restricted Nash response.
    fn rnr(y_fix: &[f64], p: f64) -> RestrictedNashResult {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let payoff = build_kuhn_payoff();
        restricted_nash_response(&sf0, &sf1, &payoff, y_fix, p)
    }

    #[test]
    /// Verifies that restricted Nash response p zero is the blueprint value.
    fn rnr_p_zero_is_the_blueprint_value() {
        let sol = rnr(&always_fold_realization(), 0.0);
        assert!(
            (sol.value - KUHN_VALUE).abs() < 1e-9,
            "RNR(p=0) value = {} (expected {KUHN_VALUE})",
            sol.value
        );
        let sf0 = compile_kuhn(0);
        assert!(sf0.constraint_residual(&sol.realization) < 1e-9);
    }

    #[test]
    /// Verifies that restricted Nash response p one matches best response.
    fn rnr_p_one_matches_best_response() {
        let y = always_fold_realization();
        let sol = rnr(&y, 1.0);
        let br = best_response_p1_kuhn(&y);
        assert!(
            (sol.value - br.value).abs() < 1e-9,
            "RNR(p=1) value = {} but BR = {}",
            sol.value,
            br.value
        );
        assert!((br.value - 1.0).abs() < 1e-9);
    }

    #[test]
    /// Verifies that restricted Nash response exploitation is monotone in p.
    fn rnr_exploitation_is_monotone_in_p() {
        let y = always_fold_realization();
        let payoff = build_kuhn_payoff();
        let mut last = f64::NEG_INFINITY;
        for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let sol = rnr(&y, p);
            let vs_model = payoff.bilinear(&sol.realization, &y);
            assert!(
                vs_model >= last - 1e-9,
                "value vs model dropped at p={p}: {vs_model} < {last}"
            );
            last = vs_model;
        }
    }

    #[test]
    /// Verifies that restricted Nash response safety can drop below game value.
    fn rnr_safety_can_drop_below_game_value() {
        let y = always_fold_realization();
        let aggressive = rnr(&y, 1.0);
        let safety = safety_verify_kuhn(&aggressive.realization);
        assert!(
            safety.value < KUHN_VALUE - 1e-6,
            "expected unsafe BR (safety {} < {KUHN_VALUE})",
            safety.value
        );
    }

    /// Compute coefficients for Kuhn faces bet probe.
    fn kuhn_faces_bet_probe_coeffs() -> Vec<f64> {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let weights: HashMap<String, f64> = sf1
            .info_sets
            .iter()
            .filter(|i| i.label.ends_with(":b"))
            .map(|i| (i.label.clone(), 1.0))
            .collect();
        crate::probe::probe_reach_coeffs(&crate::kuhn::Kuhn, &sf0, &HashMap::new(), &weights)
    }

    /// Solve probe.
    fn solve_probe(
        coeffs: &[f64],
        v_ref: f64,
        eps_safe: f64,
        beta: f64,
        rho: f64,
    ) -> RobustSafeSolution {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let payoff = build_kuhn_payoff();
        let cs = build_kuhn_confidence(&HashMap::new());
        robust_safe_response_probe(&sf0, &sf1, &payoff, &cs, v_ref, eps_safe, coeffs, beta, rho)
    }

    /// Computes info gain.
    fn info_gain(coeffs: &[f64], x: &[f64]) -> f64 {
        coeffs.iter().zip(x).map(|(c, xi)| c * xi).sum()
    }

    #[test]
    /// Verifies that probe linear program delegates to passive when beta and rho zero.
    fn probe_lp_delegates_to_passive_when_beta_and_rho_zero() {
        let coeffs = kuhn_faces_bet_probe_coeffs();
        let passive = robust_safe_response_kuhn(&HashMap::new(), KUHN_VALUE, 0.0);
        let probe = solve_probe(&coeffs, KUHN_VALUE, 0.0, 0.0, 0.0);
        assert!((passive.robust_value - probe.robust_value).abs() < 1e-9);
        for (a, b) in passive.realization.iter().zip(&probe.realization) {
            assert!((a - b).abs() < 1e-9, "realization differs: {a} vs {b}");
        }
    }

    #[test]
    /// Verifies that probe linear program with zero budget stays safe.
    fn probe_lp_with_zero_budget_stays_safe() {
        let coeffs = kuhn_faces_bet_probe_coeffs();
        let probe = solve_probe(&coeffs, KUHN_VALUE, 0.0, 10.0, 0.0);
        let safety = safety_verify_kuhn(&probe.realization);
        assert!(
            safety.value >= KUHN_VALUE - 1e-9,
            "safety value = {} < v_ref = {KUHN_VALUE}",
            safety.value
        );

        assert!((probe.robust_value - KUHN_VALUE).abs() < 1e-9);
    }

    #[test]
    /// Verifies that probe increases information gain with budget.
    fn probe_increases_information_gain_with_budget() {
        let coeffs = kuhn_faces_bet_probe_coeffs();
        let passive = robust_safe_response_kuhn(&HashMap::new(), KUHN_VALUE, 0.0);
        let hard = solve_probe(&coeffs, KUHN_VALUE, 0.0, 10.0, 0.0);
        let budgeted = solve_probe(&coeffs, KUHN_VALUE, 0.0, 10.0, 10.0);

        let ig_passive = info_gain(&coeffs, &passive.realization);
        let ig_hard = info_gain(&coeffs, &hard.realization);
        let ig_budgeted = info_gain(&coeffs, &budgeted.realization);

        assert!(ig_hard >= ig_passive - 1e-9);
        assert!(
            ig_budgeted > ig_hard + 1e-6,
            "budgeted IG {ig_budgeted} not above hard-safe IG {ig_hard}"
        );
    }

    #[test]
    /// Verifies that probe budget can trade safety for information.
    fn probe_budget_can_trade_safety_for_information() {
        let coeffs = kuhn_faces_bet_probe_coeffs();
        let rho = 0.5;
        let probe = solve_probe(&coeffs, KUHN_VALUE, 0.0, 10.0, rho);
        let safety = safety_verify_kuhn(&probe.realization);
        assert!(
            safety.value >= KUHN_VALUE - rho - 1e-9,
            "safety {} below budgeted floor {}",
            safety.value,
            KUHN_VALUE - rho
        );
    }

    #[test]
    /// Verifies that confidence duals are nonnegative with one per row.
    fn confidence_duals_are_nonnegative_with_one_per_row() {
        let sf1 = compile_kuhn(1);
        let mut intervals = HashMap::new();
        for info in &sf1.info_sets {
            intervals.insert(info.label.clone(), vec![(0.3, 0.7), (0.3, 0.7)]);
        }
        let sf0 = compile_kuhn(0);
        let payoff = build_kuhn_payoff();
        let cs = build_kuhn_confidence(&intervals);
        let sol = robust_safe_response(&sf0, &sf1, &payoff, &cs, KUHN_VALUE, 0.0);
        assert_eq!(sol.confidence_duals.len(), cs.nrows);
        assert!(cs.nrows > 0);
        assert!(sol.confidence_duals.iter().all(|&d| d >= -1e-9));

        assert!(sol.confidence_duals.iter().any(|&d| d > 1e-9));
    }

    /// Computes always fold behavior.
    fn always_fold_behavior() -> HashMap<String, Vec<f64>> {
        let sf1 = compile_kuhn(1);
        let mut m = HashMap::new();
        for info in &sf1.info_sets {
            m.insert(info.label.clone(), vec![1.0, 0.0]);
        }
        m
    }

    /// Computes guarded Kuhn.
    fn guarded_kuhn(
        intervals: &HashMap<String, Vec<(f64, f64)>>,
        y_hat_behavior: &HashMap<String, Vec<f64>>,
        v_ref: f64,
        eps_safe: f64,
        rho: f64,
        guard_rhs: f64,
    ) -> GuardedSolution {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let payoff = build_kuhn_payoff();
        let confidence = build_kuhn_confidence(intervals);
        let y_hat = sf1.realization_from_behavior(y_hat_behavior);
        confidence_guarded_point_probe(
            &sf0,
            &sf1,
            &payoff,
            &confidence,
            &y_hat,
            v_ref,
            eps_safe,
            rho,
            guard_rhs,
        )
    }

    #[test]
    /// Verifies that guard with huge slack recovers point best response.
    fn guard_with_huge_slack_recovers_point_best_response() {
        let sol = guarded_kuhn(
            &HashMap::new(),
            &always_fold_behavior(),
            KUHN_VALUE,
            10.0,
            0.0,
            f64::NEG_INFINITY,
        );
        assert!(
            (sol.point_value - 1.0).abs() < 1e-9,
            "point_value = {} (expected 1.0)",
            sol.point_value
        );
    }

    #[test]
    /// Verifies that guard with honest wide set kills phantom and stays maximin safe.
    fn guard_with_honest_wide_set_kills_phantom_and_stays_maximin_safe() {
        let guarded = guarded_kuhn(
            &HashMap::new(),
            &always_fold_behavior(),
            KUHN_VALUE,
            10.0,
            0.0,
            KUHN_VALUE,
        );

        let safety = safety_verify_kuhn(&guarded.realization);
        assert!(
            safety.value >= KUHN_VALUE - 1e-9,
            "guarded plan not maximin-safe: {} < {KUHN_VALUE}",
            safety.value
        );

        assert!(
            guarded.point_value < 1.0 - 1e-6,
            "guard did not bind: point_value = {}",
            guarded.point_value
        );
    }

    #[test]
    /// Verifies that guard linear program respects the hard floor.
    fn guard_lp_respects_the_hard_floor() {
        let rho = 0.25;
        let sol = guarded_kuhn(
            &always_fold_intervals(),
            &always_fold_behavior(),
            KUHN_VALUE,
            0.0,
            rho,
            f64::NEG_INFINITY,
        );
        let safety = safety_verify_kuhn(&sol.realization);
        assert!(
            safety.value >= KUHN_VALUE - rho - 1e-9,
            "safety {} below budgeted floor {}",
            safety.value,
            KUHN_VALUE - rho
        );
    }
}
