//! Robust cuts algorithms for safe observation. See The Safe Observation-Capacity Frontier, Certified Value Recovery, and supplementary Certification at the Unbucketed River.

use crate::best_response::treeplex_opt;
use crate::confidence::ConfidenceSet;
use crate::lp::{apply_highs_options, require_optimal};
use crate::payoff_oracle::PayoffOracle;
use crate::sequence_form::SequenceForm;
use highs::{Col, RowProblem, Sense};

/// Stores state for cut params.
pub struct CutParams {
    pub max_iters: usize,

    pub tol: f64,

    pub in_out_alpha: f64,

    pub cut_window: usize,

    pub max_wall_s: f64,
}

/// Implements operations for `CutParams`.
impl Default for CutParams {
    /// Returns the default configuration.
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol: 1e-5,
            in_out_alpha: 0.5,
            cut_window: 80,
            max_wall_s: f64::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Stores state for cut trace row.
pub struct CutTraceRow {
    pub iter: usize,
    pub wall_s: f64,
    pub master_ub: f64,
    pub best_lb: f64,
}

/// Stores state for cut solution.
pub struct CutSolution {
    pub realization: Vec<f64>,

    pub certified_value: f64,

    pub master_bound: f64,

    pub floor_value: f64,

    pub iters: usize,

    pub converged: bool,

    pub trace: Vec<CutTraceRow>,
}

/// Computes inner min linear program.
pub fn inner_min_lp(sf1: &SequenceForm, conf: &ConfidenceSet, c: &[f64]) -> (f64, Vec<f64>) {
    let n_y = sf1.num_sequences();
    assert_eq!(
        conf.ncols, n_y,
        "confidence set over a different sequence form"
    );
    let e2 = sf1.constraint_rhs();

    let build = || {
        let mut pb = RowProblem::default();
        let ys: Vec<Col> = (0..n_y).map(|j| pb.add_column(c[j], 0.0..)).collect();
        let mut e2_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf1.num_constraints()];
        for (i, j, v) in sf1.constraint_entries() {
            e2_rows[i].push((ys[j], v));
        }
        for (i, row) in e2_rows.into_iter().enumerate() {
            pb.add_row(e2[i]..=e2[i], &row);
        }
        let mut g_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); conf.nrows];
        for &(r, cix, v) in &conf.g_entries {
            g_rows[r].push((ys[cix], v));
        }
        for (r, row) in g_rows.into_iter().enumerate() {
            pb.add_row(..=conf.h[r], &row);
        }
        pb
    };

    let mut solved = None;
    // Retry numerically difficult adversary problems with progressively more
    // conservative HiGHS configurations.
    for rung in 0..3 {
        let mut model = build().optimise(Sense::Minimise);
        apply_highs_options(&mut model);
        if rung >= 1 {
            model.set_option("presolve", "off");
        }
        if rung >= 2 {
            model.set_option("solver", "ipm");
        }
        let s = model.solve();
        if s.status() == highs::HighsModelStatus::Optimal {
            solved = Some(s);
            break;
        }
    }
    let solved = solved.unwrap_or_else(|| {
        panic!("robust inner adversary LP failed all retry rungs (simplex, no-presolve, ipm)")
    });
    let y = solved.get_solution().columns().to_vec();
    let value = c.iter().zip(&y).map(|(a, b)| a * b).sum();
    (value, y)
}

/// Computes repair to floor.
pub fn repair_to_floor<O: PayoffOracle>(
    oracle: &O,
    sf1: &SequenceForm,
    x: &[f64],
    x_bp: &[f64],
    v_target: f64,
) -> (Vec<f64>, f64, f64) {
    let w_x = treeplex_opt(sf1, oracle.at_x(x), false).value;
    if w_x >= v_target {
        return (x.to_vec(), 0.0, w_x);
    }
    let w_bp = treeplex_opt(sf1, oracle.at_x(x_bp), false).value;
    assert!(
        w_bp >= v_target,
        "blueprint violates the floor target ({w_bp} < {v_target}): cannot repair"
    );
    // Concavity of worst-case value makes this blueprint mixture floor-safe.
    let alpha = ((v_target - w_x) / (w_bp - w_x)).clamp(0.0, 1.0);
    let mixed: Vec<f64> = x
        .iter()
        .zip(x_bp)
        .map(|(a, b)| (1.0 - alpha) * a + alpha * b)
        .collect();
    let w_mixed = treeplex_opt(sf1, oracle.at_x(&mixed), false).value;
    debug_assert!(
        w_mixed >= v_target - 1e-9,
        "concavity repair failed: {w_mixed}"
    );
    (mixed, alpha, w_mixed)
}

/// Computes robust response cuts.
pub fn robust_response_cuts<O: PayoffOracle>(
    oracle: &O,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    conf: &ConfidenceSet,
    v_target: f64,
    x_init: &[f64],
    params: &CutParams,
) -> CutSolution {
    let mut value_cuts: Vec<Vec<f64>> = Vec::new();
    let mut floor_cuts: Vec<Vec<f64>> = Vec::new();

    let c0 = oracle.at_x(x_init);
    let floor0 = treeplex_opt(sf1, c0.clone(), false).value;
    assert!(
        floor0 >= v_target - 1e-9,
        "x_init violates the floor target ({floor0} < {v_target})"
    );
    let (lb0, y0) = inner_min_lp(sf1, conf, &c0);
    let mut incumbent = x_init.to_vec();
    let mut best_lb = lb0;
    let mut best_floor = floor0;
    value_cuts.push(oracle.a_y(&y0));

    let mut alpha = params.in_out_alpha;
    let mut last_ub = f64::INFINITY;
    let mut stall = 0usize;
    let mut ub = f64::INFINITY;
    let mut iters = 0;
    let mut converged = false;
    let mut trace = Vec::with_capacity(params.max_iters.min(1024));
    let t0 = std::time::Instant::now();

    while iters < params.max_iters {
        iters += 1;
        let (master_ub, x_out) = solve_master(sf0, &value_cuts, &floor_cuts, v_target);
        ub = master_ub;
        trace.push(CutTraceRow {
            iter: iters,
            wall_s: t0.elapsed().as_secs_f64(),
            master_ub: ub,
            best_lb,
        });
        if ub - best_lb <= params.tol * ub.abs().max(1.0) {
            converged = true;
            break;
        }

        if last_ub - ub < 1e-9 {
            stall += 1;
            if stall >= 5 {
                alpha = (alpha + 1.0) / 2.0;
                stall = 0;
            }
        } else {
            stall = 0;
        }
        last_ub = ub;

        // The in-out point stabilizes separation while retaining the current
        // feasible incumbent as an explicit lower bound.
        let x_q: Vec<f64> = incumbent
            .iter()
            .zip(&x_out)
            .map(|(xin, xout)| (1.0 - alpha) * xin + alpha * xout)
            .collect();
        let c_q = oracle.at_x(&x_q);

        let fl = treeplex_opt(sf1, c_q.clone(), false);
        let feasible = fl.value >= v_target - 1e-12;
        if !feasible {
            floor_cuts.push(oracle.a_y(&fl.realization));
        }

        let (rv, y_min) = inner_min_lp(sf1, conf, &c_q);
        value_cuts.push(oracle.a_y(&y_min));
        // A bounded cut window limits master growth on full-deck river games.
        if value_cuts.len() > params.cut_window {
            let excess = value_cuts.len() - params.cut_window;
            value_cuts.drain(..excess);
        }
        if feasible && rv > best_lb {
            best_lb = rv;
            best_floor = fl.value;
            incumbent = x_q;
        }
        if t0.elapsed().as_secs_f64() >= params.max_wall_s {
            break;
        }
    }

    CutSolution {
        realization: incumbent,
        certified_value: best_lb,
        master_bound: ub,
        floor_value: best_floor,
        iters,
        converged,
        trace,
    }
}

/// Computes linear objective cuts.
pub fn linear_objective_cuts<O: PayoffOracle>(
    oracle: &O,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    c: &[f64],
    v_target: f64,
    x_init: &[f64],
    params: &CutParams,
) -> CutSolution {
    let dot = |x: &[f64]| -> f64 { c.iter().zip(x).map(|(a, b)| a * b).sum() };
    let floor_of = |x: &[f64]| -> crate::best_response::TreeBestResponse {
        treeplex_opt(sf1, oracle.at_x(x), false)
    };
    let f0 = floor_of(x_init);
    assert!(
        f0.value >= v_target - 1e-9,
        "x_init violates the floor target ({} < {v_target})",
        f0.value
    );
    let mut incumbent = x_init.to_vec();
    let mut best = dot(&incumbent);
    let mut best_floor = f0.value;
    let mut floor_cuts: Vec<Vec<f64>> = Vec::new();
    let mut ub = f64::INFINITY;
    let mut iters = 0;
    let mut converged = false;
    let mut trace = Vec::new();
    let t0 = std::time::Instant::now();

    while iters < params.max_iters {
        iters += 1;
        let (master_ub, x_out) = solve_master_linear(sf0, c, &floor_cuts, v_target);
        ub = master_ub;
        trace.push(CutTraceRow {
            iter: iters,
            wall_s: t0.elapsed().as_secs_f64(),
            master_ub: ub,
            best_lb: best,
        });
        if ub - best <= params.tol * ub.abs().max(1.0) {
            converged = true;
            break;
        }

        let x_q: Vec<f64> = incumbent
            .iter()
            .zip(&x_out)
            .map(|(xin, xout)| (1.0 - params.in_out_alpha) * xin + params.in_out_alpha * xout)
            .collect();
        let fl = floor_of(&x_q);
        if fl.value >= v_target - 1e-12 {
            let v = dot(&x_q);
            if v > best {
                best = v;
                best_floor = fl.value;
                incumbent = x_q;
            }

            let fl_out = floor_of(&x_out);
            if fl_out.value >= v_target - 1e-12 {
                let v = dot(&x_out);
                if v > best {
                    best = v;
                    best_floor = fl_out.value;
                    incumbent = x_out;
                }
            } else {
                floor_cuts.push(oracle.a_y(&fl_out.realization));
            }
        } else {
            floor_cuts.push(oracle.a_y(&fl.realization));
        }
        if t0.elapsed().as_secs_f64() > params.max_wall_s {
            break;
        }
    }

    CutSolution {
        realization: incumbent,
        certified_value: best,
        master_bound: ub,
        floor_value: best_floor,
        iters,
        converged,
        trace,
    }
}

/// Solve master linear.
fn solve_master_linear(
    sf0: &SequenceForm,
    c: &[f64],
    floor_cuts: &[Vec<f64>],
    v_target: f64,
) -> (f64, Vec<f64>) {
    let n_x = sf0.num_sequences();
    let e1 = sf0.constraint_rhs();

    let mut pb = RowProblem::default();
    let xs: Vec<Col> = (0..n_x).map(|j| pb.add_column(c[j], 0.0..)).collect();
    let mut e_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, j, v) in sf0.constraint_entries() {
        e_rows[i].push((xs[j], v));
    }
    for (i, row) in e_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }
    for cut in floor_cuts {
        let row: Vec<(Col, f64)> = cut
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(j, v)| (xs[j], *v))
            .collect();
        pb.add_row(v_target.., &row);
    }
    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "linear-objective cut master");
    let x = solved.get_solution().columns().to_vec();
    let value = c.iter().zip(&x).map(|(a, b)| a * b).sum();
    (value, x)
}

/// Solve master.
fn solve_master(
    sf0: &SequenceForm,
    value_cuts: &[Vec<f64>],
    floor_cuts: &[Vec<f64>],
    v_target: f64,
) -> (f64, Vec<f64>) {
    let n_x = sf0.num_sequences();
    let e1 = sf0.constraint_rhs();

    let mut pb = RowProblem::default();
    let xs: Vec<Col> = (0..n_x).map(|_| pb.add_column(0.0, 0.0..)).collect();
    let t = pb.add_column(1.0, f64::NEG_INFINITY..);

    let mut e_rows: Vec<Vec<(Col, f64)>> = vec![Vec::new(); sf0.num_constraints()];
    for (i, j, v) in sf0.constraint_entries() {
        e_rows[i].push((xs[j], v));
    }
    for (i, row) in e_rows.into_iter().enumerate() {
        pb.add_row(e1[i]..=e1[i], &row);
    }
    for cut in value_cuts {
        let mut row: Vec<(Col, f64)> = cut
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(j, v)| (xs[j], -*v))
            .collect();
        row.push((t, 1.0));
        pb.add_row(..=0.0, &row);
    }
    for cut in floor_cuts {
        let row: Vec<(Col, f64)> = cut
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(j, v)| (xs[j], *v))
            .collect();
        pb.add_row(v_target.., &row);
    }

    let mut model = pb.optimise(Sense::Maximise);
    apply_highs_options(&mut model);
    let solved = model.solve();
    require_optimal(solved.status(), "robust cut master");
    let cols = solved.get_solution().columns().to_vec();
    let (x, tval) = cols.split_at(n_x);
    (tval[0], x.to_vec())
}

#[cfg(test)]
/// Provides shared fixtures for this module's regression tests.
pub(crate) mod test_util {
    use super::*;
    use std::collections::HashMap;

    /// Construct confidence constraints for box.
    pub fn box_confidence(
        sf1: &SequenceForm,
        b1: &HashMap<String, Vec<f64>>,
        w: f64,
    ) -> ConfidenceSet {
        let mut intervals: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        for info in &sf1.info_sets {
            let n = info.children.len();
            let dist = b1
                .get(&info.label)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / n as f64; n]);
            intervals.insert(
                info.label.clone(),
                dist.iter()
                    .map(|p| ((p - w).max(0.0), (p + w).min(1.0)))
                    .collect(),
            );
        }
        crate::confidence::build(sf1, &intervals)
    }
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::test_util::box_confidence;
    use super::*;
    use crate::holdem::{build_holdem, canonical_holdem, compile_holdem};
    use crate::payoff_oracle::RangeOracle;
    use crate::river_range::test_util::random_behavior;
    use crate::river_range::RangeGame;

    #[test]
    /// Verifies that cutting plane matches exact robust linear program on compact river.
    fn cutting_plane_matches_exact_robust_lp_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = compile_holdem(0);
        let sf1 = compile_holdem(1);
        let pm = build_holdem();

        let bp = crate::lp::solve_blueprint(&sf0, &sf1, &pm);
        let rho = 0.5;
        let v_target = bp.value - rho;
        let conf = box_confidence(&sf1, &random_behavior(&rg, 1, 77), 0.2);

        let exact = crate::lp::robust_safe_response(&sf0, &sf1, &pm, &conf, bp.value, rho);

        let params = CutParams::default();
        let range_oracle = RangeOracle::new(&rg, &sf0, &sf1);
        for (name, sol) in [
            (
                "matrix",
                robust_response_cuts(&pm, &sf0, &sf1, &conf, v_target, &bp.realization, &params),
            ),
            (
                "range",
                robust_response_cuts(
                    &range_oracle,
                    &sf0,
                    &sf1,
                    &conf,
                    v_target,
                    &bp.realization,
                    &params,
                ),
            ),
        ] {
            assert!(
                sol.converged,
                "{name}: no convergence in {} iters",
                sol.iters
            );
            assert!(
                (sol.certified_value - exact.robust_value).abs() <= 1e-5,
                "{name}: certified {} vs exact {} ({} iters)",
                sol.certified_value,
                exact.robust_value,
                sol.iters
            );
            assert!(
                sol.floor_value >= v_target - 1e-8,
                "{name}: floor {} < target {v_target}",
                sol.floor_value
            );
            assert!(
                sol.iters <= 120,
                "{name}: too many iterations {}",
                sol.iters
            );
        }
    }

    #[test]
    #[ignore = "full river robust-solver timing; run explicitly in release mode"]
    /// Verifies that full river smoke robust solvers.
    fn full_river_smoke_robust_solvers() {
        use crate::hand_eval::card;
        use crate::holdem::{HoldemRules, RiverEndgame};
        use crate::river_solve::{RangeCfr, Variant};
        use crate::robust_lagrangian::{robust_response_lagrangian, LagrangianParams};
        use std::time::Instant;

        let board = [
            card(12, 3),
            card(11, 3),
            card(10, 1),
            card(9, 0),
            card(7, 2),
        ];
        let game = RiverEndgame::full(HoldemRules::river_small(), board);
        let rg = RangeGame::new(&game);
        let t0 = Instant::now();
        let sf0 = rg.compile_sequence_form(0);
        let sf1 = rg.compile_sequence_form(1);
        let oracle = RangeOracle::new(&rg, &sf0, &sf1);
        println!(
            "full river setup (sf compile + oracle map): {:?}",
            t0.elapsed()
        );

        let t0 = Instant::now();
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..2000 {
            cfr.iterate();
        }
        let x_bp = sf0.realization_from_behavior(&cfr.average_behavior(0));
        let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
        println!(
            "full river blueprint: {:?}, exact v_ref {v_ref:.6}",
            t0.elapsed()
        );

        let rho = 0.5;
        let v_target = v_ref - rho;
        let conf = box_confidence(&sf1, &cfr.average_behavior(1), 0.1);
        println!("full river confidence rows: {}", conf.nrows);

        let t0 = Instant::now();
        let params = CutParams {
            max_iters: 150,
            ..CutParams::default()
        };
        let cp = robust_response_cuts(&oracle, &sf0, &sf1, &conf, v_target, &x_bp, &params);
        println!(
            "full river cutting-plane: {:?}  iters {}  certified {:.6}  master bound {:.6}  floor {:.6}  converged {}",
            t0.elapsed(),
            cp.iters,
            cp.certified_value,
            cp.master_bound,
            cp.floor_value,
            cp.converged,
        );
        for th in [1e-2, 1e-3, 1e-4, 1e-5] {
            if let Some(row) = cp.trace.iter().find(|r| r.master_ub - r.best_lb <= th) {
                println!(
                    "  gap<{th:.0e} first at iter {} (wall {:.1}s)",
                    row.iter, row.wall_s
                );
            }
        }

        let t0 = Instant::now();
        let lag = robust_response_lagrangian(
            &oracle,
            &sf0,
            &sf1,
            &conf,
            v_target,
            &LagrangianParams {
                iters: 10_000,
                check_every: 1_000,
                ..LagrangianParams::default()
            },
        );
        let (fixed, alpha, w) = repair_to_floor(&oracle, &sf1, &lag.realization, &x_bp, v_target);
        let (certified, _) = inner_min_lp(&sf1, &conf, &oracle.at_x(&fixed));
        println!(
            "full river Lagrangian: {:?}  anytime bound {:.6}  repaired(alpha {alpha:.4}) floor {w:.6}  exact certified {certified:.6}",
            t0.elapsed(),
            lag.anytime_bound,
        );
        assert!(cp.floor_value >= v_target - 1e-8);
        assert!(w >= v_target - 1e-9);
    }

    /// Computes robust solver benchmark.
    fn robust_solver_benchmark(suffix: &str, shape: &str) {
        use crate::holdem::turn_river_game;
        use crate::payoff::build;
        use crate::sequence_form::compile;
        use std::collections::HashMap;
        use std::time::Instant;

        std::env::set_var("SAFE_OBSERVATION_HIGHS_TIME_LIMIT", "600");
        std::env::set_var("SAFE_OBSERVATION_HIGHS_THREADS", "1");

        let game = turn_river_game(suffix).expect("known suffix");
        let sf0 = compile(&game, 0);
        let sf1 = compile(&game, 1);
        let pm = build(&game, &sf0, &sf1);
        let bp = crate::lp::solve_blueprint(&sf0, &sf1, &pm);
        let rho = 0.5;
        let v_target = bp.value - rho;

        let conf = match shape {
            "dense" => box_confidence(&sf1, &HashMap::new(), 0.2),
            "sparse" => {
                let mut some: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
                for (i, info) in sf1.info_sets.iter().enumerate() {
                    if i % 8 == 0 {
                        let n = info.children.len();
                        let u = 1.0 / n as f64;
                        some.insert(
                            info.label.clone(),
                            vec![((u - 0.2_f64).max(0.0), (u + 0.2_f64).min(1.0)); n],
                        );
                    }
                }
                crate::confidence::build(&sf1, &some)
            }
            _ => unreachable!(),
        };

        let t0 = Instant::now();
        let exact = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::lp::robust_safe_response(&sf0, &sf1, &pm, &conf, bp.value, rho)
        }))
        .ok();
        let lp_wall = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let cp = robust_response_cuts(
            &pm,
            &sf0,
            &sf1,
            &conf,
            v_target,
            &bp.realization,
            &CutParams::default(),
        );
        let cp_wall = t0.elapsed().as_secs_f64();

        let lp_value = exact.as_ref().map(|e| e.robust_value);
        println!(
            "holdem_tr{suffix}/{shape} ({} rows): LP {} in {lp_wall:.1}s | CP {:.6} (bound {:.6}, gap {:.1e}) in {cp_wall:.1}s ({} iters, converged {})",
            conf.nrows,
            lp_value.map_or("TIMEOUT/FAIL".into(), |v| format!("{v:.6}")),
            cp.certified_value,
            cp.master_bound,
            cp.master_bound - cp.certified_value,
            cp.iters,
            cp.converged,
        );
        let json = format!(
            "{{\"game\": \"holdem_tr{suffix}\", \"shape\": \"{shape}\", \"rows\": {}, \
             \"v_ref\": {:.9}, \"rho\": {rho}, \"lp_value\": {}, \"lp_wall_s\": {lp_wall:.3}, \
             \"cp_value\": {:.9}, \"cp_bound\": {:.9}, \"cp_floor\": {:.9}, \
             \"cp_wall_s\": {cp_wall:.3}, \"cp_iters\": {}, \"cp_converged\": {}}}\n",
            conf.nrows,
            bp.value,
            lp_value.map_or("null".into(), |v| format!("{v:.9}")),
            cp.certified_value,
            cp.master_bound,
            cp.floor_value,
            cp.iters,
            cp.converged,
        );
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../results");
        std::fs::write(
            format!("{path}/robust_solver_holdem_tr{suffix}_{shape}.json"),
            json,
        )
        .expect("write checkpoint");

        assert!(cp.floor_value >= v_target - 1e-8);
        if let Some(v) = lp_value {
            assert!(
                cp.certified_value <= v + 1e-6 && v <= cp.master_bound + 1e-6,
                "{suffix}/{shape}: LP {v} outside CP bracket [{}, {}]",
                cp.certified_value,
                cp.master_bound
            );
            if cp.converged {
                assert!(
                    (cp.certified_value - v).abs() <= 1e-4,
                    "{suffix}/{shape}: converged CP {} vs LP {v}",
                    cp.certified_value
                );
            }
        }
    }

    #[test]
    #[ignore = "robust solver benchmark; run explicitly in release mode"]
    /// Verifies that robust solver b2 dense.
    fn robust_solver_b2_dense() {
        robust_solver_benchmark("_b2", "dense");
    }
    #[test]
    #[ignore = "robust solver benchmark; run explicitly in release mode"]
    /// Verifies that robust solver b2 sparse.
    fn robust_solver_b2_sparse() {
        robust_solver_benchmark("_b2", "sparse");
    }
    #[test]
    #[ignore = "robust solver benchmark; run explicitly in release mode"]
    /// Verifies that robust solver b4 dense.
    fn robust_solver_b4_dense() {
        robust_solver_benchmark("_b4", "dense");
    }
    #[test]
    #[ignore = "robust solver benchmark; run explicitly in release mode"]
    /// Verifies that robust solver b4 sparse.
    fn robust_solver_b4_sparse() {
        robust_solver_benchmark("_b4", "sparse");
    }

    #[test]
    /// Verifies that linear cuts match safety constrained br on compact river.
    fn linear_cuts_match_safety_constrained_br_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = compile_holdem(0);
        let sf1 = compile_holdem(1);
        let pm = build_holdem();
        let bp = crate::lp::solve_blueprint(&sf0, &sf1, &pm);
        let rho = 0.5;
        let v_target = bp.value - rho;
        let y = sf1.realization_from_behavior(&random_behavior(&rg, 1, 400));
        let exact =
            crate::lp::safety_constrained_best_response_p1(&sf0, &sf1, &pm, &y, bp.value, rho);
        let c = pm.matvec_a_y(&y);
        let sol = linear_objective_cuts(
            &pm,
            &sf0,
            &sf1,
            &c,
            v_target,
            &bp.realization,
            &CutParams::default(),
        );
        assert!(sol.converged, "no convergence in {} iters", sol.iters);
        assert!(
            (sol.certified_value - exact.value).abs() <= 1e-5,
            "linear cuts {} vs exact LP {}",
            sol.certified_value,
            exact.value
        );
        assert!(sol.floor_value >= v_target - 1e-8);
    }

    #[test]
    /// Verifies that repair restores floor on compact river.
    fn repair_restores_floor_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = compile_holdem(0);
        let sf1 = compile_holdem(1);
        let pm = build_holdem();
        let bp = crate::lp::solve_blueprint(&sf0, &sf1, &pm);

        let y = sf1.realization_from_behavior(&random_behavior(&rg, 1, 5));
        let aggressive =
            crate::best_response::best_response_p1_from_matrix(&sf0, &pm, &y).realization;
        let v_target = bp.value - 0.1;
        let (fixed, alpha, w) = repair_to_floor(&pm, &sf1, &aggressive, &bp.realization, v_target);
        assert!(w >= v_target - 1e-9, "repaired floor {w} < {v_target}");
        assert!((0.0..=1.0).contains(&alpha));
        assert_eq!(fixed.len(), aggressive.len());
    }
}
