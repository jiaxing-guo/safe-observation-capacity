//! Robust lagrangian algorithms for safe observation. See The Safe Observation-Capacity Frontier, Certified Value Recovery, and supplementary Certification at the Unbucketed River.

use crate::best_response::treeplex_opt;
use crate::confidence::ConfidenceSet;
use crate::payoff_oracle::PayoffOracle;
use crate::sequence_form::SequenceForm;

/// Stores state for treeplex rm.
pub struct TreeplexRm<'a> {
    sf: &'a SequenceForm,

    regret: Vec<f64>,

    offset: Vec<usize>,

    avg: Vec<f64>,
    avg_weight: f64,
}

/// Implements operations for `TreeplexRm<'a>`.
impl<'a> TreeplexRm<'a> {
    /// Constructs a new value from the supplied configuration.
    pub fn new(sf: &'a SequenceForm) -> Self {
        let mut offset = Vec::with_capacity(sf.info_sets.len());
        let mut total = 0usize;
        for info in &sf.info_sets {
            offset.push(total);
            total += info.children.len();
        }
        Self {
            sf,
            regret: vec![0.0; total],
            offset,
            avg: vec![0.0; sf.num_sequences()],
            avg_weight: 0.0,
        }
    }

    /// Return the opponent policy as a sequence-form realization plan.
    pub fn realization(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.sf.num_sequences()];
        x[0] = 1.0;
        for (k, info) in self.sf.info_sets.iter().enumerate() {
            let parent = x[info.parent_seq];
            let m = info.children.len();
            let r = &self.regret[self.offset[k]..self.offset[k] + m];
            let sum: f64 = r.iter().map(|v| v.max(0.0)).sum();
            for (a, &(_, child)) in info.children.iter().enumerate() {
                let p = if sum > 0.0 {
                    r[a].max(0.0) / sum
                } else {
                    1.0 / m as f64
                };
                x[child] = parent * p;
            }
        }
        x
    }

    /// Update state from newly observed opponent actions.
    pub fn observe(&mut self, x: &[f64], seq_util: &[f64], w: f64) {
        let mut sv = seq_util.to_vec();
        for (k, info) in self.sf.info_sets.iter().enumerate().rev() {
            let m = info.children.len();
            let parent = x[info.parent_seq];
            let mut value = 0.0;
            for &(_, child) in &info.children {
                let p = if parent > 0.0 {
                    x[child] / parent
                } else {
                    1.0 / m as f64
                };
                value += p * sv[child];
            }
            for (a, &(_, child)) in info.children.iter().enumerate() {
                let slot = &mut self.regret[self.offset[k] + a];
                *slot = (*slot + sv[child] - value).max(0.0);
            }
            sv[info.parent_seq] += value;
        }
        for (acc, xi) in self.avg.iter_mut().zip(x) {
            *acc += w * xi;
        }
        self.avg_weight += w;
    }

    /// Computes average.
    pub fn average(&self) -> Vec<f64> {
        if self.avg_weight == 0.0 {
            return self.realization();
        }
        self.avg.iter().map(|v| v / self.avg_weight).collect()
    }
}

/// Stores state for lagrangian params.
pub struct LagrangianParams {
    pub iters: usize,

    pub eta_lambda: f64,

    pub eta_mu: f64,

    pub lambda_max: f64,
    pub mu_max: f64,

    pub check_every: usize,
}

/// Implements operations for `LagrangianParams`.
impl Default for LagrangianParams {
    /// Returns the default configuration.
    fn default() -> Self {
        Self {
            iters: 20_000,
            eta_lambda: 1.0,
            eta_mu: 1.0,
            lambda_max: 50.0,
            mu_max: 50.0,
            check_every: 2_000,
        }
    }
}

/// Stores state for lagrangian solution.
pub struct LagrangianSolution {
    pub realization: Vec<f64>,

    pub anytime_bound: f64,

    pub floor_value: f64,

    pub lambda: Vec<f64>,
    pub mu: f64,
    pub iters: usize,
}

/// Computes lambda certificate.
pub fn lambda_certificate(
    sf1: &SequenceForm,
    conf: &ConfidenceSet,
    c: &[f64],
    lambda: &[f64],
) -> f64 {
    let mut shifted = c.to_vec();
    for &(r, col, v) in &conf.g_entries {
        shifted[col] += lambda[r] * v;
    }
    let lam_h: f64 = lambda.iter().zip(&conf.h).map(|(l, h)| l * h).sum();
    treeplex_opt(sf1, shifted, false).value - lam_h
}

/// Computes robust response lagrangian.
pub fn robust_response_lagrangian<O: PayoffOracle>(
    oracle: &O,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    conf: &ConfidenceSet,
    v_target: f64,
    params: &LagrangianParams,
) -> LagrangianSolution {
    let mut xrm = TreeplexRm::new(sf0);
    let mut yrm = TreeplexRm::new(sf1);
    let mut lambda = vec![0.0_f64; conf.nrows];
    let mut mu = 0.0_f64;
    let mut best_bound = f64::NEG_INFINITY;
    let mut best_x: Option<Vec<f64>> = None;

    for t in 1..=params.iters {
        let x_t = xrm.realization();
        let y_t = yrm.realization();
        let w = t as f64;
        let step = 1.0 / (t as f64).sqrt();

        let mut y_loss = oracle.at_x(&x_t);
        for &(r, col, v) in &conf.g_entries {
            y_loss[col] += lambda[r] * v;
        }
        let y_util: Vec<f64> = y_loss.iter().map(|v| -v).collect();

        let c_x = oracle.at_x(&x_t);
        let floor_br = treeplex_opt(sf1, c_x, false);
        let mut x_util = oracle.a_y(&y_t);
        if mu > 0.0 {
            for (u, ay) in x_util.iter_mut().zip(oracle.a_y(&floor_br.realization)) {
                *u += mu * ay;
            }
        }

        xrm.observe(&x_t, &x_util, w);
        yrm.observe(&y_t, &y_util, w);

        let mut gy = vec![0.0; conf.nrows];
        for &(r, col, v) in &conf.g_entries {
            gy[r] += v * y_t[col];
        }
        for (l, (g, h)) in lambda.iter_mut().zip(gy.iter().zip(&conf.h)) {
            *l = (*l + params.eta_lambda * step * (g - h)).clamp(0.0, params.lambda_max);
        }
        mu = (mu + params.eta_mu * step * (v_target - floor_br.value)).clamp(0.0, params.mu_max);

        if t % params.check_every == 0 || t == params.iters {
            let x_bar = xrm.average();
            let c_bar = oracle.at_x(&x_bar);
            let bound = lambda_certificate(sf1, conf, &c_bar, &lambda);
            if bound > best_bound {
                best_bound = bound;
                best_x = Some(x_bar);
            }
        }
    }

    let realization = best_x.unwrap_or_else(|| xrm.average());
    let floor_value = treeplex_opt(sf1, oracle.at_x(&realization), false).value;
    LagrangianSolution {
        realization,
        anytime_bound: best_bound,
        floor_value,
        lambda,
        mu,
        iters: params.iters,
    }
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;
    use crate::holdem::{build_holdem, canonical_holdem, compile_holdem};
    use crate::river_range::test_util::random_behavior;
    use crate::river_range::RangeGame;
    use crate::robust_cuts::test_util::box_confidence;
    use crate::robust_cuts::{inner_min_lp, repair_to_floor};

    #[test]
    /// Verifies that lagrangian certifies near exact on compact river.
    fn lagrangian_certifies_near_exact_on_compact_river() {
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

        let sol = robust_response_lagrangian(
            &pm,
            &sf0,
            &sf1,
            &conf,
            v_target,
            &LagrangianParams::default(),
        );

        assert!(
            sol.anytime_bound <= exact.robust_value + 1e-7,
            "anytime bound {} exceeds exact optimum {}",
            sol.anytime_bound,
            exact.robust_value
        );

        let (fixed, _alpha, w) =
            repair_to_floor(&pm, &sf1, &sol.realization, &bp.realization, v_target);
        assert!(w >= v_target - 1e-9, "repaired floor {w} < {v_target}");
        let (certified, _y) = inner_min_lp(&sf1, &conf, &pm.matvec_at_x(&fixed));
        assert!(
            certified >= exact.robust_value - 0.02,
            "certified {certified} more than 1%-scale below exact {}",
            exact.robust_value
        );
        assert!(
            certified <= exact.robust_value + 1e-7,
            "certified {certified} exceeds the exact optimum {}",
            exact.robust_value
        );
    }
}
