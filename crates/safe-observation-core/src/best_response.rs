//! Best response algorithms for safe observation. See The Safe Observation-Capacity Frontier, Certified Value Recovery, and supplementary Certification at the Unbucketed River.

use crate::game::Game;
use crate::payoff::{apply_a_y, apply_at_x, PayoffMatrix};
use crate::sequence_form::SequenceForm;

/// Stores state for tree best response.
pub struct TreeBestResponse {
    pub value: f64,

    pub realization: Vec<f64>,
}

/// Computes best response dp.
fn best_response_dp(
    sf_r: &SequenceForm,
    mut seq_value: Vec<f64>,
    maximize: bool,
) -> TreeBestResponse {
    debug_assert_eq!(seq_value.len(), sf_r.num_sequences());
    let mut best_child = vec![0usize; sf_r.num_infosets()];
    for (k, info) in sf_r.info_sets.iter().enumerate().rev() {
        let mut best_val = if maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_seq = info.children[0].1;
        for &(_, child) in &info.children {
            let v = seq_value[child];
            if (maximize && v > best_val) || (!maximize && v < best_val) {
                best_val = v;
                best_seq = child;
            }
        }
        best_child[k] = best_seq;
        seq_value[info.parent_seq] += best_val;
    }
    let value = seq_value[0];

    let mut realization = vec![0.0_f64; sf_r.num_sequences()];
    realization[0] = 1.0;
    for (k, info) in sf_r.info_sets.iter().enumerate() {
        let parent = realization[info.parent_seq];
        if parent != 0.0 {
            realization[best_child[k]] = parent;
        }
    }
    TreeBestResponse { value, realization }
}

/// Computes treeplex opt.
pub fn treeplex_opt(sf_r: &SequenceForm, seq_value: Vec<f64>, maximize: bool) -> TreeBestResponse {
    best_response_dp(sf_r, seq_value, maximize)
}

/// Computes safety verify from matrix.
pub fn safety_verify_from_matrix(
    sf1: &SequenceForm,
    payoff: &PayoffMatrix,
    x: &[f64],
) -> TreeBestResponse {
    best_response_dp(sf1, payoff.matvec_at_x(x), false)
}

/// Computes best response player-one from matrix.
pub fn best_response_p1_from_matrix(
    sf0: &SequenceForm,
    payoff: &PayoffMatrix,
    y: &[f64],
) -> TreeBestResponse {
    best_response_dp(sf0, payoff.matvec_a_y(y), true)
}

/// Computes safety verify tree.
pub fn safety_verify_tree<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    x: &[f64],
) -> TreeBestResponse {
    best_response_dp(sf1, apply_at_x(game, sf0, sf1, x), false)
}

/// Computes best response player-one tree.
pub fn best_response_p1_tree<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    y: &[f64],
) -> TreeBestResponse {
    best_response_dp(sf0, apply_a_y(game, sf0, sf1, y), true)
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::goofspiel::Goofspiel;
    use crate::kuhn::Kuhn;
    use crate::leduc::{compile_leduc, Leduc};
    use crate::lp::{best_response_p1, safety_verify};
    use crate::payoff::{build, build_kuhn};
    use crate::sequence_form::{compile, compile_kuhn};

    /// Computes pseudo.
    fn pseudo(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect()
    }

    /// Computes random plan.
    fn random_plan(sf: &SequenceForm, seed: u64) -> Vec<f64> {
        let mut behavior: HashMap<String, Vec<f64>> = HashMap::new();
        for (i, info) in sf.info_sets.iter().enumerate() {
            let raw = pseudo(info.children.len(), seed.wrapping_add(i as u64 * 97));
            let sum: f64 = raw.iter().sum();
            let dist = if sum > 0.0 {
                raw.iter().map(|v| v / sum).collect()
            } else {
                vec![1.0 / info.children.len() as f64; info.children.len()]
            };
            behavior.insert(info.label.clone(), dist);
        }
        sf.realization_from_behavior(&behavior)
    }

    #[test]
    /// Verifies that safety matches linear program Kuhn.
    fn safety_matches_lp_kuhn() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let a = build_kuhn();
        for seed in 0..8 {
            let x = random_plan(&sf0, 0x5A5A + seed);
            let lp = safety_verify(&sf1, &a, &x);
            let mat = safety_verify_from_matrix(&sf1, &a, &x);
            let tree = safety_verify_tree(&Kuhn, &sf0, &sf1, &x);
            assert!(
                (lp.value - mat.value).abs() < 1e-9,
                "kuhn safety matrix seed {seed}"
            );

            assert!(
                (mat.value - tree.value).abs() < 1e-12,
                "backends must agree"
            );

            assert!(sf1.constraint_residual(&mat.realization) < 1e-9);
            assert!((a.bilinear(&x, &mat.realization) - mat.value).abs() < 1e-9);
        }
    }

    #[test]
    /// Verifies that best response matches linear program Kuhn.
    fn best_response_matches_lp_kuhn() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let a = build_kuhn();
        for seed in 0..8 {
            let y = random_plan(&sf1, 0xBEEF + seed);
            let lp = best_response_p1(&sf0, &a, &y);
            let mat = best_response_p1_from_matrix(&sf0, &a, &y);
            let tree = best_response_p1_tree(&Kuhn, &sf0, &sf1, &y);
            assert!(
                (lp.value - mat.value).abs() < 1e-9,
                "kuhn BR matrix seed {seed}"
            );
            assert!((mat.value - tree.value).abs() < 1e-12);
            assert!(sf0.constraint_residual(&mat.realization) < 1e-9);
            assert!((a.bilinear(&mat.realization, &y) - mat.value).abs() < 1e-9);
        }
    }

    #[test]
    /// Verifies that safety matches linear program Leduc.
    fn safety_matches_lp_leduc() {
        let sf0 = compile_leduc(0);
        let sf1 = compile_leduc(1);
        let a = build(&Leduc, &sf0, &sf1);
        for seed in 0..4 {
            let x = random_plan(&sf0, 0x1EDC + seed);
            let lp = safety_verify(&sf1, &a, &x);
            let mat = safety_verify_from_matrix(&sf1, &a, &x);
            let tree = safety_verify_tree(&Leduc, &sf0, &sf1, &x);
            assert!(
                (lp.value - mat.value).abs() < 1e-7,
                "leduc safety seed {seed}: lp {} mat {}",
                lp.value,
                mat.value
            );
            assert!((mat.value - tree.value).abs() < 1e-9);
            assert!(sf1.constraint_residual(&mat.realization) < 1e-9);
            assert!((a.bilinear(&x, &mat.realization) - mat.value).abs() < 1e-7);
        }
    }

    #[test]
    /// Verifies that best response matches linear program Leduc.
    fn best_response_matches_lp_leduc() {
        let sf0 = compile_leduc(0);
        let sf1 = compile_leduc(1);
        let a = build(&Leduc, &sf0, &sf1);
        for seed in 0..4 {
            let y = random_plan(&sf1, 0x1EDD + seed);
            let lp = best_response_p1(&sf0, &a, &y);
            let mat = best_response_p1_from_matrix(&sf0, &a, &y);
            assert!(
                (lp.value - mat.value).abs() < 1e-7,
                "leduc BR seed {seed}: lp {} mat {}",
                lp.value,
                mat.value
            );
            assert!(sf0.constraint_residual(&mat.realization) < 1e-9);
        }
    }

    #[test]
    /// Verifies that safety matches linear program goofspiel.
    fn safety_matches_lp_goofspiel() {
        let sf0 = compile(&Goofspiel, 0);
        let sf1 = compile(&Goofspiel, 1);
        let a = build(&Goofspiel, &sf0, &sf1);
        for seed in 0..4 {
            let x = random_plan(&sf0, 0x600F + seed);
            let lp = safety_verify(&sf1, &a, &x);
            let mat = safety_verify_from_matrix(&sf1, &a, &x);
            let tree = safety_verify_tree(&Goofspiel, &sf0, &sf1, &x);
            assert!(
                (lp.value - mat.value).abs() < 1e-7,
                "goofspiel safety seed {seed}"
            );
            assert!((mat.value - tree.value).abs() < 1e-9);
            assert!(sf1.constraint_residual(&mat.realization) < 1e-9);
        }
    }

    #[test]
    /// Verifies that backends reproducible and agree.
    fn backends_reproducible_and_agree() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let a = build_kuhn();
        let x = random_plan(&sf0, 7);
        let p = safety_verify_from_matrix(&sf1, &a, &x);
        let q = safety_verify_from_matrix(&sf1, &a, &x);
        assert_eq!(p.value, q.value);
        assert_eq!(p.realization, q.realization);
    }

    #[test]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    /// Verifies that bench backends vs linear program Leduc.
    fn bench_backends_vs_lp_leduc() {
        use std::time::Instant;
        let sf0 = compile_leduc(0);
        let sf1 = compile_leduc(1);
        let a = build(&Leduc, &sf0, &sf1);
        let plans: Vec<Vec<f64>> = (0..200).map(|s| random_plan(&sf0, s)).collect();

        let timed = |label: &str, f: &dyn Fn(&[f64]) -> f64| {
            let t = Instant::now();
            let mut acc = 0.0;
            for x in &plans {
                acc += f(x);
            }
            let dt = t.elapsed();
            println!(
                "  {label:28} {:>8.3} ms/call   (check {acc:.4})",
                dt.as_secs_f64() * 1e3 / plans.len() as f64
            );
            dt
        };

        println!("leduc safety_verify x{} plans:", plans.len());
        let lp = timed("LP (HiGHS)", &|x| safety_verify(&sf1, &a, x).value);
        let mat = timed("DP from materialised A", &|x| {
            safety_verify_from_matrix(&sf1, &a, x).value
        });
        let tree = timed("DP from tree-walk oracle", &|x| {
            safety_verify_tree(&Leduc, &sf0, &sf1, x).value
        });
        println!(
            "  matrix-DP speedup vs LP: {:.1}x   oracle-DP vs LP: {:.1}x",
            lp.as_secs_f64() / mat.as_secs_f64(),
            lp.as_secs_f64() / tree.as_secs_f64(),
        );
    }
}
