//! Probe algorithms for safe observation. See Safe Active De-censoring and supplementary Algorithms.

use std::collections::HashMap;

use crate::game::{Game, Node};
use crate::sequence_form::{InfoSet, SequenceForm};

/// Compute coefficients for probe reach.
pub fn probe_reach_coeffs<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    opp_behavior: &HashMap<String, Vec<f64>>,
    weights: &HashMap<String, f64>,
) -> Vec<f64> {
    let by_label: HashMap<&str, &InfoSet> = sf0
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), i))
        .collect();
    let mut c = vec![0.0; sf0.num_sequences()];
    let root = game.root();
    let mut ctx = Walk {
        game,
        by_label: &by_label,
        opp_behavior,
        weights,
        c: &mut c,
    };
    ctx.walk(&root, 0, 1.0, 1.0);
    c
}

/// Traverse the game tree while accumulating reach contributions.
struct Walk<'a, G: Game> {
    game: &'a G,
    by_label: &'a HashMap<&'a str, &'a InfoSet>,
    opp_behavior: &'a HashMap<String, Vec<f64>>,
    weights: &'a HashMap<String, f64>,
    c: &'a mut [f64],
}

/// Implements operations for `Walk<'_, G>`.
impl<G: Game> Walk<'_, G> {
    /// Traverse the game tree while accumulating reach contributions.
    fn walk(&mut self, state: &G::State, agent_seq: usize, chance: f64, opp_reach: f64) {
        match self.game.node(state) {
            Node::Terminal(_) => {}
            Node::Chance(outcomes) => {
                for (p, next) in &outcomes {
                    self.walk(next, agent_seq, chance * p, opp_reach);
                }
            }
            Node::Decision {
                player,
                infoset,
                actions,
            } => {
                if player == 0 {
                    // Agent actions advance the active realization sequence.
                    let info = self.by_label[infoset.as_str()];
                    for (ch, next) in &actions {
                        let child = info
                            .children
                            .iter()
                            .find(|(c, _)| c == ch)
                            .expect("agent action not in info-set children")
                            .1;
                        self.walk(next, child, chance, opp_reach);
                    }
                } else {
                    // Opponent nodes contribute before branching; continuation
                    // probabilities affect only their descendants.
                    let w = self.weights.get(&infoset).copied().unwrap_or(0.0);
                    if w != 0.0 {
                        self.c[agent_seq] += chance * opp_reach * w;
                    }
                    let n = actions.len();
                    let behavior = self.opp_behavior.get(&infoset);
                    for (a, (_ch, next)) in actions.iter().enumerate() {
                        let prob = match behavior {
                            Some(b) => b[a],
                            None => 1.0 / n as f64,
                        };
                        self.walk(next, agent_seq, chance, opp_reach * prob);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;
    use crate::kuhn::Kuhn;
    use crate::sequence_form::compile_kuhn;

    /// Compute weights for all ones.
    fn all_ones_weights(sf1: &SequenceForm) -> HashMap<String, f64> {
        sf1.info_sets
            .iter()
            .map(|i| (i.label.clone(), 1.0))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    /// Compute information gain by direct scalar tree traversal.
    fn scalar_ig(
        game: &Kuhn,
        state: &<Kuhn as Game>::State,
        agent_behavior: &HashMap<String, Vec<f64>>,
        opp_behavior: &HashMap<String, Vec<f64>>,
        weights: &HashMap<String, f64>,
        chance: f64,
        agent_reach: f64,
        opp_reach: f64,
    ) -> f64 {
        match game.node(state) {
            Node::Terminal(_) => 0.0,
            Node::Chance(outcomes) => outcomes
                .iter()
                .map(|(p, next)| {
                    scalar_ig(
                        game,
                        next,
                        agent_behavior,
                        opp_behavior,
                        weights,
                        chance * p,
                        agent_reach,
                        opp_reach,
                    )
                })
                .sum(),
            Node::Decision {
                player,
                infoset,
                actions,
            } => {
                let n = actions.len();
                if player == 0 {
                    let dist = agent_behavior.get(&infoset);
                    actions
                        .iter()
                        .enumerate()
                        .map(|(a, (_ch, next))| {
                            let p = dist.map_or(1.0 / n as f64, |d| d[a]);
                            scalar_ig(
                                game,
                                next,
                                agent_behavior,
                                opp_behavior,
                                weights,
                                chance,
                                agent_reach * p,
                                opp_reach,
                            )
                        })
                        .sum()
                } else {
                    let w = weights.get(&infoset).copied().unwrap_or(0.0);
                    let here = w * chance * agent_reach * opp_reach;
                    let dist = opp_behavior.get(&infoset);
                    here + actions
                        .iter()
                        .enumerate()
                        .map(|(a, (_ch, next))| {
                            let p = dist.map_or(1.0 / n as f64, |d| d[a]);
                            scalar_ig(
                                game,
                                next,
                                agent_behavior,
                                opp_behavior,
                                weights,
                                chance,
                                agent_reach,
                                opp_reach * p,
                            )
                        })
                        .sum::<f64>()
                }
            }
        }
    }

    #[test]
    /// Verifies that zero weights give zero coefficients.
    fn zero_weights_give_zero_coeffs() {
        let sf0 = compile_kuhn(0);
        let c = probe_reach_coeffs(&Kuhn, &sf0, &HashMap::new(), &HashMap::new());
        assert_eq!(c.len(), sf0.num_sequences());
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    /// Verifies that coefficients are nonnegative and some positive.
    fn coeffs_are_nonnegative_and_some_positive() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let weights = all_ones_weights(&sf1);
        let c = probe_reach_coeffs(&Kuhn, &sf0, &HashMap::new(), &weights);
        assert!(c.iter().all(|&v| v >= 0.0));
        assert!(c.iter().any(|&v| v > 0.0));
    }

    #[test]
    /// Verifies that coefficients match direct scalar walk.
    fn coeffs_match_direct_scalar_walk() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let weights = all_ones_weights(&sf1);

        let mut opp_behavior = HashMap::new();
        for (k, info) in sf1.info_sets.iter().enumerate() {
            let p = 0.2 + 0.1 * (k % 3) as f64;
            opp_behavior.insert(info.label.clone(), vec![p, 1.0 - p]);
        }
        let c = probe_reach_coeffs(&Kuhn, &sf0, &opp_behavior, &weights);

        for trial in 0..3 {
            let mut agent_behavior = HashMap::new();
            for (k, info) in sf0.info_sets.iter().enumerate() {
                let p = 0.3 + 0.15 * ((k + trial) % 4) as f64;
                agent_behavior.insert(info.label.clone(), vec![p, 1.0 - p]);
            }
            let x = sf0.realization_from_behavior(&agent_behavior);
            let dot: f64 = c.iter().zip(&x).map(|(ci, xi)| ci * xi).sum();
            let direct = scalar_ig(
                &Kuhn,
                &Kuhn.root(),
                &agent_behavior,
                &opp_behavior,
                &weights,
                1.0,
                1.0,
                1.0,
            );
            assert!(
                (dot - direct).abs() < 1e-12,
                "trial {trial}: c.x = {dot}, direct = {direct}"
            );
        }
    }
}
