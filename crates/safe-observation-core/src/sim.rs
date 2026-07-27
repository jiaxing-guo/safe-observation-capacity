//! Sim algorithms for safe observation. See supplementary Reproducibility for its role in the release workflow.

use std::collections::HashMap;

use crate::game::{Game, Node};
use crate::kuhn::Kuhn;

/// Stores state for split mix64.
struct SplitMix64 {
    state: u64,
}

/// Implements operations for `SplitMix64`.
impl SplitMix64 {
    /// Constructs a new value from the supplied configuration.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Computes next integer.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draw the next uniform floating-point sample.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Stores state for sim result.
pub struct SimResult {
    pub total_payoff: f64,

    pub p2_counts: HashMap<String, Vec<u64>>,
}

/// Draw a sample from the configured distribution.
fn sample(rng: &mut SplitMix64, probs: &[f64]) -> usize {
    let u = rng.next_f64();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample outcome.
fn sample_outcome<S>(rng: &mut SplitMix64, outcomes: &[(f64, S)]) -> usize {
    let u = rng.next_f64();
    let mut acc = 0.0;
    for (i, (p, _)) in outcomes.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    outcomes.len() - 1
}

/// Simulate trajectories under the configured policies.
pub fn simulate<G: Game>(
    game: &G,
    strat0: &HashMap<String, Vec<f64>>,
    strat1: &HashMap<String, Vec<f64>>,
    episodes: u64,
    seed: u64,
) -> SimResult {
    let mut rng = SplitMix64::new(seed);
    let mut total_payoff = 0.0;
    let mut p2_counts: HashMap<String, Vec<u64>> = HashMap::new();

    for _ in 0..episodes {
        let mut state = game.root();
        loop {
            match game.node(&state) {
                Node::Terminal(value) => {
                    total_payoff += value;
                    break;
                }
                Node::Chance(outcomes) => {
                    let idx = sample_outcome(&mut rng, &outcomes);
                    state = outcomes.into_iter().nth(idx).unwrap().1;
                }
                Node::Decision {
                    player,
                    infoset,
                    actions,
                } => {
                    let strat = if player == 0 { strat0 } else { strat1 };
                    let probs = strat
                        .get(&infoset)
                        .unwrap_or_else(|| panic!("missing behaviour at {infoset}"));
                    let a = sample(&mut rng, probs);
                    if player == 1 {
                        p2_counts
                            .entry(infoset)
                            .or_insert_with(|| vec![0; actions.len()])[a] += 1;
                    }
                    state = actions.into_iter().nth(a).unwrap().1;
                }
            }
        }
    }

    SimResult {
        total_payoff,
        p2_counts,
    }
}

/// Stores state for showdown sim result.
pub struct ShowdownSimResult {
    pub total_payoff: f64,

    pub p2_counts_showdown: HashMap<String, Vec<u64>>,

    pub p2_counts_folded: HashMap<String, Vec<u64>>,
}

/// Simulate showdown.
pub fn simulate_showdown<G: Game>(
    game: &G,
    strat0: &HashMap<String, Vec<f64>>,
    strat1: &HashMap<String, Vec<f64>>,
    episodes: u64,
    seed: u64,
) -> ShowdownSimResult {
    let mut rng = SplitMix64::new(seed);
    let mut total_payoff = 0.0;
    let mut p2_counts_showdown: HashMap<String, Vec<u64>> = HashMap::new();
    let mut p2_counts_folded: HashMap<String, Vec<u64>> = HashMap::new();

    for _ in 0..episodes {
        let mut state = game.root();

        let mut opp_decisions: Vec<(String, usize, usize)> = Vec::new();
        let mut hand_folded = false;
        loop {
            match game.node(&state) {
                Node::Terminal(value) => {
                    total_payoff += value;
                    break;
                }
                Node::Chance(outcomes) => {
                    let idx = sample_outcome(&mut rng, &outcomes);
                    state = outcomes.into_iter().nth(idx).unwrap().1;
                }
                Node::Decision {
                    player,
                    infoset,
                    actions,
                } => {
                    let strat = if player == 0 { strat0 } else { strat1 };
                    let probs = strat
                        .get(&infoset)
                        .unwrap_or_else(|| panic!("missing behaviour at {infoset}"));
                    let a = sample(&mut rng, probs);
                    if actions[a].0 == 'f' {
                        hand_folded = true;
                    }
                    if player == 1 {
                        opp_decisions.push((infoset, a, actions.len()));
                    }
                    state = actions.into_iter().nth(a).unwrap().1;
                }
            }
        }
        let dest = if hand_folded {
            &mut p2_counts_folded
        } else {
            &mut p2_counts_showdown
        };
        for (infoset, a, n_actions) in opp_decisions {
            dest.entry(infoset).or_insert_with(|| vec![0; n_actions])[a] += 1;
        }
    }

    ShowdownSimResult {
        total_payoff,
        p2_counts_showdown,
        p2_counts_folded,
    }
}

/// Simulate Kuhn.
pub fn simulate_kuhn(
    x1: &HashMap<String, Vec<f64>>,
    y2: &HashMap<String, Vec<f64>>,
    episodes: u64,
    seed: u64,
) -> SimResult {
    simulate(&Kuhn, x1, y2, episodes, seed)
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;
    use crate::sequence_form::compile_kuhn;

    /// Computes uniform behavior.
    fn uniform_behavior(player: usize) -> HashMap<String, Vec<f64>> {
        let sf = compile_kuhn(player);
        sf.info_sets
            .iter()
            .map(|info| (info.label.clone(), vec![0.5, 0.5]))
            .collect()
    }

    /// Computes always pass player-two.
    fn always_pass_p2() -> HashMap<String, Vec<f64>> {
        let sf = compile_kuhn(1);
        sf.info_sets
            .iter()
            .map(|info| (info.label.clone(), vec![1.0, 0.0]))
            .collect()
    }

    #[test]
    /// Verifies that deterministic for fixed seed.
    fn deterministic_for_fixed_seed() {
        let x1 = uniform_behavior(0);
        let y2 = uniform_behavior(1);
        let a = simulate_kuhn(&x1, &y2, 1000, 2026);
        let b = simulate_kuhn(&x1, &y2, 1000, 2026);
        assert_eq!(a.total_payoff, b.total_payoff);
        assert_eq!(a.p2_counts, b.p2_counts);
    }

    #[test]
    /// Verifies that counts total matches visits.
    fn counts_total_matches_visits() {
        let x1 = uniform_behavior(0);
        let y2 = uniform_behavior(1);
        let r = simulate_kuhn(&x1, &y2, 5000, 2026);
        let total: u64 = r.p2_counts.values().map(|c| c[0] + c[1]).sum();

        assert!(total <= 5000);
        assert!(total > 0);
    }

    #[test]
    /// Verifies that always pass opponent only passes.
    fn always_pass_opponent_only_passes() {
        let x1 = uniform_behavior(0);
        let y2 = always_pass_p2();
        let r = simulate_kuhn(&x1, &y2, 3000, 2026);
        for counts in r.p2_counts.values() {
            assert_eq!(counts[1], 0, "always-pass opponent must never bet");
        }
    }

    #[test]
    /// Verifies that average payoff in reasonable range.
    fn average_payoff_in_reasonable_range() {
        let x1 = uniform_behavior(0);
        let y2 = uniform_behavior(1);
        let r = simulate_kuhn(&x1, &y2, 20000, 2026);
        let avg = r.total_payoff / 20000.0;
        assert!((-2.0..=2.0).contains(&avg));
    }

    /// Computes uniform Leduc.
    fn uniform_leduc(player: usize) -> HashMap<String, Vec<f64>> {
        let sf = crate::leduc::compile_leduc(player);
        sf.info_sets
            .iter()
            .map(|info| {
                let k = info.children.len();
                (info.label.clone(), vec![1.0 / k as f64; k])
            })
            .collect()
    }

    #[test]
    /// Verifies that showdown split sums to full counts Leduc.
    fn showdown_split_sums_to_full_counts_leduc() {
        use crate::leduc::Leduc;
        let x0 = uniform_leduc(0);
        let y1 = uniform_leduc(1);
        let full = simulate(&Leduc, &x0, &y1, 8000, 2026);
        let split = simulate_showdown(&Leduc, &x0, &y1, 8000, 2026);

        assert_eq!(full.total_payoff, split.total_payoff);

        assert!(!split.p2_counts_showdown.is_empty());
        assert!(!split.p2_counts_folded.is_empty());

        for (label, counts) in &full.p2_counts {
            let s = split.p2_counts_showdown.get(label);
            let f = split.p2_counts_folded.get(label);
            let zeros = vec![0u64; counts.len()];
            let s = s.unwrap_or(&zeros);
            let f = f.unwrap_or(&zeros);
            for i in 0..counts.len() {
                assert_eq!(
                    counts[i],
                    s[i] + f[i],
                    "showdown+folded must equal full at {label}[{i}]"
                );
            }
        }
    }

    #[test]
    /// Verifies that showdown deterministic for fixed seed.
    fn showdown_deterministic_for_fixed_seed() {
        use crate::leduc::Leduc;
        let x0 = uniform_leduc(0);
        let y1 = uniform_leduc(1);
        let a = simulate_showdown(&Leduc, &x0, &y1, 3000, 2026);
        let b = simulate_showdown(&Leduc, &x0, &y1, 3000, 2026);
        assert_eq!(a.total_payoff, b.total_payoff);
        assert_eq!(a.p2_counts_showdown, b.p2_counts_showdown);
        assert_eq!(a.p2_counts_folded, b.p2_counts_folded);
    }
}
