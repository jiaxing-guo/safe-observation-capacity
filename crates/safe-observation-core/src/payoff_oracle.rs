use crate::payoff::PayoffMatrix;
use crate::river_range::{PubNode, RangeGame};
use crate::sequence_form::SequenceForm;

pub trait PayoffOracle {
    fn at_x(&self, x: &[f64]) -> Vec<f64>;

    fn a_y(&self, y: &[f64]) -> Vec<f64>;
}

impl PayoffOracle for PayoffMatrix {
    fn at_x(&self, x: &[f64]) -> Vec<f64> {
        self.matvec_at_x(x)
    }
    fn a_y(&self, y: &[f64]) -> Vec<f64> {
        self.matvec_a_y(y)
    }
}

pub struct RangeOracle<'a> {
    rg: &'a RangeGame<'a>,

    seq: [Vec<u32>; 2],
    n: [usize; 2],
    ny: usize,
    nx: usize,
}

impl<'a> RangeOracle<'a> {
    pub fn new(rg: &'a RangeGame<'a>, sf0: &SequenceForm, sf1: &SequenceForm) -> Self {
        let n = [rg.combos(0), rg.combos(1)];
        let nodes = rg.tree.nodes.len();
        let mut seq = [vec![0u32; nodes * n[0]], vec![0u32; nodes * n[1]]];
        let sfs = [sf0, sf1];

        let mut stack: Vec<(usize, [Vec<u32>; 2])> =
            vec![(0, [vec![0u32; n[0]], vec![0u32; n[1]]])];
        while let Some((id, cur)) = stack.pop() {
            for p in 0..2 {
                seq[p][id * n[p]..(id + 1) * n[p]].copy_from_slice(&cur[p]);
            }
            if let PubNode::Decision {
                player,
                hist,
                children,
            } = &rg.tree.nodes[id]
            {
                for &(a, child) in children {
                    let mut next = cur.clone();
                    for (k, slot) in next[*player].iter_mut().enumerate() {
                        let label = format!("{}>{}", rg.label(*player, k, hist), a);
                        *slot = sfs[*player]
                            .sequence_index(&label)
                            .unwrap_or_else(|| panic!("unresolved sequence {label}"))
                            as u32;
                    }
                    stack.push((child, next));
                }
            }
        }
        Self {
            rg,
            seq,
            n,
            nx: sf0.num_sequences(),
            ny: sf1.num_sequences(),
        }
    }
}

impl RangeOracle<'_> {
    pub fn seq_at(&self, player: usize, node: usize, combo: usize) -> usize {
        self.seq[player][node * self.n[player] + combo] as usize
    }
}

impl PayoffOracle for RangeOracle<'_> {
    fn at_x(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.ny];
        for (id, node) in self.rg.tree.nodes.iter().enumerate() {
            if matches!(node, PubNode::Decision { .. }) {
                continue;
            }
            let xr: Vec<f64> = self.seq[0][id * self.n[0]..(id + 1) * self.n[0]]
                .iter()
                .map(|&s| x[s as usize])
                .collect();
            let v = self.rg.terminal_values(1, node, &xr);
            let own = &self.seq[1][id * self.n[1]..(id + 1) * self.n[1]];
            for (j, &s) in own.iter().enumerate() {
                out[s as usize] -= v[j];
            }
        }
        out
    }

    fn a_y(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nx];
        for (id, node) in self.rg.tree.nodes.iter().enumerate() {
            if matches!(node, PubNode::Decision { .. }) {
                continue;
            }
            let yr: Vec<f64> = self.seq[1][id * self.n[1]..(id + 1) * self.n[1]]
                .iter()
                .map(|&s| y[s as usize])
                .collect();
            let v = self.rg.terminal_values(0, node, &yr);
            let own = &self.seq[0][id * self.n[0]..(id + 1) * self.n[0]];
            for (i, &s) in own.iter().enumerate() {
                out[s as usize] += v[i];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holdem::{build_holdem, canonical_holdem, compile_holdem};
    use crate::river_range::test_util::random_behavior;
    use crate::river_range::RangeGame;

    #[test]
    fn range_oracle_matvecs_match_payoff_matrix_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = compile_holdem(0);
        let sf1 = compile_holdem(1);
        let pm = build_holdem();
        let oracle = RangeOracle::new(&rg, &sf0, &sf1);
        for seed in [31u64, 32] {
            let x = sf0.realization_from_behavior(&random_behavior(&rg, 0, seed));
            let y = sf1.realization_from_behavior(&random_behavior(&rg, 1, seed + 7));

            let (fast, exact) = (oracle.at_x(&x), pm.matvec_at_x(&x));
            for (s, (f, e)) in fast.iter().zip(&exact).enumerate() {
                assert!(
                    (f - e).abs() <= 1e-10 * (1.0 + e.abs()),
                    "at_x[{s}]: {f} vs {e} (seed {seed})"
                );
            }
            let (fast, exact) = (oracle.a_y(&y), pm.matvec_a_y(&y));
            for (s, (f, e)) in fast.iter().zip(&exact).enumerate() {
                assert!(
                    (f - e).abs() <= 1e-10 * (1.0 + e.abs()),
                    "a_y[{s}]: {f} vs {e} (seed {seed})"
                );
            }
        }
    }
}
