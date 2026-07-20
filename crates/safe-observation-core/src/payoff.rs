use std::collections::BTreeMap;

use ndarray::Array2;

use crate::game::{Game, Node};
use crate::kuhn::Kuhn;
use crate::sequence_form::{compile_kuhn, SequenceForm};

pub struct PayoffMatrix {
    pub nrows: usize,

    pub ncols: usize,

    pub entries: Vec<(usize, usize, f64)>,
}

impl PayoffMatrix {
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    pub fn dense(&self) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((self.nrows, self.ncols));
        for &(r, c, v) in &self.entries {
            a[[r, c]] += v;
        }
        a
    }

    pub fn matvec_a_y(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nrows];
        for &(r, c, v) in &self.entries {
            out[r] += v * y[c];
        }
        out
    }

    pub fn matvec_at_x(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.ncols];
        for &(r, c, v) in &self.entries {
            out[c] += v * x[r];
        }
        out
    }

    pub fn bilinear(&self, x: &[f64], y: &[f64]) -> f64 {
        self.entries.iter().map(|&(r, c, v)| x[r] * v * y[c]).sum()
    }
}

#[allow(clippy::too_many_arguments)]
fn terminal_walk<G: Game, F: FnMut(usize, usize, f64)>(
    game: &G,
    state: &G::State,
    prob: f64,
    seq0: usize,
    seq1: usize,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    visit: &mut F,
) {
    match game.node(state) {
        Node::Terminal(value) => {
            visit(seq0, seq1, prob * value);
        }
        Node::Chance(outcomes) => {
            for (p, next) in &outcomes {
                terminal_walk(game, next, prob * p, seq0, seq1, sf0, sf1, visit);
            }
        }
        Node::Decision {
            player,
            infoset,
            actions,
        } => {
            for (a, next) in &actions {
                let label = format!("{infoset}>{a}");
                if player == 0 {
                    let child = sf0.sequence_index(&label).expect("player 1 sequence label");
                    terminal_walk(game, next, prob, child, seq1, sf0, sf1, visit);
                } else {
                    let child = sf1.sequence_index(&label).expect("player 2 sequence label");
                    terminal_walk(game, next, prob, seq0, child, sf0, sf1, visit);
                }
            }
        }
    }
}

pub fn build<G: Game>(game: &G, sf0: &SequenceForm, sf1: &SequenceForm) -> PayoffMatrix {
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    terminal_walk(
        game,
        &game.root(),
        1.0,
        0,
        0,
        sf0,
        sf1,
        &mut |s0, s1, contrib| {
            *acc.entry((s0, s1)).or_insert(0.0) += contrib;
        },
    );
    let entries = acc.into_iter().map(|((r, c), v)| (r, c, v)).collect();
    PayoffMatrix {
        nrows: sf0.num_sequences(),
        ncols: sf1.num_sequences(),
        entries,
    }
}

pub fn apply_a_y<G: Game>(game: &G, sf0: &SequenceForm, sf1: &SequenceForm, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; sf0.num_sequences()];
    terminal_walk(
        game,
        &game.root(),
        1.0,
        0,
        0,
        sf0,
        sf1,
        &mut |s0, s1, contrib| {
            out[s0] += contrib * y[s1];
        },
    );
    out
}

pub fn apply_at_x<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    x: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; sf1.num_sequences()];
    terminal_walk(
        game,
        &game.root(),
        1.0,
        0,
        0,
        sf0,
        sf1,
        &mut |s0, s1, contrib| {
            out[s1] += contrib * x[s0];
        },
    );
    out
}

pub fn build_kuhn() -> PayoffMatrix {
    build(&Kuhn, &compile_kuhn(0), &compile_kuhn(1))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::kuhn::{deals, is_terminal, terminal_value_p1};

    const SYMBOLS: [char; 2] = ['p', 'b'];

    fn expected_value(
        b0: &HashMap<String, [f64; 2]>,
        b1: &HashMap<String, [f64; 2]>,
        cards: (usize, usize),
        history: &str,
    ) -> f64 {
        if is_terminal(history) {
            return terminal_value_p1(history, cards);
        }
        let acting = history.len() % 2;
        let (card, behavior) = if acting == 0 {
            (cards.0, b0)
        } else {
            (cards.1, b1)
        };
        let probs = behavior
            .get(&format!("{card}:{history}"))
            .copied()
            .unwrap_or([0.5, 0.5]);
        let mut value = 0.0;
        for (a, &sym) in SYMBOLS.iter().enumerate() {
            let mut next = history.to_string();
            next.push(sym);
            value += probs[a] * expected_value(b0, b1, cards, &next);
        }
        value
    }

    fn reference_total(b0: &HashMap<String, [f64; 2]>, b1: &HashMap<String, [f64; 2]>) -> f64 {
        deals()
            .iter()
            .map(|&cards| expected_value(b0, b1, cards, "") / 6.0)
            .sum()
    }

    fn to_vec_map(m: &HashMap<String, [f64; 2]>) -> HashMap<String, Vec<f64>> {
        m.iter().map(|(k, v)| (k.clone(), v.to_vec())).collect()
    }

    #[test]
    fn shape_and_nnz() {
        let a = build_kuhn();
        assert_eq!(a.nrows, 13);
        assert_eq!(a.ncols, 13);

        assert_eq!(a.nnz(), 30);
    }

    #[test]
    fn bilinear_matches_uniform_reference() {
        let a = build_kuhn();
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let x = sf0.realization_from_behavior(&HashMap::new());
        let y = sf1.realization_from_behavior(&HashMap::new());
        let expected = reference_total(&HashMap::new(), &HashMap::new());
        assert!((a.bilinear(&x, &y) - expected).abs() < 1e-12);
    }

    #[test]
    fn bilinear_matches_asymmetric_reference() {
        let a = build_kuhn();
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);

        let mut b0 = HashMap::new();
        for (i, info) in sf0.info_sets.iter().enumerate() {
            let bet = 0.2 + 0.1 * (i as f64);
            b0.insert(info.label.clone(), [1.0 - bet, bet]);
        }
        let mut b1 = HashMap::new();
        for (i, info) in sf1.info_sets.iter().enumerate() {
            let bet = 0.7 - 0.08 * (i as f64);
            b1.insert(info.label.clone(), [1.0 - bet, bet]);
        }

        let x = sf0.realization_from_behavior(&to_vec_map(&b0));
        let y = sf1.realization_from_behavior(&to_vec_map(&b1));
        let expected = reference_total(&b0, &b1);
        assert!((a.bilinear(&x, &y) - expected).abs() < 1e-12);
    }

    #[test]
    fn terminal_payoff_matrix_entries() {
        let a = build_kuhn();
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let dense = a.dense();

        let r = sf0.sequence_index("2:>b").unwrap();
        let c = sf1.sequence_index("0:b>b").unwrap();
        assert!((dense[[r, c]] - 2.0 / 6.0).abs() < 1e-12);

        let r = sf0.sequence_index("0:>b").unwrap();
        let c = sf1.sequence_index("2:b>b").unwrap();
        assert!((dense[[r, c]] + 2.0 / 6.0).abs() < 1e-12);

        let r = sf0.sequence_index("2:>p").unwrap();
        let c = sf1.sequence_index("0:p>p").unwrap();
        assert!((dense[[r, c]] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn matvec_consistent_with_bilinear() {
        let a = build_kuhn();
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let x = sf0.realization_from_behavior(&HashMap::new());
        let y = sf1.realization_from_behavior(&HashMap::new());

        let ay = a.matvec_a_y(&y);
        let dot_xay: f64 = x.iter().zip(&ay).map(|(xi, ai)| xi * ai).sum();
        assert!((dot_xay - a.bilinear(&x, &y)).abs() < 1e-12);

        let atx = a.matvec_at_x(&x);
        let dot_atxy: f64 = atx.iter().zip(&y).map(|(ai, yi)| ai * yi).sum();
        assert!((dot_atxy - a.bilinear(&x, &y)).abs() < 1e-12);
    }

    fn pseudo_vec(n: usize, seed: u64) -> Vec<f64> {
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

    fn assert_oracle_matches<G: Game>(game: &G, sf0: &SequenceForm, sf1: &SequenceForm) {
        let a = build(game, sf0, sf1);
        let y = pseudo_vec(sf1.num_sequences(), 0x51ED);
        let x = pseudo_vec(sf0.num_sequences(), 0xA11CE);

        let ay_mat = a.matvec_a_y(&y);
        let ay_oracle = apply_a_y(game, sf0, sf1, &y);
        assert_eq!(ay_mat.len(), ay_oracle.len());
        for (m, o) in ay_mat.iter().zip(&ay_oracle) {
            assert!((m - o).abs() < 1e-12, "apply_a_y mismatch: {m} vs {o}");
        }

        let atx_mat = a.matvec_at_x(&x);
        let atx_oracle = apply_at_x(game, sf0, sf1, &x);
        assert_eq!(atx_mat.len(), atx_oracle.len());
        for (m, o) in atx_mat.iter().zip(&atx_oracle) {
            assert!((m - o).abs() < 1e-12, "apply_at_x mismatch: {m} vs {o}");
        }
    }

    #[test]
    fn oracle_matches_materialized_kuhn() {
        assert_oracle_matches(&Kuhn, &compile_kuhn(0), &compile_kuhn(1));
    }

    #[test]
    fn oracle_matches_materialized_leduc() {
        use crate::leduc::{compile_leduc, Leduc};
        assert_oracle_matches(&Leduc, &compile_leduc(0), &compile_leduc(1));
    }

    #[test]
    fn oracle_matches_materialized_goofspiel() {
        use crate::goofspiel::Goofspiel;
        use crate::sequence_form::compile;
        assert_oracle_matches(&Goofspiel, &compile(&Goofspiel, 0), &compile(&Goofspiel, 1));
    }
}
