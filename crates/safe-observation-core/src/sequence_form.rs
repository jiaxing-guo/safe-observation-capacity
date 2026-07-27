//! Sequence form algorithms for safe observation. See Preliminaries and Problem Setup.

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use crate::game::{Game, Node};
use crate::kuhn::Kuhn;

#[derive(Clone, Debug)]
/// Stores state for info set.
pub struct InfoSet {
    pub label: String,

    pub parent_seq: usize,

    pub children: Vec<(char, usize)>,
}

/// Stores state for sequence form.
pub struct SequenceForm {
    pub player: usize,

    pub sequences: Vec<String>,

    pub info_sets: Vec<InfoSet>,
    seq_index: HashMap<String, usize>,
}

/// Implements operations for `SequenceForm`.
impl SequenceForm {
    /// Constructs a value from parts.
    pub(crate) fn from_parts(
        player: usize,
        sequences: Vec<String>,
        info_sets: Vec<InfoSet>,
    ) -> Self {
        let seq_index = sequences
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        Self {
            player,
            sequences,
            info_sets,
            seq_index,
        }
    }

    /// Computes num sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Computes num information sets.
    pub fn num_infosets(&self) -> usize {
        self.info_sets.len()
    }

    /// Computes sequence index.
    pub fn sequence_index(&self, label: &str) -> Option<usize> {
        self.seq_index.get(label).copied()
    }

    /// Computes num constraints.
    pub fn num_constraints(&self) -> usize {
        1 + self.info_sets.len()
    }

    /// Computes constraint entries.
    pub fn constraint_entries(&self) -> Vec<(usize, usize, f64)> {
        // Row zero fixes the empty sequence. Every later row enforces
        // sum(children) - parent = 0 at one information set.
        let mut entries = vec![(0usize, 0usize, 1.0f64)];
        for (k, info) in self.info_sets.iter().enumerate() {
            let row = 1 + k;
            for &(_, child) in &info.children {
                entries.push((row, child, 1.0));
            }
            entries.push((row, info.parent_seq, -1.0));
        }
        entries
    }

    /// Computes constraint rhs.
    pub fn constraint_rhs(&self) -> Vec<f64> {
        let mut e = vec![0.0; self.num_constraints()];
        e[0] = 1.0;
        e
    }

    /// Computes dense e matrix.
    pub fn dense_e_matrix(&self) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((self.num_constraints(), self.num_sequences()));
        for (row, col, value) in self.constraint_entries() {
            m[[row, col]] = value;
        }
        m
    }

    /// Computes constraint residual.
    pub fn constraint_residual(&self, x: &[f64]) -> f64 {
        let x = Array1::from(x.to_vec());
        let ex = self.dense_e_matrix().dot(&x);
        let rhs = Array1::from(self.constraint_rhs());
        (&ex - &rhs)
            .mapv(f64::abs)
            .into_iter()
            .fold(0.0_f64, f64::max)
    }

    /// Computes realization from behavior.
    pub fn realization_from_behavior(&self, behavior: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        let mut x = vec![0.0; self.sequences.len()];
        // Realization mass is propagated from the empty sequence through each
        // local behavioral distribution.
        x[0] = 1.0;
        for info in &self.info_sets {
            let parent = x[info.parent_seq];
            let n = info.children.len();
            for (i, &(_, child)) in info.children.iter().enumerate() {
                let p = match behavior.get(&info.label) {
                    Some(dist) => dist[i],
                    None => 1.0 / n as f64,
                };
                x[child] = parent * p;
            }
        }
        x
    }

    /// Ensure sequence.
    fn ensure_sequence(&mut self, label: &str) -> usize {
        if let Some(&i) = self.seq_index.get(label) {
            return i;
        }
        let i = self.sequences.len();
        self.sequences.push(label.to_string());
        self.seq_index.insert(label.to_string(), i);
        i
    }
}

/// Compile walk.
fn compile_walk<G: Game>(
    game: &G,
    state: &G::State,
    player: usize,
    cur_seq: usize,
    sf: &mut SequenceForm,
    info_index: &mut HashMap<String, usize>,
) {
    match game.node(state) {
        Node::Terminal(_) => {}
        Node::Chance(outcomes) => {
            for (_p, next) in &outcomes {
                compile_walk(game, next, player, cur_seq, sf, info_index);
            }
        }
        Node::Decision {
            player: acting,
            infoset,
            actions,
        } => {
            if acting != player {
                // Other-player and chance branches do not change this
                // player's active sequence.
                for (_a, next) in &actions {
                    compile_walk(game, next, player, cur_seq, sf, info_index);
                }
                return;
            }

            let children: Vec<(char, usize)> = match info_index.get(&infoset) {
                Some(&idx) => sf.info_sets[idx].children.clone(),
                None => {
                    // An information set is compiled once even when several
                    // game-tree histories share it.
                    let children: Vec<(char, usize)> = actions
                        .iter()
                        .map(|(a, _)| (*a, sf.ensure_sequence(&format!("{infoset}>{a}"))))
                        .collect();
                    let idx = sf.info_sets.len();
                    sf.info_sets.push(InfoSet {
                        label: infoset.clone(),
                        parent_seq: cur_seq,
                        children: children.clone(),
                    });
                    info_index.insert(infoset, idx);
                    children
                }
            };
            for ((_a, next), (_lbl, child)) in actions.iter().zip(&children) {
                compile_walk(game, next, player, *child, sf, info_index);
            }
        }
    }
}

/// Compile a game tree into sequence form.
pub fn compile<G: Game>(game: &G, player: usize) -> SequenceForm {
    let mut sf = SequenceForm {
        player,
        sequences: vec![String::new()],
        info_sets: Vec::new(),
        seq_index: HashMap::from([(String::new(), 0)]),
    };
    let mut info_index: HashMap<String, usize> = HashMap::new();
    compile_walk(game, &game.root(), player, 0, &mut sf, &mut info_index);
    sf
}

/// Compile Kuhn.
pub fn compile_kuhn(player: usize) -> SequenceForm {
    compile(&Kuhn, player)
}

/// Computes sizes.
pub fn sizes(player: usize) -> (usize, usize) {
    let sf = compile_kuhn(player);
    (sf.num_sequences(), sf.num_infosets())
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;

    #[test]
    /// Verifies that Kuhn has thirteen sequences six information sets.
    fn kuhn_has_thirteen_sequences_six_infosets() {
        assert_eq!(sizes(0), (13, 6));
        assert_eq!(sizes(1), (13, 6));
    }

    #[test]
    /// Verifies that empty sequence is index zero.
    fn empty_sequence_is_index_zero() {
        let sf = compile_kuhn(1);
        assert_eq!(sf.sequences[0], "");
        assert_eq!(sf.sequence_index(""), Some(0));
    }

    #[test]
    /// Verifies that constraint system has expected shape.
    fn constraint_system_has_expected_shape() {
        let sf = compile_kuhn(0);
        assert_eq!(sf.num_constraints(), 7);
        let e = sf.constraint_rhs();
        assert_eq!(e[0], 1.0);
        assert!(e[1..].iter().all(|&v| v == 0.0));
    }

    #[test]
    /// Verifies that child weights sum to parent for uniform plan.
    fn child_weights_sum_to_parent_for_uniform_plan() {
        for player in 0..2 {
            let sf = compile_kuhn(player);
            let x = sf.realization_from_behavior(&HashMap::new());
            assert!((x[0] - 1.0).abs() < 1e-12, "root weight must be 1");

            assert!(sf.constraint_residual(&x) < 1e-12);
        }
    }

    #[test]
    /// Verifies that biased plan also satisfies constraints.
    fn biased_plan_also_satisfies_constraints() {
        let sf = compile_kuhn(1);
        let mut behavior = HashMap::new();

        for info in &sf.info_sets {
            behavior.insert(info.label.clone(), vec![0.25, 0.75]);
        }
        let x = sf.realization_from_behavior(&behavior);
        assert!(sf.constraint_residual(&x) < 1e-12);
    }

    #[test]
    /// Verifies that nonfeasible vector has positive residual.
    fn nonfeasible_vector_has_positive_residual() {
        let sf = compile_kuhn(0);
        let mut x = sf.realization_from_behavior(&HashMap::new());
        x[1] += 0.5;
        assert!(sf.constraint_residual(&x) > 1e-6);
    }

    #[test]
    /// Verifies that uniform realization has expected weights.
    fn uniform_realization_has_expected_weights() {
        let sf = compile_kuhn(0);
        let x = sf.realization_from_behavior(&HashMap::new());
        let p = sf.sequence_index("0:>p").expect("sequence 0:>p");
        let pbp = sf.sequence_index("0:pb>p").expect("sequence 0:pb>p");
        assert!((x[p] - 0.5).abs() < 1e-12);
        assert!((x[pbp] - 0.25).abs() < 1e-12);
    }

    #[test]
    /// Verifies that player0 second move parent is first action.
    fn player0_second_move_parent_is_first_action() {
        let sf = compile_kuhn(0);
        let parent = sf.sequence_index("0:>p").expect("sequence 0:>p");
        let info = sf
            .info_sets
            .iter()
            .find(|i| i.label == "0:pb")
            .expect("info set 0:pb");
        assert_eq!(info.parent_seq, parent);
    }
}
