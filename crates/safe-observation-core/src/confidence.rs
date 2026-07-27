//! Confidence algorithms for safe observation. See Public and Active Confidence Sets and supplementary Confidence Sets under Censoring.

use std::collections::{BTreeMap, HashMap};

use crate::game::{Game, Node};
use crate::sequence_form::{compile_kuhn, InfoSet, SequenceForm};

/// Stores state for confidence set.
pub struct ConfidenceSet {
    pub ncols: usize,

    pub nrows: usize,

    pub g_entries: Vec<(usize, usize, f64)>,

    pub h: Vec<f64>,

    pub row_meta: Vec<(String, usize)>,
}

/// Implements operations for `ConfidenceSet`.
impl ConfidenceSet {
    /// Computes max violation.
    pub fn max_violation(&self, y: &[f64]) -> f64 {
        let mut gy = vec![0.0; self.nrows];
        for &(r, c, v) in &self.g_entries {
            gy[r] += v * y[c];
        }
        gy.iter()
            .zip(&self.h)
            .map(|(a, b)| a - b)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Intersect this confidence set with the supplied constraints.
    pub fn intersect(mut self, other: ConfidenceSet) -> ConfidenceSet {
        assert_eq!(
            self.ncols, other.ncols,
            "cannot intersect confidence sets over different sequence forms \
             ({} vs {} columns)",
            self.ncols, other.ncols
        );
        let offset = self.nrows;
        self.g_entries.extend(
            other
                .g_entries
                .into_iter()
                .map(|(r, c, v)| (r + offset, c, v)),
        );
        self.h.extend(other.h);
        self.row_meta.extend(other.row_meta);
        self.nrows += other.nrows;
        self
    }
}

/// Build the configured game or confidence object.
pub fn build(sf: &SequenceForm, intervals: &HashMap<String, Vec<(f64, f64)>>) -> ConfidenceSet {
    let mut g_entries = Vec::new();
    let mut row_meta: Vec<(String, usize)> = Vec::new();
    let mut nrows = 0usize;
    for info in &sf.info_sets {
        let parent = info.parent_seq;
        let bounds = intervals.get(&info.label);
        for (i, &(_, child)) in info.children.iter().enumerate() {
            let (l, u) = bounds.and_then(|b| b.get(i)).copied().unwrap_or((0.0, 1.0));
            // Conditional bounds l <= pi(a|I) <= u become linear sequence-form
            // inequalities l*y_parent <= y_child <= u*y_parent.
            if l > 0.0 {
                g_entries.push((nrows, parent, l));
                g_entries.push((nrows, child, -1.0));
                row_meta.push((info.label.clone(), i));
                nrows += 1;
            }
            if u < 1.0 {
                g_entries.push((nrows, child, 1.0));
                g_entries.push((nrows, parent, -u));
                row_meta.push((info.label.clone(), i));
                nrows += 1;
            }
        }
    }
    let h = vec![0.0; nrows];
    ConfidenceSet {
        ncols: sf.num_sequences(),
        nrows,
        g_entries,
        h,
        row_meta,
    }
}

/// Build Kuhn.
pub fn build_kuhn(intervals: &HashMap<String, Vec<(f64, f64)>>) -> ConfidenceSet {
    let sf = compile_kuhn(1);
    build(&sf, intervals)
}

/// Build boxes.
pub fn build_boxes(sf: &SequenceForm, boxes: &HashMap<String, Vec<(f64, f64)>>) -> ConfidenceSet {
    let mut g_entries = Vec::new();
    let mut row_meta: Vec<(String, usize)> = Vec::new();
    let mut h = Vec::new();
    let mut nrows = 0usize;
    for info in &sf.info_sets {
        let bounds = match boxes.get(&info.label) {
            Some(b) => b,
            None => continue,
        };
        for (i, &(_, child)) in info.children.iter().enumerate() {
            let (l, u) = match bounds.get(i).copied() {
                Some(b) => b,
                None => continue,
            };
            if u < 1.0 {
                g_entries.push((nrows, child, 1.0));
                h.push(u);
                row_meta.push((info.label.clone(), i));
                nrows += 1;
            }
            if l > 0.0 {
                g_entries.push((nrows, child, -1.0));
                h.push(-l);
                row_meta.push((info.label.clone(), i));
                nrows += 1;
            }
        }
    }
    ConfidenceSet {
        ncols: sf.num_sequences(),
        nrows,
        g_entries,
        h,
        row_meta,
    }
}

/// Build linear.
pub fn build_linear(
    sf: &SequenceForm,
    entries: Vec<(usize, usize, f64)>,
    h: Vec<f64>,
    row_meta: Vec<(String, usize)>,
) -> ConfidenceSet {
    let nrows = h.len();
    assert_eq!(
        row_meta.len(),
        nrows,
        "row_meta length {} must match h length {nrows}",
        row_meta.len()
    );
    for &(row, col, _) in &entries {
        assert!(row < nrows, "linear constraint row {row} >= nrows {nrows}");
        assert!(
            col < sf.num_sequences(),
            "linear constraint col {col} >= ncols {}",
            sf.num_sequences()
        );
    }
    ConfidenceSet {
        ncols: sf.num_sequences(),
        nrows,
        g_entries: entries,
        h,
        row_meta,
    }
}

/// Build public.
pub fn build_public(
    sf: &SequenceForm,
    groups: &HashMap<String, Vec<String>>,
    intervals: &HashMap<String, Vec<(f64, f64)>>,
    weights: &HashMap<String, f64>,
) -> ConfidenceSet {
    let by_label: HashMap<&str, &crate::sequence_form::InfoSet> =
        sf.info_sets.iter().map(|i| (i.label.as_str(), i)).collect();

    let mut g_entries = Vec::new();
    let mut row_meta: Vec<(String, usize)> = Vec::new();
    let mut nrows = 0usize;
    let mut group_keys: Vec<&String> = groups.keys().collect();
    group_keys.sort_unstable();
    for public_key in group_keys {
        let labels = &groups[public_key];
        let members: Vec<&crate::sequence_form::InfoSet> = labels
            .iter()
            .filter_map(|l| by_label.get(l.as_str()).copied())
            .collect();
        let bounds = match intervals.get(public_key) {
            Some(b) if !members.is_empty() => b,
            _ => continue,
        };

        let n_actions = members[0].children.len();

        let weight_of =
            |m: &crate::sequence_form::InfoSet| weights.get(&m.label).copied().unwrap_or(1.0);

        // Coalesce repeated columns because several private histories in one
        // public fiber may share sequence variables.
        let push_row =
            |g: &mut Vec<(usize, usize, f64)>, row: &mut usize, cells: &[(usize, f64)]| {
                let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
                for &(col, coef) in cells {
                    *acc.entry(col).or_insert(0.0) += coef;
                }
                for (col, coef) in acc {
                    if coef != 0.0 {
                        g.push((*row, col, coef));
                    }
                }
                *row += 1;
            };
        for (a, &(l, u)) in bounds.iter().enumerate().take(n_actions) {
            if l > 0.0 {
                let cells: Vec<(usize, f64)> = members
                    .iter()
                    .flat_map(|m| {
                        let w = weight_of(m);
                        [(m.parent_seq, l * w), (m.children[a].1, -w)]
                    })
                    .collect();
                push_row(&mut g_entries, &mut nrows, &cells);
                row_meta.push((public_key.clone(), a));
            }
            if u < 1.0 {
                let cells: Vec<(usize, f64)> = members
                    .iter()
                    .flat_map(|m| {
                        let w = weight_of(m);
                        [(m.children[a].1, w), (m.parent_seq, -u * w)]
                    })
                    .collect();
                push_row(&mut g_entries, &mut nrows, &cells);
                row_meta.push((public_key.clone(), a));
            }
        }
    }
    let h = vec![0.0; nrows];
    ConfidenceSet {
        ncols: sf.num_sequences(),
        nrows,
        g_entries,
        h,
        row_meta,
    }
}

/// Compute weights for opponent reach.
pub fn opponent_reach_weights<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    x_agent: &[f64],
) -> HashMap<String, f64> {
    let by_label: HashMap<&str, &InfoSet> = sf0
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), i))
        .collect();
    let mut omega: HashMap<String, f64> = HashMap::new();
    let root = game.root();
    let mut ctx = ReachWalk {
        game,
        by_label: &by_label,
        x_agent,
        omega: &mut omega,
    };
    ctx.walk(&root, 0, 1.0);
    omega
}

/// Stores state for reach walk.
struct ReachWalk<'a, G: Game> {
    game: &'a G,
    by_label: &'a HashMap<&'a str, &'a InfoSet>,
    x_agent: &'a [f64],
    omega: &'a mut HashMap<String, f64>,
}

/// Implements operations for `ReachWalk<'_, G>`.
impl<G: Game> ReachWalk<'_, G> {
    /// Traverse the game tree while accumulating reach contributions.
    fn walk(&mut self, state: &G::State, agent_seq: usize, chance: f64) {
        match self.game.node(state) {
            Node::Terminal(_) => {}
            Node::Chance(outcomes) => {
                for (p, next) in &outcomes {
                    self.walk(next, agent_seq, chance * p);
                }
            }
            Node::Decision {
                player,
                infoset,
                actions,
            } => {
                if player == 0 {
                    let info = self.by_label[infoset.as_str()];
                    for (ch, next) in &actions {
                        let child = info
                            .children
                            .iter()
                            .find(|(c, _)| c == ch)
                            .expect("agent action not in info-set children")
                            .1;
                        self.walk(next, child, chance);
                    }
                } else {
                    // Reach excludes opponent behavior: it is the observable
                    // coefficient induced by chance and the deployed agent.
                    *self.omega.entry(infoset.clone()).or_insert(0.0) +=
                        chance * self.x_agent[agent_seq];
                    for (_ch, next) in &actions {
                        self.walk(next, agent_seq, chance);
                    }
                }
            }
        }
    }
}

/// Computes agent showdown reach.
pub fn agent_showdown_reach<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    x_agent: &[f64],
) -> HashMap<String, Vec<(f64, bool)>> {
    let by_label: HashMap<&str, &InfoSet> = sf0
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), i))
        .collect();
    let mut sd: HashMap<String, Vec<f64>> = HashMap::new();
    let mut deeper: HashMap<String, Vec<bool>> = HashMap::new();
    let mut is_last: HashMap<String, Vec<bool>> = HashMap::new();
    let root = game.root();
    let mut path: Vec<(String, usize)> = Vec::new();
    let mut ctx = SdWalk {
        game,
        by_label: &by_label,
        x_agent,
        sd: &mut sd,
        deeper: &mut deeper,
        is_last: &mut is_last,
    };
    ctx.walk(&root, 0, 1.0, false, &mut path);
    let mut out: HashMap<String, Vec<(f64, bool)>> = HashMap::new();
    for (label, masses) in &sd {
        let dv = &deeper[label];
        let lv = &is_last[label];
        let row = masses
            .iter()
            .enumerate()
            .map(|(a, &w)| (w, lv[a] && !dv[a]))
            .collect();
        out.insert(label.clone(), row);
    }
    out
}

/// Stores state for sd walk.
struct SdWalk<'a, G: Game> {
    game: &'a G,
    by_label: &'a HashMap<&'a str, &'a InfoSet>,
    x_agent: &'a [f64],
    sd: &'a mut HashMap<String, Vec<f64>>,
    deeper: &'a mut HashMap<String, Vec<bool>>,
    is_last: &'a mut HashMap<String, Vec<bool>>,
}

/// Implements operations for `SdWalk<'_, G>`.
impl<G: Game> SdWalk<'_, G> {
    /// Return an existing entry or create it when absent.
    fn ensure(&mut self, label: &str, n: usize) {
        self.sd
            .entry(label.to_string())
            .or_insert_with(|| vec![0.0; n]);
        self.deeper
            .entry(label.to_string())
            .or_insert_with(|| vec![false; n]);
        self.is_last
            .entry(label.to_string())
            .or_insert_with(|| vec![false; n]);
    }

    /// Traverse the game tree while accumulating reach contributions.
    fn walk(
        &mut self,
        state: &G::State,
        agent_seq: usize,
        chance: f64,
        folded: bool,
        path: &mut Vec<(String, usize)>,
    ) {
        match self.game.node(state) {
            Node::Terminal(_) => {
                if folded {
                    return;
                }
                if let Some((last_label, last_a)) = path.last() {
                    // A non-fold terminal reveals the pending opponent path.
                    // Direct mass belongs to the closing action; earlier
                    // actions are flagged as having a deeper reveal.
                    let add = chance * self.x_agent[agent_seq];
                    self.sd.get_mut(last_label).unwrap()[*last_a] += add;
                    self.is_last.get_mut(last_label).unwrap()[*last_a] = true;
                    for (lbl, ai) in &path[..path.len() - 1] {
                        self.deeper.get_mut(lbl).unwrap()[*ai] = true;
                    }
                }
            }
            Node::Chance(outcomes) => {
                for (p, next) in &outcomes {
                    self.walk(next, agent_seq, chance * p, folded, path);
                }
            }
            Node::Decision {
                player,
                infoset,
                actions,
            } => {
                if player == 0 {
                    let info = self.by_label[infoset.as_str()];
                    for (ch, next) in &actions {
                        let child = info
                            .children
                            .iter()
                            .find(|(c, _)| c == ch)
                            .expect("agent action not in info-set children")
                            .1;
                        self.walk(next, child, chance, folded || *ch == 'f', path);
                    }
                } else {
                    self.ensure(&infoset, actions.len());
                    for (a, (ch, next)) in actions.iter().enumerate() {
                        path.push((infoset.clone(), a));
                        self.walk(next, agent_seq, chance, folded || *ch == 'f', path);
                        path.pop();
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

    /// Computes uniform intervals.
    fn uniform_intervals(half: f64) -> HashMap<String, Vec<(f64, f64)>> {
        let sf = compile_kuhn(1);
        let mut m = HashMap::new();
        for info in &sf.info_sets {
            m.insert(
                info.label.clone(),
                vec![(0.5 - half, 0.5 + half); info.children.len()],
            );
        }
        m
    }

    #[test]
    /// Verifies that row meta matches rows and points to active intervals.
    fn row_meta_matches_rows_and_points_to_active_intervals() {
        let sf = compile_kuhn(1);
        let cs = build(&sf, &uniform_intervals(0.1));
        assert_eq!(cs.row_meta.len(), cs.nrows);
        assert!(cs.nrows > 0);
        for (label, action) in &cs.row_meta {
            let info = sf
                .info_sets
                .iter()
                .find(|i| &i.label == label)
                .expect("row_meta label is a real info set");
            assert!(*action < info.children.len());
        }
    }

    #[test]
    /// Verifies that empty intervals give no rows.
    fn empty_intervals_give_no_rows() {
        let cs = build_kuhn(&HashMap::new());
        assert_eq!(cs.nrows, 0);
        assert_eq!(cs.ncols, 13);

        let sf = compile_kuhn(1);
        let y = sf.realization_from_behavior(&HashMap::new());
        assert!(cs.max_violation(&y) <= 0.0);
    }

    #[test]
    /// Verifies that wide intervals are vacuous.
    fn wide_intervals_are_vacuous() {
        let mut m = HashMap::new();
        let sf = compile_kuhn(1);
        for info in &sf.info_sets {
            m.insert(info.label.clone(), vec![(0.0, 1.0); info.children.len()]);
        }
        assert_eq!(build_kuhn(&m).nrows, 0);
    }

    #[test]
    /// Verifies that contains true opponent.
    fn contains_true_opponent() {
        let cs = build_kuhn(&uniform_intervals(0.1));
        assert!(cs.nrows > 0);
        let sf = compile_kuhn(1);
        let y = sf.realization_from_behavior(&HashMap::new());
        assert!(
            cs.max_violation(&y) <= 1e-12,
            "violation = {}",
            cs.max_violation(&y)
        );
    }

    #[test]
    /// Verifies that excludes inconsistent plan.
    fn excludes_inconsistent_plan() {
        let cs = build_kuhn(&uniform_intervals(0.1));
        let sf = compile_kuhn(1);
        let mut behavior = HashMap::new();
        for info in &sf.info_sets {
            behavior.insert(info.label.clone(), vec![1.0, 0.0]);
        }
        let y = sf.realization_from_behavior(&behavior);

        assert!(
            (cs.max_violation(&y) - 0.4).abs() < 1e-12,
            "violation = {}",
            cs.max_violation(&y)
        );
    }

    #[test]
    /// Verifies that rhs is zero.
    fn rhs_is_zero() {
        let cs = build_kuhn(&uniform_intervals(0.2));
        assert!(cs.h.iter().all(|&v| v == 0.0));
        assert_eq!(cs.h.len(), cs.nrows);
    }

    #[test]
    /// Verifies that public set contains aggregate consistent opponent.
    fn public_set_contains_aggregate_consistent_opponent() {
        let sf = compile_kuhn(1);
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for info in &sf.info_sets {
            let history = info.label.split(':').nth(1).unwrap_or("");
            groups
                .entry(format!(":{history}"))
                .or_default()
                .push(info.label.clone());
        }

        let mut intervals = HashMap::new();
        for key in groups.keys() {
            intervals.insert(key.clone(), vec![(0.4, 0.6), (0.4, 0.6)]);
        }
        let cs = build_public(&sf, &groups, &intervals, &HashMap::new());
        assert!(cs.nrows > 0);

        let y = sf.realization_from_behavior(&HashMap::new());
        assert!(cs.max_violation(&y) <= 1e-12);
    }

    #[test]
    /// Verifies that public set is looser than full for offsetting leaks.
    fn public_set_is_looser_than_full_for_offsetting_leaks() {
        let sf = compile_kuhn(1);
        let mut behavior = HashMap::new();
        for info in &sf.info_sets {
            let card = info.label.chars().next().unwrap();
            let bet = match card {
                '0' => 0.0,
                '2' => 1.0,
                _ => 0.5,
            };
            behavior.insert(info.label.clone(), vec![1.0 - bet, bet]);
        }
        let y = sf.realization_from_behavior(&behavior);

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for info in &sf.info_sets {
            let history = info.label.split(':').nth(1).unwrap_or("");
            groups
                .entry(format!(":{history}"))
                .or_default()
                .push(info.label.clone());
        }
        let mut intervals = HashMap::new();
        for key in groups.keys() {
            intervals.insert(key.clone(), vec![(0.4, 0.6), (0.4, 0.6)]);
        }

        let public = build_public(&sf, &groups, &intervals, &HashMap::new());
        assert!(public.max_violation(&y) <= 1e-9);

        let mut full_intervals = HashMap::new();
        for info in &sf.info_sets {
            full_intervals.insert(info.label.clone(), vec![(0.4, 0.6), (0.4, 0.6)]);
        }
        let full = build(&sf, &full_intervals);
        assert!(full.max_violation(&y) > 0.1);
    }

    #[test]
    /// Verifies that weighted public aggregate changes the constraint.
    fn weighted_public_aggregate_changes_the_constraint() {
        let sf = compile_kuhn(1);
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for info in &sf.info_sets {
            let history = info.label.split(':').nth(1).unwrap_or("");
            groups
                .entry(format!(":{history}"))
                .or_default()
                .push(info.label.clone());
        }
        let mut intervals = HashMap::new();
        for key in groups.keys() {
            intervals.insert(key.clone(), vec![(0.4, 0.6), (0.4, 0.6)]);
        }

        let unweighted = build_public(&sf, &groups, &intervals, &HashMap::new());

        let mut uniform = HashMap::new();
        for info in &sf.info_sets {
            uniform.insert(info.label.clone(), 1.0);
        }
        let same = build_public(&sf, &groups, &intervals, &uniform);
        assert_eq!(unweighted.g_entries, same.g_entries);

        let multi = groups
            .values()
            .find(|m| m.len() >= 2)
            .expect("a multi-member public class");
        let mut skew = HashMap::new();
        skew.insert(multi[0].clone(), 5.0);
        let weighted = build_public(&sf, &groups, &intervals, &skew);
        assert_eq!(weighted.nrows, unweighted.nrows);
        assert_ne!(weighted.g_entries, unweighted.g_entries);
    }

    #[test]
    /// Verifies that opponent reach weights are positive under a mixed agent.
    fn opponent_reach_weights_are_positive_under_a_mixed_agent() {
        use crate::kuhn::Kuhn;

        let sf0 = compile_kuhn(0);
        let x = sf0.realization_from_behavior(&HashMap::new());
        let omega = opponent_reach_weights(&Kuhn, &sf0, &x);
        let sf1 = compile_kuhn(1);
        for info in &sf1.info_sets {
            let w = omega.get(&info.label).copied().unwrap_or(0.0);
            assert!(
                w > 0.0,
                "info set {} has non-positive reach weight",
                info.label
            );
        }
    }
}
