use std::collections::HashMap;

use crate::hand_eval::card_str;
use crate::holdem::{HoldemRules, HoldemState, RiverEndgame};

pub enum PubNode {
    Decision {
        player: usize,
        hist: String,
        children: Vec<(char, usize)>,
    },

    Fold {
        folder: usize,
        committed: [u32; 2],
    },

    Showdown {
        committed: [u32; 2],
    },
}

pub struct PublicTree {
    pub nodes: Vec<PubNode>,
}

impl PublicTree {
    pub fn build(rules: &HoldemRules) -> Self {
        let mut root = rules.root();
        root.s0 = Some(0);
        root.s1 = Some(0);
        let mut nodes = Vec::new();
        Self::build_from(rules, &root, &mut nodes);
        Self { nodes }
    }

    fn build_from(rules: &HoldemRules, s: &HoldemState, nodes: &mut Vec<PubNode>) -> usize {
        if let Some(f) = s.folder {
            let id = nodes.len();
            nodes.push(PubNode::Fold {
                folder: f,
                committed: s.committed,
            });
            return id;
        }
        if s.closed {
            let id = nodes.len();
            nodes.push(PubNode::Showdown {
                committed: s.committed,
            });
            return id;
        }
        let player = s.to_act;
        let hist = s.hist.clone();
        let id = nodes.len();

        nodes.push(PubNode::Showdown { committed: [0, 0] });
        let mut children = Vec::new();
        for (a, next) in rules.legal_actions(s) {
            let child = Self::build_from(rules, &next, nodes);
            children.push((a, child));
        }
        nodes[id] = PubNode::Decision {
            player,
            hist,
            children,
        };
        id
    }

    pub fn num_decisions(&self, player: usize) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, PubNode::Decision { player: p, .. } if *p == player))
            .count()
    }

    pub fn num_action_slots(&self, player: usize) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                PubNode::Decision {
                    player: p,
                    children,
                    ..
                } if *p == player => Some(children.len()),
                _ => None,
            })
            .sum()
    }
}

pub struct RangeGame<'a> {
    pub(crate) game: &'a RiverEndgame,

    pub tree: PublicTree,

    same: [Vec<Option<usize>>; 2],

    order: [Vec<u32>; 2],

    pub(crate) inv_deals: f64,
}

impl<'a> RangeGame<'a> {
    pub fn new(game: &'a RiverEndgame) -> Self {
        let tree = PublicTree::build(&game.rules());
        let same_of = |a: usize, b: usize| -> Vec<Option<usize>> {
            let mut index: HashMap<[u8; 2], usize> = HashMap::new();
            for (j, h) in game.range(b).iter().enumerate() {
                index.insert(*h, j);
            }
            game.range(a)
                .iter()
                .map(|h| index.get(h).copied())
                .collect()
        };
        let order_of = |p: usize| -> Vec<u32> {
            let mut order: Vec<u32> = (0..game.range(p).len() as u32).collect();
            order.sort_by_key(|&i| game.scores(p)[i as usize]);
            order
        };
        Self {
            game,
            tree,
            same: [same_of(0, 1), same_of(1, 0)],
            order: [order_of(0), order_of(1)],
            inv_deals: 1.0 / game.num_deals() as f64,
        }
    }

    pub fn combos(&self, player: usize) -> usize {
        self.game.range(player).len()
    }

    pub fn compatible(&self, i: usize, j: usize) -> bool {
        let a = self.game.range(0)[i];
        let b = self.game.range(1)[j];
        a[0] != b[0] && a[0] != b[1] && a[1] != b[0] && a[1] != b[1]
    }

    pub fn deal_weight(&self) -> f64 {
        self.inv_deals
    }

    pub fn compile_sequence_form(&self, player: usize) -> crate::sequence_form::SequenceForm {
        use crate::sequence_form::InfoSet;
        let n = self.combos(player);
        let mut sequences = vec![String::new()];
        let mut info_sets = Vec::new();

        let mut stack: Vec<(usize, std::rc::Rc<Vec<usize>>)> =
            vec![(0, std::rc::Rc::new(vec![0usize; n]))];
        while let Some((id, cur)) = stack.pop() {
            let PubNode::Decision {
                player: p,
                hist,
                children,
            } = &self.tree.nodes[id]
            else {
                continue;
            };
            if *p != player {
                for &(_a, child) in children {
                    stack.push((child, cur.clone()));
                }
                continue;
            }

            let mut per_child: Vec<Vec<usize>> = vec![vec![0; n]; children.len()];
            for k in 0..n {
                let label = self.label(player, k, hist);
                let kids: Vec<(char, usize)> = children
                    .iter()
                    .enumerate()
                    .map(|(ai, &(a, _))| {
                        let idx = sequences.len();
                        sequences.push(format!("{label}>{a}"));
                        per_child[ai][k] = idx;
                        (a, idx)
                    })
                    .collect();
                info_sets.push(InfoSet {
                    label,
                    parent_seq: cur[k],
                    children: kids,
                });
            }
            for (ai, &(_a, child)) in children.iter().enumerate() {
                stack.push((child, std::rc::Rc::new(std::mem::take(&mut per_child[ai]))));
            }
        }
        crate::sequence_form::SequenceForm::from_parts(player, sequences, info_sets)
    }

    pub fn label(&self, player: usize, k: usize, hist: &str) -> String {
        let h = self.game.range(player)[k];
        format!("{}{}|{}", card_str(h[0]), card_str(h[1]), hist)
    }

    pub fn bilinear_from_behavior(
        &self,
        b0: &HashMap<String, Vec<f64>>,
        b1: &HashMap<String, Vec<f64>>,
    ) -> f64 {
        let r0 = vec![1.0; self.game.range(0).len()];
        let r1 = vec![1.0; self.game.range(1).len()];
        self.eval_node(0, &r0, &r1, b0, b1)
    }

    fn eval_node(
        &self,
        id: usize,
        r0: &[f64],
        r1: &[f64],
        b0: &HashMap<String, Vec<f64>>,
        b1: &HashMap<String, Vec<f64>>,
    ) -> f64 {
        match &self.tree.nodes[id] {
            PubNode::Fold { folder, committed } => {
                let v = if *folder == 0 {
                    -(committed[0] as f64)
                } else {
                    committed[1] as f64
                };
                v * self.inv_deals * self.compat_sum(r0, r1)
            }
            PubNode::Showdown { committed } => {
                self.inv_deals * self.showdown_sum_naive(r0, r1, *committed)
            }
            PubNode::Decision {
                player,
                hist,
                children,
            } => {
                let (behavior, reach) = if *player == 0 { (b0, r0) } else { (b1, r1) };
                let n = reach.len();
                let uniform = 1.0 / children.len() as f64;
                let mut total = 0.0;
                for (ai, &(_a, child)) in children.iter().enumerate() {
                    let mut rc = reach.to_vec();
                    for (k, w) in rc.iter_mut().enumerate() {
                        debug_assert!(k < n);
                        let p = behavior
                            .get(&self.label(*player, k, hist))
                            .map_or(uniform, |dist| dist[ai]);
                        *w *= p;
                    }
                    total += if *player == 0 {
                        self.eval_node(child, &rc, r1, b0, b1)
                    } else {
                        self.eval_node(child, r0, &rc, b0, b1)
                    };
                }
                total
            }
        }
    }

    pub fn compat_sum(&self, r0: &[f64], r1: &[f64]) -> f64 {
        let t0: f64 = r0.iter().sum();
        let t1: f64 = r1.iter().sum();
        let mut a0 = [0.0_f64; 52];
        let mut a1 = [0.0_f64; 52];
        for (k, h) in self.game.range(0).iter().enumerate() {
            a0[h[0] as usize] += r0[k];
            a0[h[1] as usize] += r0[k];
        }
        for (k, h) in self.game.range(1).iter().enumerate() {
            a1[h[0] as usize] += r1[k];
            a1[h[1] as usize] += r1[k];
        }
        let one_shared: f64 = (0..52).map(|c| a0[c] * a1[c]).sum();
        let both_shared: f64 = self.same[0]
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.map(|j| r0[i] * r1[j]))
            .sum();
        t0 * t1 - one_shared + both_shared
    }

    pub fn fold_masses(&self, my: usize, opp_reach: &[f64]) -> Vec<f64> {
        let opp = 1 - my;
        let t: f64 = opp_reach.iter().sum();
        let mut c = [0.0_f64; 52];
        for (j, h) in self.game.range(opp).iter().enumerate() {
            c[h[0] as usize] += opp_reach[j];
            c[h[1] as usize] += opp_reach[j];
        }
        self.game
            .range(my)
            .iter()
            .zip(&self.same[my])
            .map(|(h, same)| {
                let back = same.map_or(0.0, |j| opp_reach[j]);
                t - c[h[0] as usize] - c[h[1] as usize] + back
            })
            .collect()
    }

    pub fn showdown_masses(&self, my: usize, opp_reach: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let opp = 1 - my;
        let my_scores = self.game.scores(my);
        let opp_scores = self.game.scores(opp);
        let my_range = self.game.range(my);
        let opp_range = self.game.range(opp);
        let ord_my = &self.order[my];
        let ord_opp = &self.order[opp];
        let n_my = my_range.len();
        let n_opp = opp_range.len();

        let mut below = vec![0.0_f64; n_my];
        let mut above = vec![0.0_f64; n_my];

        let mut t = 0.0;
        let mut c = [0.0_f64; 52];
        let mut jp = 0;
        let mut ip = 0;
        while ip < n_my {
            let s = my_scores[ord_my[ip] as usize];
            while jp < n_opp && opp_scores[ord_opp[jp] as usize] < s {
                let j = ord_opp[jp] as usize;
                let w = opp_reach[j];
                t += w;
                c[opp_range[j][0] as usize] += w;
                c[opp_range[j][1] as usize] += w;
                jp += 1;
            }
            while ip < n_my && my_scores[ord_my[ip] as usize] == s {
                let i = ord_my[ip] as usize;
                let h = my_range[i];
                below[i] = t - c[h[0] as usize] - c[h[1] as usize];
                ip += 1;
            }
        }

        let mut t = 0.0;
        let mut c = [0.0_f64; 52];
        let mut jp = n_opp;
        let mut ip = n_my;
        while ip > 0 {
            let s = my_scores[ord_my[ip - 1] as usize];
            while jp > 0 && opp_scores[ord_opp[jp - 1] as usize] > s {
                let j = ord_opp[jp - 1] as usize;
                let w = opp_reach[j];
                t += w;
                c[opp_range[j][0] as usize] += w;
                c[opp_range[j][1] as usize] += w;
                jp -= 1;
            }
            while ip > 0 && my_scores[ord_my[ip - 1] as usize] == s {
                let i = ord_my[ip - 1] as usize;
                let h = my_range[i];
                above[i] = t - c[h[0] as usize] - c[h[1] as usize];
                ip -= 1;
            }
        }
        (below, above)
    }

    pub fn terminal_values(&self, my: usize, node: &PubNode, opp_reach: &[f64]) -> Vec<f64> {
        match node {
            PubNode::Fold { folder, committed } => {
                let v = if *folder == my {
                    -(committed[my] as f64)
                } else {
                    committed[*folder] as f64
                };
                let mut mass = self.fold_masses(my, opp_reach);
                for m in mass.iter_mut() {
                    *m *= v * self.inv_deals;
                }
                mass
            }
            PubNode::Showdown { committed } => {
                let win = committed[1 - my] as f64;
                let lose = -(committed[my] as f64);
                let (below, above) = self.showdown_masses(my, opp_reach);
                below
                    .iter()
                    .zip(&above)
                    .map(|(b, a)| self.inv_deals * (win * b + lose * a))
                    .collect()
            }
            PubNode::Decision { .. } => unreachable!("terminal_values on a decision node"),
        }
    }

    fn showdown_sum_naive(&self, r0: &[f64], r1: &[f64], committed: [u32; 2]) -> f64 {
        let range0 = self.game.range(0);
        let range1 = self.game.range(1);
        let s0 = self.game.scores(0);
        let s1 = self.game.scores(1);
        let win = committed[1] as f64;
        let lose = -(committed[0] as f64);
        let mut acc = 0.0;
        for (i, h0) in range0.iter().enumerate() {
            let w0 = r0[i];
            if w0 == 0.0 {
                continue;
            }
            for (j, h1) in range1.iter().enumerate() {
                if h0[0] == h1[0] || h0[0] == h1[1] || h0[1] == h1[0] || h0[1] == h1[1] {
                    continue;
                }
                let v = match s0[i].cmp(&s1[j]) {
                    std::cmp::Ordering::Greater => win,
                    std::cmp::Ordering::Less => lose,
                    std::cmp::Ordering::Equal => continue,
                };
                acc += w0 * r1[j] * v;
            }
        }
        acc
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    pub struct SplitMix64(pub u64);
    impl SplitMix64 {
        pub fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z = z ^ (z >> 31);
            (z >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    pub fn random_behavior(rg: &RangeGame, player: usize, seed: u64) -> HashMap<String, Vec<f64>> {
        let mut rng = SplitMix64(seed);
        let mut out = HashMap::new();
        let n = rg.game.range(player).len();
        for node in &rg.tree.nodes {
            if let PubNode::Decision {
                player: p,
                hist,
                children,
            } = node
            {
                if *p != player {
                    continue;
                }
                for k in 0..n {
                    let mut dist: Vec<f64> =
                        (0..children.len()).map(|_| 0.1 + rng.next_f64()).collect();
                    let s: f64 = dist.iter().sum();
                    dist.iter_mut().for_each(|w| *w /= s);
                    out.insert(rg.label(player, k, hist), dist);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::{random_behavior, SplitMix64};
    use super::*;
    use crate::holdem::{build_holdem, canonical_holdem, compile_holdem};

    #[test]
    fn public_tree_matches_sequence_form_sizes_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        for player in 0..2 {
            let sf = compile_holdem(player);
            assert_eq!(sf.num_infosets(), 45 * rg.tree.num_decisions(player));
            assert_eq!(
                sf.num_sequences(),
                1 + 45 * rg.tree.num_action_slots(player)
            );
        }
    }

    #[test]
    fn labels_resolve_in_sequence_form_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sfs = [compile_holdem(0), compile_holdem(1)];
        for node in &rg.tree.nodes {
            if let PubNode::Decision {
                player,
                hist,
                children,
            } = node
            {
                let (a, _) = children[0];
                for k in 0..rg.game.range(*player).len() {
                    let seq = format!("{}>{}", rg.label(*player, k, hist), a);
                    assert!(
                        sfs[*player].sequence_index(&seq).is_some(),
                        "unresolved sequence label {seq}"
                    );
                }
            }
        }
    }

    #[test]
    fn bilinear_matches_exact_payoff_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = compile_holdem(0);
        let sf1 = compile_holdem(1);
        let pm = build_holdem();
        for seed in [1u64, 2, 3] {
            let b0 = random_behavior(&rg, 0, seed);
            let b1 = random_behavior(&rg, 1, seed.wrapping_add(1000));
            let x = sf0.realization_from_behavior(&b0);
            let y = sf1.realization_from_behavior(&b1);
            let exact = pm.bilinear(&x, &y);
            let fast = rg.bilinear_from_behavior(&b0, &b1);
            assert!(
                (exact - fast).abs() <= 1e-10 * (1.0 + exact.abs()),
                "seed {seed}: exact {exact} vs range {fast}"
            );
        }
    }

    #[test]
    #[ignore = "full river smoke timing; run explicitly in release mode"]
    fn full_river_smoke_full_river_bilinear() {
        use crate::hand_eval::card;
        use crate::holdem::RiverEndgame;
        use std::time::Instant;

        let board = [
            card(12, 3),
            card(11, 3),
            card(10, 1),
            card(9, 0),
            card(7, 2),
        ];
        let t = Instant::now();
        let game = RiverEndgame::full(crate::holdem::HoldemRules::river_small(), board);
        let rg = RangeGame::new(&game);
        let build = t.elapsed();
        let b0 = random_behavior(&rg, 0, 42);
        let b1 = random_behavior(&rg, 1, 43);
        let t = Instant::now();
        let v = rg.bilinear_from_behavior(&b0, &b1);
        let eval = t.elapsed();
        println!(
            "full river smoke: combos=({}, {}) deals={} public_nodes={} decisions=({}, {}) \
             build={build:?} eval={eval:?} value={v:.6}",
            game.range(0).len(),
            game.range(1).len(),
            game.num_deals(),
            rg.tree.nodes.len(),
            rg.tree.num_decisions(0),
            rg.tree.num_decisions(1),
        );
        assert!(v.is_finite());
    }

    #[test]
    fn showdown_masses_match_naive_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let mut rng = SplitMix64(11);
        for my in 0..2 {
            let opp = 1 - my;
            let n_my = game.range(my).len();
            let n_opp = game.range(opp).len();
            let opp_reach: Vec<f64> = (0..n_opp).map(|_| rng.next_f64()).collect();
            let (below, above) = rg.showdown_masses(my, &opp_reach);
            for i in 0..n_my {
                let h0 = game.range(my)[i];
                let (mut nb, mut na) = (0.0, 0.0);
                for (j, h1) in game.range(opp).iter().enumerate() {
                    if h0[0] == h1[0] || h0[0] == h1[1] || h0[1] == h1[0] || h0[1] == h1[1] {
                        continue;
                    }
                    match game.scores(opp)[j].cmp(&game.scores(my)[i]) {
                        std::cmp::Ordering::Less => nb += opp_reach[j],
                        std::cmp::Ordering::Greater => na += opp_reach[j],
                        std::cmp::Ordering::Equal => {}
                    }
                }
                assert!(
                    (below[i] - nb).abs() <= 1e-12 * (1.0 + nb.abs()),
                    "below[{i}] seat {my}"
                );
                assert!(
                    (above[i] - na).abs() <= 1e-12 * (1.0 + na.abs()),
                    "above[{i}] seat {my}"
                );
            }
        }
    }

    #[test]
    fn fold_masses_match_naive_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let mut rng = SplitMix64(13);
        for my in 0..2 {
            let opp = 1 - my;
            let opp_reach: Vec<f64> = (0..game.range(opp).len()).map(|_| rng.next_f64()).collect();
            let masses = rg.fold_masses(my, &opp_reach);
            for (i, h0) in game.range(my).iter().enumerate() {
                let mut naive = 0.0;
                for (j, h1) in game.range(opp).iter().enumerate() {
                    if h0[0] != h1[0] && h0[0] != h1[1] && h0[1] != h1[0] && h0[1] != h1[1] {
                        naive += opp_reach[j];
                    }
                }
                assert!((masses[i] - naive).abs() <= 1e-12 * (1.0 + naive.abs()));
            }
        }
    }

    #[test]
    fn direct_sequence_form_matches_generic_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        for player in 0..2 {
            let fast = rg.compile_sequence_form(player);
            let slow = compile_holdem(player);
            assert_eq!(fast.num_sequences(), slow.num_sequences());
            assert_eq!(fast.num_infosets(), slow.num_infosets());
            for label in &slow.sequences {
                assert!(
                    fast.sequence_index(label).is_some(),
                    "generic sequence {label} missing from the direct compiler"
                );
            }
            let b = random_behavior(&rg, player, 91 + player as u64);
            let x = fast.realization_from_behavior(&b);
            assert!(fast.constraint_residual(&x) < 1e-12);
        }
    }

    #[test]
    fn compat_fast_path_matches_naive_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let mut rng = SplitMix64(7);
        let r0: Vec<f64> = (0..game.range(0).len()).map(|_| rng.next_f64()).collect();
        let r1: Vec<f64> = (0..game.range(1).len()).map(|_| rng.next_f64()).collect();
        let mut naive = 0.0;
        for (i, h0) in game.range(0).iter().enumerate() {
            for (j, h1) in game.range(1).iter().enumerate() {
                if h0[0] != h1[0] && h0[0] != h1[1] && h0[1] != h1[0] && h0[1] != h1[1] {
                    naive += r0[i] * r1[j];
                }
            }
        }
        let fast = rg.compat_sum(&r0, &r1);
        assert!((naive - fast).abs() <= 1e-12 * (1.0 + naive.abs()));
    }
}
