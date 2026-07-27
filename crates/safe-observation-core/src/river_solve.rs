//! River solve algorithms for safe observation. See The Safe Observation-Capacity Frontier, Certified Value Recovery, and supplementary Certification at the Unbucketed River.

use std::collections::HashMap;

use crate::river_range::{PubNode, RangeGame};

#[derive(Clone, Copy, Debug)]
/// Enumerates the supported variant variants.
pub enum Variant {
    CfrPlus,

    Dcfr { alpha: f64, beta: f64, gamma: f64 },

    PcfrPlus,
}

/// Implements operations for `Variant`.
impl Variant {
    /// Computes dcfr.
    pub fn dcfr() -> Self {
        Variant::Dcfr {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        }
    }
}

/// Stores state for slot.
struct Slot {
    player: usize,
    offset: usize,
    n_actions: usize,
}

/// Stores state for range CFR.
pub struct RangeCfr<'a> {
    rg: &'a RangeGame<'a>,
    variant: Variant,

    n: [usize; 2],

    slots: Vec<Option<Slot>>,

    regret: [Vec<f64>; 2],

    pred: [Vec<f64>; 2],

    avg: [Vec<f64>; 2],

    t: u64,
}

/// Implements operations for `RangeCfr<'a>`.
impl<'a> RangeCfr<'a> {
    /// Constructs a new value from the supplied configuration.
    pub fn new(rg: &'a RangeGame<'a>, variant: Variant) -> Self {
        let n = [rg.game.range(0).len(), rg.game.range(1).len()];
        let mut sizes = [0usize; 2];
        let slots = rg
            .tree
            .nodes
            .iter()
            .map(|node| match node {
                PubNode::Decision {
                    player, children, ..
                } => {
                    let slot = Slot {
                        player: *player,
                        offset: sizes[*player],
                        n_actions: children.len(),
                    };
                    sizes[*player] += n[*player] * children.len();
                    Some(slot)
                }
                _ => None,
            })
            .collect();
        Self {
            rg,
            variant,
            n,
            slots,
            regret: [vec![0.0; sizes[0]], vec![0.0; sizes[1]]],
            pred: [vec![0.0; sizes[0]], vec![0.0; sizes[1]]],
            avg: [vec![0.0; sizes[0]], vec![0.0; sizes[1]]],
            t: 0,
        }
    }

    /// Computes iterations.
    pub fn iterations(&self) -> u64 {
        self.t
    }

    /// Computes sigma into.
    fn sigma_into(&self, p: usize, slot: &Slot, k: usize, out: &mut [f64]) {
        let m = slot.n_actions;
        let base = slot.offset + k * m;
        let mut sum = 0.0;
        for (a, o) in out.iter_mut().enumerate().take(m) {
            let mut r = self.regret[p][base + a];
            if matches!(self.variant, Variant::PcfrPlus) {
                r += self.pred[p][base + a];
            }
            let r = r.max(0.0);
            *o = r;
            sum += r;
        }
        if sum > 0.0 {
            for o in out.iter_mut().take(m) {
                *o /= sum;
            }
        } else {
            for o in out.iter_mut().take(m) {
                *o = 1.0 / m as f64;
            }
        }
    }

    /// Computes iterate.
    pub fn iterate(&mut self) {
        self.t += 1;
        let w_avg = match self.variant {
            Variant::CfrPlus => self.t as f64,
            Variant::Dcfr { .. } => 1.0,
            Variant::PcfrPlus => (self.t as f64) * (self.t as f64),
        };
        for my in 0..2 {
            let my_reach = vec![1.0; self.n[my]];
            let opp_reach = vec![1.0; self.n[1 - my]];
            self.walk(my, 0, &my_reach, &opp_reach, w_avg);
        }
        if let Variant::Dcfr { alpha, beta, gamma } = self.variant {
            let t = self.t as f64;
            let dp = t.powf(alpha) / (t.powf(alpha) + 1.0);
            let dn = t.powf(beta) / (t.powf(beta) + 1.0);
            let dg = (t / (t + 1.0)).powf(gamma);
            for p in 0..2 {
                for r in self.regret[p].iter_mut() {
                    *r *= if *r > 0.0 { dp } else { dn };
                }
                for a in self.avg[p].iter_mut() {
                    *a *= dg;
                }
            }
        }
    }

    /// Traverse the game tree while accumulating reach contributions.
    fn walk(
        &mut self,
        my: usize,
        id: usize,
        my_reach: &[f64],
        opp_reach: &[f64],
        w_avg: f64,
    ) -> Vec<f64> {
        let rg = self.rg;
        let node = &rg.tree.nodes[id];
        let (player, children) = match node {
            PubNode::Decision {
                player, children, ..
            } => (*player, children),
            terminal => return rg.terminal_values(my, terminal, opp_reach),
        };
        let m = children.len();
        let slot = self.slots[id].as_ref().expect("decision slot");
        debug_assert_eq!(slot.player, player);
        let (n_act, offset) = (slot.n_actions, slot.offset);

        let n = self.n[player];
        let mut sig = vec![0.0; n * m];
        {
            let slot = Slot {
                player,
                offset,
                n_actions: n_act,
            };
            let mut buf = vec![0.0; m];
            for k in 0..n {
                self.sigma_into(player, &slot, k, &mut buf);
                sig[k * m..k * m + m].copy_from_slice(&buf);
            }
        }

        if player == my {
            let mut vals: Vec<Vec<f64>> = Vec::with_capacity(m);
            for (ai, &(_a, child)) in children.iter().enumerate() {
                let mut cr = my_reach.to_vec();
                for (k, w) in cr.iter_mut().enumerate() {
                    *w *= sig[k * m + ai];
                }
                vals.push(self.walk(my, child, &cr, opp_reach, w_avg));
            }
            let mut v = vec![0.0; self.n[my]];
            for (k, vk) in v.iter_mut().enumerate() {
                for (ai, va) in vals.iter().enumerate() {
                    *vk += sig[k * m + ai] * va[k];
                }
            }
            for k in 0..self.n[my] {
                for (ai, va) in vals.iter().enumerate() {
                    let idx = offset + k * m + ai;
                    let inst = va[k] - v[k];
                    match self.variant {
                        Variant::CfrPlus => {
                            self.regret[my][idx] = (self.regret[my][idx] + inst).max(0.0);
                        }
                        Variant::PcfrPlus => {
                            self.regret[my][idx] = (self.regret[my][idx] + inst).max(0.0);
                            self.pred[my][idx] = inst;
                        }
                        Variant::Dcfr { .. } => {
                            self.regret[my][idx] += inst;
                        }
                    }
                    self.avg[my][idx] += w_avg * my_reach[k] * sig[k * m + ai];
                }
            }
            v
        } else {
            let mut v = vec![0.0; self.n[my]];
            for (ai, &(_a, child)) in children.iter().enumerate() {
                let mut or = opp_reach.to_vec();
                for (k, w) in or.iter_mut().enumerate() {
                    *w *= sig[k * m + ai];
                }
                let va = self.walk(my, child, my_reach, &or, w_avg);
                for (vk, vak) in v.iter_mut().zip(&va) {
                    *vk += vak;
                }
            }
            v
        }
    }

    /// Computes average behavior.
    pub fn average_behavior(&self, player: usize) -> HashMap<String, Vec<f64>> {
        let mut out = HashMap::new();
        for (id, node) in self.rg.tree.nodes.iter().enumerate() {
            let PubNode::Decision {
                player: p, hist, ..
            } = node
            else {
                continue;
            };
            if *p != player {
                continue;
            }
            let slot = self.slots[id].as_ref().expect("decision slot");
            let m = slot.n_actions;
            for k in 0..self.n[player] {
                let base = slot.offset + k * m;
                let w = &self.avg[player][base..base + m];
                let sum: f64 = w.iter().sum();
                let dist = if sum > 0.0 {
                    w.iter().map(|x| x / sum).collect()
                } else {
                    vec![1.0 / m as f64; m]
                };
                out.insert(self.rg.label(player, k, hist), dist);
            }
        }
        out
    }
}

/// Compute the best response value.
pub fn best_response_value(
    rg: &RangeGame,
    my: usize,
    opp_behavior: &HashMap<String, Vec<f64>>,
) -> f64 {
    let opp_reach = vec![1.0; rg.game.range(1 - my).len()];
    br_walk(rg, my, 0, &opp_reach, opp_behavior).iter().sum()
}

/// Computes br walk.
fn br_walk(
    rg: &RangeGame,
    my: usize,
    id: usize,
    opp_reach: &[f64],
    opp_behavior: &HashMap<String, Vec<f64>>,
) -> Vec<f64> {
    let node = &rg.tree.nodes[id];
    let (player, hist, children) = match node {
        PubNode::Decision {
            player,
            hist,
            children,
        } => (*player, hist, children),
        terminal => return rg.terminal_values(my, terminal, opp_reach),
    };
    let m = children.len();
    if player == my {
        let mut best: Option<Vec<f64>> = None;
        for &(_a, child) in children {
            let va = br_walk(rg, my, child, opp_reach, opp_behavior);
            best = Some(match best {
                None => va,
                Some(mut b) => {
                    for (bk, vk) in b.iter_mut().zip(&va) {
                        if *vk > *bk {
                            *bk = *vk;
                        }
                    }
                    b
                }
            });
        }
        best.expect("decision node with no actions")
    } else {
        let uniform = 1.0 / m as f64;
        let n_opp = opp_reach.len();
        let mut v = vec![0.0; rg.game.range(my).len()];
        for (ai, &(_a, child)) in children.iter().enumerate() {
            let mut or = opp_reach.to_vec();
            for (j, w) in or.iter_mut().enumerate().take(n_opp) {
                let p = opp_behavior
                    .get(&rg.label(player, j, hist))
                    .map_or(uniform, |dist| dist[ai]);
                *w *= p;
            }
            let va = br_walk(rg, my, child, &or, opp_behavior);
            for (vk, vak) in v.iter_mut().zip(&va) {
                *vk += vak;
            }
        }
        v
    }
}

/// Computes exploitability.
pub fn exploitability(
    rg: &RangeGame,
    b0: &HashMap<String, Vec<f64>>,
    b1: &HashMap<String, Vec<f64>>,
) -> f64 {
    best_response_value(rg, 0, b1) + best_response_value(rg, 1, b0)
}

/// Compute the security value.
pub fn security_value(rg: &RangeGame, b0: &HashMap<String, Vec<f64>>) -> f64 {
    -best_response_value(rg, 1, b0)
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;
    use crate::best_response::{best_response_p1_from_matrix, safety_verify_from_matrix};
    use crate::holdem::{build_holdem, canonical_holdem, compile_holdem};
    use crate::river_range::test_util::random_behavior;
    use crate::river_range::RangeGame;

    #[test]
    /// Verifies that range br matches matrix dp on compact river.
    fn range_br_matches_matrix_dp_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = compile_holdem(0);
        let sf1 = compile_holdem(1);
        let pm = build_holdem();
        for seed in [21u64, 22] {
            let b0 = random_behavior(&rg, 0, seed);
            let b1 = random_behavior(&rg, 1, seed.wrapping_add(500));
            let x = sf0.realization_from_behavior(&b0);
            let y = sf1.realization_from_behavior(&b1);

            let br0 = best_response_value(&rg, 0, &b1);
            let exact0 = best_response_p1_from_matrix(&sf0, &pm, &y).value;
            assert!(
                (br0 - exact0).abs() <= 1e-10 * (1.0 + exact0.abs()),
                "seed {seed}: br0 {br0} vs exact {exact0}"
            );

            let vref = security_value(&rg, &b0);
            let exact_vref = safety_verify_from_matrix(&sf1, &pm, &x).value;
            assert!(
                (vref - exact_vref).abs() <= 1e-10 * (1.0 + exact_vref.abs()),
                "seed {seed}: vref {vref} vs exact {exact_vref}"
            );
        }
    }

    #[test]
    /// Verifies that range CFR plus matches linear program blueprint on compact river.
    fn range_cfr_plus_matches_lp_blueprint_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..1500 {
            cfr.iterate();
        }
        let b0 = cfr.average_behavior(0);
        let b1 = cfr.average_behavior(1);
        let expl = exploitability(&rg, &b0, &b1);
        assert!(expl >= -1e-9, "exploitability must be nonnegative: {expl}");
        assert!(expl < 2e-3, "CFR+ exploitability too high: {expl}");

        let lp =
            crate::lp::solve_blueprint(&compile_holdem(0), &compile_holdem(1), &build_holdem());
        let value = rg.bilinear_from_behavior(&b0, &b1);
        assert!(
            (value - lp.value).abs() < 1.5e-3,
            "CFR value {value} vs LP blueprint {}",
            lp.value
        );

        let vref = security_value(&rg, &b0);
        assert!(vref <= lp.value + 1e-9 && vref >= lp.value - expl - 1e-9);
    }

    #[test]
    /// Verifies that dcfr and pcfr variants converge on compact river.
    fn dcfr_and_pcfr_variants_converge_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        for variant in [Variant::dcfr(), Variant::PcfrPlus] {
            let mut cfr = RangeCfr::new(&rg, variant);
            for _ in 0..600 {
                cfr.iterate();
            }
            let expl = exploitability(&rg, &cfr.average_behavior(0), &cfr.average_behavior(1));
            assert!(expl < 2e-2, "{variant:?} exploitability {expl}");
        }
    }

    #[test]
    #[ignore = "full river blueprint timing; run explicitly in release mode"]
    /// Verifies that full river smoke blueprint CFR.
    fn full_river_smoke_blueprint_cfr() {
        use crate::hand_eval::card;
        use crate::holdem::{HoldemRules, RiverEndgame};
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
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        let t0 = Instant::now();
        let mut last = (0.0, 0.0);
        for burst in 0..8 {
            for _ in 0..250 {
                cfr.iterate();
            }
            let b0 = cfr.average_behavior(0);
            let b1 = cfr.average_behavior(1);
            let expl = exploitability(&rg, &b0, &b1);
            let vref = security_value(&rg, &b0);
            last = (expl, vref);
            println!(
                "full river CFR+ iter {:5}  wall {:8.2?}  exploitability {expl:.6}  v_ref {vref:.6}",
                (burst + 1) * 250,
                t0.elapsed(),
            );
        }
        assert!(last.0.is_finite() && last.1.is_finite());
    }
}
