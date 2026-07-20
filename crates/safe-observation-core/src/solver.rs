use std::collections::HashMap;

use crate::best_response::safety_verify_tree;
use crate::cfr::{normalize, regret_matching};
use crate::game::{Game, Node};
use crate::sequence_form::SequenceForm;

struct InfoNode {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

pub struct CfrBlueprint {
    pub realization: Vec<f64>,

    pub avg_behavior0: HashMap<String, Vec<f64>>,
    pub avg_behavior1: HashMap<String, Vec<f64>>,

    pub value: f64,
}

fn cfr_walk<G: Game>(
    game: &G,
    state: &G::State,
    nodes: &mut HashMap<String, InfoNode>,
    reach0: f64,
    reach1: f64,
    chance: f64,
) -> f64 {
    match game.node(state) {
        Node::Terminal(value) => value,
        Node::Chance(outcomes) => {
            let mut value = 0.0;
            for (p, next) in &outcomes {
                value += p * cfr_walk(game, next, nodes, reach0, reach1, chance * p);
            }
            value
        }
        Node::Decision {
            player,
            infoset,
            actions,
        } => {
            let n = actions.len();

            let strat = {
                let node = nodes.entry(infoset.clone()).or_insert_with(|| InfoNode {
                    regret_sum: vec![0.0; n],
                    strategy_sum: vec![0.0; n],
                });
                let s = regret_matching(&node.regret_sum);
                let own_reach = if player == 0 { reach0 } else { reach1 };
                for (slot, &sv) in node.strategy_sum.iter_mut().zip(&s) {
                    *slot += own_reach * sv;
                }
                s
            };

            let mut child_val = vec![0.0; n];
            let mut node_val = 0.0;
            for (a, (_sym, next)) in actions.iter().enumerate() {
                let cv = if player == 0 {
                    cfr_walk(game, next, nodes, reach0 * strat[a], reach1, chance)
                } else {
                    cfr_walk(game, next, nodes, reach0, reach1 * strat[a], chance)
                };
                child_val[a] = cv;
                node_val += strat[a] * cv;
            }

            let cf_reach = if player == 0 {
                chance * reach1
            } else {
                chance * reach0
            };
            let sign = if player == 0 { 1.0 } else { -1.0 };
            let node = nodes.get_mut(&infoset).expect("node was just inserted");
            for (slot, &cv) in node.regret_sum.iter_mut().zip(&child_val) {
                *slot += sign * cf_reach * (cv - node_val);
            }
            node_val
        }
    }
}

fn average_behavior(
    sf: &SequenceForm,
    nodes: &HashMap<String, InfoNode>,
) -> HashMap<String, Vec<f64>> {
    let mut out = HashMap::with_capacity(sf.info_sets.len());
    for info in &sf.info_sets {
        if let Some(node) = nodes.get(&info.label) {
            out.insert(info.label.clone(), normalize(&node.strategy_sum));
        }
    }
    out
}

pub fn solve_blueprint_cfr<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    iterations: u64,
) -> CfrBlueprint {
    let mut nodes: HashMap<String, InfoNode> = HashMap::new();
    let root = game.root();
    for _ in 0..iterations {
        cfr_walk(game, &root, &mut nodes, 1.0, 1.0, 1.0);
    }

    let avg_behavior0 = average_behavior(sf0, &nodes);
    let avg_behavior1 = average_behavior(sf1, &nodes);
    let realization = sf0.realization_from_behavior(&avg_behavior0);
    let value = safety_verify_tree(game, sf0, sf1, &realization).value;

    CfrBlueprint {
        realization,
        avg_behavior0,
        avg_behavior1,
        value,
    }
}

struct EsRng {
    state: u64,
}

impl EsRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn sample(&mut self, probs: &[f64]) -> usize {
        let u = self.next_f64();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                return i;
            }
        }
        probs.len() - 1
    }
}

fn external_sampling<G: Game>(
    game: &G,
    state: &G::State,
    traverser: usize,
    nodes: &mut HashMap<String, InfoNode>,
    rng: &mut EsRng,
    iter_weight: f64,
) -> f64 {
    let sampled = {
        let mut draw = || rng.next_f64();
        game.sample_chance(state, &mut draw)
    };
    if let Some(next) = sampled {
        return external_sampling(game, &next, traverser, nodes, rng, iter_weight);
    }
    match game.node(state) {
        Node::Terminal(value) => {
            if traverser == 0 {
                value
            } else {
                -value
            }
        }
        Node::Chance(outcomes) => {
            let probs: Vec<f64> = outcomes.iter().map(|(p, _)| *p).collect();
            let idx = rng.sample(&probs);
            external_sampling(game, &outcomes[idx].1, traverser, nodes, rng, iter_weight)
        }
        Node::Decision {
            player,
            infoset,
            actions,
        } => {
            let n = actions.len();
            let strat = {
                let node = nodes.entry(infoset.clone()).or_insert_with(|| InfoNode {
                    regret_sum: vec![0.0; n],
                    strategy_sum: vec![0.0; n],
                });
                regret_matching(&node.regret_sum)
            };

            if player == traverser {
                let mut child_val = vec![0.0; n];
                let mut node_val = 0.0;
                for (a, (_sym, next)) in actions.iter().enumerate() {
                    let cv = external_sampling(game, next, traverser, nodes, rng, iter_weight);
                    child_val[a] = cv;
                    node_val += strat[a] * cv;
                }
                let node = nodes.get_mut(&infoset).expect("node was just inserted");
                for (slot, &cv) in node.regret_sum.iter_mut().zip(&child_val) {
                    *slot = (*slot + cv - node_val).max(0.0);
                }
                node_val
            } else {
                {
                    let node = nodes.get_mut(&infoset).expect("node was just inserted");
                    for (slot, &sv) in node.strategy_sum.iter_mut().zip(&strat) {
                        *slot += iter_weight * sv;
                    }
                }
                let idx = rng.sample(&strat);
                external_sampling(game, &actions[idx].1, traverser, nodes, rng, iter_weight)
            }
        }
    }
}

pub fn solve_blueprint_mccfr<G: Game>(
    game: &G,
    sf0: &SequenceForm,
    sf1: &SequenceForm,
    iterations: u64,
    seed: u64,
) -> CfrBlueprint {
    let mut nodes: HashMap<String, InfoNode> = HashMap::new();
    let mut rng = EsRng::new(seed);
    let root = game.root();
    for t in 1..=iterations {
        let w = t as f64;
        external_sampling(game, &root, 0, &mut nodes, &mut rng, w);
        external_sampling(game, &root, 1, &mut nodes, &mut rng, w);
    }

    let avg_behavior0 = average_behavior(sf0, &nodes);
    let avg_behavior1 = average_behavior(sf1, &nodes);
    let realization = sf0.realization_from_behavior(&avg_behavior0);
    let value = safety_verify_tree(game, sf0, sf1, &realization).value;

    CfrBlueprint {
        realization,
        avg_behavior0,
        avg_behavior1,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goofspiel::Goofspiel;
    use crate::kuhn::Kuhn;
    use crate::leduc::{compile_leduc, Leduc, LeducBig};
    use crate::lp::{solve_blueprint, solve_blueprint_kuhn};
    use crate::payoff::{build, build_kuhn};
    use crate::sequence_form::{compile, compile_kuhn};

    const KUHN_VALUE: f64 = -1.0 / 18.0;

    #[test]
    fn cfr_converges_to_kuhn_value() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let bp = solve_blueprint_cfr(&Kuhn, &sf0, &sf1, 4000);

        assert!(
            (bp.value - KUHN_VALUE).abs() < 5e-3,
            "cfr kuhn value {} vs {KUHN_VALUE}",
            bp.value
        );

        let lp = solve_blueprint_kuhn();
        assert!((bp.value - lp.value).abs() < 5e-3);

        assert!(sf0.constraint_residual(&bp.realization) < 1e-9);
    }

    #[test]
    fn cfr_value_is_a_valid_lower_bound_kuhn() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        for iters in [50, 200, 1000] {
            let bp = solve_blueprint_cfr(&Kuhn, &sf0, &sf1, iters);
            assert!(
                bp.value <= KUHN_VALUE + 1e-9,
                "cfr value {} exceeds game value at {iters} iters",
                bp.value
            );
        }
    }

    #[test]
    fn cfr_converges_to_leduc_value() {
        let sf0 = compile_leduc(0);
        let sf1 = compile_leduc(1);
        let a = build(&Leduc, &sf0, &sf1);
        let exact = solve_blueprint(&sf0, &sf1, &a).value;
        let bp = solve_blueprint_cfr(&Leduc, &sf0, &sf1, 2000);
        assert!(
            (bp.value - exact).abs() < 2e-2,
            "cfr leduc value {} vs exact {exact}",
            bp.value
        );
        assert!(
            bp.value <= exact + 1e-6,
            "cfr value must not exceed game value"
        );
        assert!(sf0.constraint_residual(&bp.realization) < 1e-9);
    }

    #[test]
    fn cfr_converges_to_goofspiel_value() {
        let sf0 = compile(&Goofspiel, 0);
        let sf1 = compile(&Goofspiel, 1);
        let bp = solve_blueprint_cfr(&Goofspiel, &sf0, &sf1, 2000);
        assert!(
            bp.value.abs() < 2e-2,
            "cfr goofspiel value {} vs 0",
            bp.value
        );
        assert!(bp.value <= 1e-6);
    }

    #[test]
    fn cfr_is_reproducible() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let a = solve_blueprint_cfr(&Kuhn, &sf0, &sf1, 300);
        let b = solve_blueprint_cfr(&Kuhn, &sf0, &sf1, 300);
        assert_eq!(a.value, b.value);
        assert_eq!(a.realization, b.realization);
    }

    #[test]
    fn cfr_blueprint_is_approximately_safe_kuhn() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let a = build_kuhn();
        let bp = solve_blueprint_cfr(&Kuhn, &sf0, &sf1, 4000);
        let lp_safety = crate::lp::safety_verify(&sf1, &a, &bp.realization).value;
        assert!(
            lp_safety > KUHN_VALUE - 5e-3,
            "cfr blueprint safety {lp_safety} below game value",
        );
    }

    #[test]
    #[ignore = "scaling diagnostic (~10s release / minutes debug); run with --release -- --ignored"]
    fn cfr_matches_lp_on_big_leduc() {
        let sf0 = compile(&LeducBig, 0);
        let sf1 = compile(&LeducBig, 1);
        let a = build(&LeducBig, &sf0, &sf1);
        let exact = solve_blueprint(&sf0, &sf1, &a).value;
        let bp = solve_blueprint_cfr(&LeducBig, &sf0, &sf1, 1500);
        assert!(
            (bp.value - exact).abs() < 3e-2,
            "big-leduc cfr value {} vs exact {exact}",
            bp.value
        );

        assert!(bp.value <= exact + 1e-6);
        assert!(sf0.constraint_residual(&bp.realization) < 1e-9);
    }

    #[test]
    fn big_leduc_is_a_well_formed_larger_game() {
        let sf0 = compile(&LeducBig, 0);
        let sf1 = compile(&LeducBig, 1);
        assert!(sf0.num_sequences() > 3 * compile_leduc(0).num_sequences());
        let a = build(&LeducBig, &sf0, &sf1);
        let lp = solve_blueprint(&sf0, &sf1, &a);

        let safety = crate::lp::safety_verify(&sf1, &a, &lp.realization).value;
        assert!(
            (safety - lp.value).abs() < 1e-6,
            "big-leduc blueprint not safe"
        );
        assert!(sf0.constraint_residual(&lp.realization) < 1e-9);
    }

    #[test]
    #[ignore = "scaling benchmark; run with --release -- --ignored --nocapture"]
    fn bench_scaling_leduc_vs_big() {
        use std::time::Instant;
        for (name, sf0, sf1) in [
            ("Leduc", compile_leduc(0), compile_leduc(1)),
            ("LeducBig", compile(&LeducBig, 0), compile(&LeducBig, 1)),
        ] {
            let big = name == "LeducBig";

            println!(
                "\n{name}: {} seq/player, {} infosets/player",
                sf0.num_sequences(),
                sf1.num_infosets()
            );

            let t = Instant::now();
            let a = if big {
                build(&LeducBig, &sf0, &sf1)
            } else {
                build(&Leduc, &sf0, &sf1)
            };
            println!("  build A: {:?} ({} nnz)", t.elapsed(), a.nnz());

            let t = Instant::now();
            let lp = solve_blueprint(&sf0, &sf1, &a);
            println!("  LP blueprint: {:?} (value {:.6})", t.elapsed(), lp.value);

            let iters = 1500;
            let t = Instant::now();
            let bp = if big {
                solve_blueprint_cfr(&LeducBig, &sf0, &sf1, iters)
            } else {
                solve_blueprint_cfr(&Leduc, &sf0, &sf1, iters)
            };
            println!(
                "  CFR blueprint ({iters} it): {:?} (value {:.6}, gap {:.2e})",
                t.elapsed(),
                bp.value,
                (bp.value - lp.value).abs()
            );

            let plans: Vec<Vec<f64>> = (0..50)
                .map(|s| {
                    let mut beh = std::collections::HashMap::new();
                    for (i, info) in sf0.info_sets.iter().enumerate() {
                        let n = info.children.len();
                        beh.insert(info.label.clone(), vec![1.0 / n as f64; n]);
                        let _ = (s, i);
                    }
                    sf0.realization_from_behavior(&beh)
                })
                .collect();
            let t = Instant::now();
            for x in &plans {
                let _ = crate::lp::safety_verify(&sf1, &a, x).value;
            }
            let lp_sv = t.elapsed();
            let t = Instant::now();
            for x in &plans {
                let _ = crate::best_response::safety_verify_from_matrix(&sf1, &a, x).value;
            }
            let dp_sv = t.elapsed();
            println!(
                "  safety_verify x{}: LP {:.3} ms/call, matrix-DP {:.3} ms/call, {:.0}x",
                plans.len(),
                lp_sv.as_secs_f64() * 1e3 / plans.len() as f64,
                dp_sv.as_secs_f64() * 1e3 / plans.len() as f64,
                lp_sv.as_secs_f64() / dp_sv.as_secs_f64(),
            );
        }
    }

    #[test]
    fn mccfr_converges_to_kuhn_value() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let bp = solve_blueprint_mccfr(&Kuhn, &sf0, &sf1, 20_000, 2026);
        assert!(
            (bp.value - KUHN_VALUE).abs() < 1.5e-2,
            "mccfr kuhn value {} vs {KUHN_VALUE}",
            bp.value
        );
        assert!(bp.value <= KUHN_VALUE + 1e-9, "value must be a lower bound");
        assert!(sf0.constraint_residual(&bp.realization) < 1e-9);
    }

    #[test]
    fn mccfr_is_reproducible() {
        let sf0 = compile_kuhn(0);
        let sf1 = compile_kuhn(1);
        let a = solve_blueprint_mccfr(&Kuhn, &sf0, &sf1, 500, 7);
        let b = solve_blueprint_mccfr(&Kuhn, &sf0, &sf1, 500, 7);
        assert_eq!(a.value, b.value);
        assert_eq!(a.realization, b.realization);
    }

    #[test]
    fn mccfr_matches_lp_on_small_river() {
        use crate::hand_eval::card;
        use crate::holdem::{HoldemRules, RiverEndgame};

        let board = [
            card(12, 3),
            card(11, 3),
            card(10, 1),
            card(9, 0),
            card(7, 2),
        ];
        let range0 = vec![
            [card(8, 3), card(2, 3)],
            [card(7, 0), card(7, 1)],
            [card(0, 0), card(1, 1)],
        ];
        let range1 = vec![
            [card(12, 0), card(5, 0)],
            [card(8, 1), card(6, 1)],
            [card(3, 2), card(4, 2)],
        ];
        let game = RiverEndgame::new(HoldemRules::river_toy(), board, range0, range1);
        let sf0 = compile(&game, 0);
        let sf1 = compile(&game, 1);
        let a = build(&game, &sf0, &sf1);
        let exact = solve_blueprint(&sf0, &sf1, &a).value;

        let bp = solve_blueprint_mccfr(&game, &sf0, &sf1, 150_000, 2026);
        assert!(
            (bp.value - exact).abs() < 3e-2,
            "mccfr river value {} vs exact LP {exact}",
            bp.value
        );
        assert!(
            bp.value <= exact + 1e-6,
            "value must not exceed the game value"
        );
        assert!(sf0.constraint_residual(&bp.realization) < 1e-9);
    }

    struct ScaleRow {
        combos: usize,
        seq: usize,
        nnz: usize,
        lp_value: f64,
        mccfr_value: f64,
        exploitability: f64,
        t_compile_ms: f64,
        t_build_ms: f64,
        t_lp_ms: f64,
        t_mccfr_ms: f64,
        t_exploit_ms: f64,
    }

    #[test]
    #[ignore = "HUNL scale diagnostic; run with --release -- --ignored --nocapture"]
    fn bench_holdem_scale() {
        use crate::best_response::{best_response_p1_tree, safety_verify_tree};
        use crate::hand_eval::card;
        use crate::holdem::{full_river_range, HoldemRules, RiverEndgame};
        use std::time::Instant;

        let board = [
            card(12, 3),
            card(11, 3),
            card(10, 1),
            card(9, 0),
            card(7, 2),
        ];
        let avail: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
        let pairs = |cards: &[u8]| -> Vec<[u8; 2]> {
            let mut out = Vec::new();
            for i in 0..cards.len() {
                for j in (i + 1)..cards.len() {
                    out.push([cards[i], cards[j]]);
                }
            }
            out
        };
        let rules = HoldemRules::river_small();
        let mccfr_iters = 20_000u64;

        let card_counts = [8usize, 10, 13, 16];

        let rows: Vec<ScaleRow> = std::thread::scope(|scope| {
            let handles: Vec<_> = card_counts
                .iter()
                .map(|&m| {
                    let range0 = pairs(&avail[0..m]);
                    let range1 = pairs(&avail[m..2 * m]);
                    scope.spawn(move || {
                        let game = RiverEndgame::new(rules, board, range0, range1);

                        let t = Instant::now();
                        let sf0 = compile(&game, 0);
                        let sf1 = compile(&game, 1);
                        let t_compile = t.elapsed();

                        let t = Instant::now();
                        let a = build(&game, &sf0, &sf1);
                        let t_build = t.elapsed();

                        let t = Instant::now();
                        let lp = solve_blueprint(&sf0, &sf1, &a);
                        let t_lp = t.elapsed();

                        let t = Instant::now();
                        let bp = solve_blueprint_mccfr(&game, &sf0, &sf1, mccfr_iters, 2026);
                        let t_mccfr = t.elapsed();

                        let t = Instant::now();
                        let y1 = sf1.realization_from_behavior(&bp.avg_behavior1);
                        let br = best_response_p1_tree(&game, &sf0, &sf1, &y1).value;
                        let worst = safety_verify_tree(&game, &sf0, &sf1, &bp.realization).value;
                        let t_exploit = t.elapsed();

                        ScaleRow {
                            combos: sf0.num_infosets(),
                            seq: sf0.num_sequences(),
                            nnz: a.nnz(),
                            lp_value: lp.value,
                            mccfr_value: bp.value,
                            exploitability: br - worst,
                            t_compile_ms: t_compile.as_secs_f64() * 1e3,
                            t_build_ms: t_build.as_secs_f64() * 1e3,
                            t_lp_ms: t_lp.as_secs_f64() * 1e3,
                            t_mccfr_ms: t_mccfr.as_secs_f64() * 1e3,
                            t_exploit_ms: t_exploit.as_secs_f64() * 1e3,
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        println!(
            "\nHUNL river endgame scale (rules=river_small, MCCFR={mccfr_iters} it, {} cores parallel):",
            card_counts.len()
        );
        println!(
            "{:>8} {:>8} {:>10} {:>11} {:>11} {:>11} | {:>10} {:>10} {:>10} {:>11} {:>10}",
            "infoset",
            "seq/pl",
            "A nnz",
            "LP value",
            "MCCFR val",
            "NashConv",
            "compile",
            "build A",
            "LP",
            "MCCFR",
            "exploit",
        );
        for r in &rows {
            println!(
                "{:>8} {:>8} {:>10} {:>+11.5} {:>+11.5} {:>11.5} | {:>9.1}m {:>9.1}m {:>9.1}m {:>10.1}m {:>9.1}m",
                r.combos,
                r.seq,
                r.nnz,
                r.lp_value,
                r.mccfr_value,
                r.exploitability,
                r.t_compile_ms,
                r.t_build_ms,
                r.t_lp_ms,
                r.t_mccfr_ms,
                r.t_exploit_ms,
            );
        }

        let full = full_river_range(&board);
        let t = Instant::now();
        let game_full = RiverEndgame::full(rules, board);
        let sf0_full = compile(&game_full, 0);
        let sf1_full = compile(&game_full, 1);
        let t_full = t.elapsed();
        println!(
            "\nfull range: {} combos/player, {} deals, {} seq/player, {} infosets/player (compile {:.2}s)",
            full.len(),
            game_full.num_deals(),
            sf0_full.num_sequences(),
            sf0_full.num_infosets(),
            t_full.as_secs_f64(),
        );

        let full_iters = 2_000u64;
        let t = Instant::now();
        let bp_full = solve_blueprint_mccfr(&game_full, &sf0_full, &sf1_full, full_iters, 2026);
        let t_mccfr_full = t.elapsed();
        let t = Instant::now();
        let y1 = sf1_full.realization_from_behavior(&bp_full.avg_behavior1);
        let br = crate::best_response::best_response_p1_tree(&game_full, &sf0_full, &sf1_full, &y1)
            .value;
        let worst = crate::best_response::safety_verify_tree(
            &game_full,
            &sf0_full,
            &sf1_full,
            &bp_full.realization,
        )
        .value;
        let t_exploit_full = t.elapsed();
        println!(
            "  MCCFR ({full_iters} it): {:.1}s ({:.2} ms/it), NashConv {:.4}, exploit-path {:.1}ms  [LP: INFEASIBLE]",
            t_mccfr_full.as_secs_f64(),
            t_mccfr_full.as_secs_f64() * 1e3 / full_iters as f64,
            br - worst,
            t_exploit_full.as_secs_f64() * 1e3,
        );
    }
}
