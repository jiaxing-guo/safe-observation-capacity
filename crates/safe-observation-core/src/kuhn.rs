use std::collections::{BTreeMap, HashMap};

use crate::cfr::{normalize, regret_matching};
use crate::game::{Game, Node as GameNode};

const NUM_ACTIONS: usize = 2;
const SYMBOLS: [char; NUM_ACTIONS] = ['p', 'b'];

pub(crate) const fn deals() -> [(usize, usize); 6] {
    [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
}

pub(crate) fn is_terminal(history: &str) -> bool {
    matches!(history, "pp" | "bp" | "bb" | "pbp" | "pbb")
}

#[derive(Default, Clone)]
struct Node {
    regret_sum: [f64; NUM_ACTIONS],
    strategy_sum: [f64; NUM_ACTIONS],
}

pub struct KuhnSolution {
    pub value: f64,

    pub strategy: BTreeMap<String, [f64; NUM_ACTIONS]>,
}

fn info_set(card: usize, history: &str) -> String {
    format!("{card}{history}")
}

fn extend(history: &str, action: usize) -> String {
    let mut next = String::with_capacity(history.len() + 1);
    next.push_str(history);
    next.push(SYMBOLS[action]);
    next
}

fn cfr(
    nodes: &mut HashMap<String, Node>,
    cards: (usize, usize),
    history: &str,
    p0: f64,
    p1: f64,
) -> f64 {
    let plays = history.len();
    let player = plays % 2;

    if plays >= 2 {
        let bytes = history.as_bytes();
        let terminal_pass = bytes[plays - 1] == b'p';
        let double_bet = bytes[plays - 2] == b'b' && bytes[plays - 1] == b'b';
        let (own, opp) = if player == 0 {
            (cards.0, cards.1)
        } else {
            (cards.1, cards.0)
        };
        let higher = own > opp;
        if terminal_pass {
            if history == "pp" {
                return if higher { 1.0 } else { -1.0 };
            }

            return 1.0;
        } else if double_bet {
            return if higher { 2.0 } else { -2.0 };
        }
    }

    let card = if player == 0 { cards.0 } else { cards.1 };
    let key = info_set(card, history);

    let strategy = {
        let node = nodes.entry(key.clone()).or_default();
        let strat = regret_matching(&node.regret_sum);
        let realization = if player == 0 { p0 } else { p1 };
        for (slot, &s) in node.strategy_sum.iter_mut().zip(&strat) {
            *slot += realization * s;
        }
        strat
    };

    let mut util = [0.0; NUM_ACTIONS];
    let mut node_util = 0.0;
    for a in 0..NUM_ACTIONS {
        let next = extend(history, a);
        util[a] = if player == 0 {
            -cfr(nodes, cards, &next, p0 * strategy[a], p1)
        } else {
            -cfr(nodes, cards, &next, p0, p1 * strategy[a])
        };
        node_util += strategy[a] * util[a];
    }

    let opp_reach = if player == 0 { p1 } else { p0 };
    let node = nodes.get_mut(&key).expect("node was just inserted");
    for (slot, &u) in node.regret_sum.iter_mut().zip(&util) {
        *slot += opp_reach * (u - node_util);
    }

    node_util
}

pub(crate) fn terminal_value_p1(history: &str, cards: (usize, usize)) -> f64 {
    let (c0, c1) = cards;
    match history {
        "pp" => {
            if c0 > c1 {
                1.0
            } else {
                -1.0
            }
        }
        "bp" => 1.0,
        "pbp" => -1.0,
        "bb" | "pbb" => {
            if c0 > c1 {
                2.0
            } else {
                -2.0
            }
        }
        _ => unreachable!("non-terminal history passed to terminal_value_p1: {history}"),
    }
}

fn eval_node(
    strategy: &BTreeMap<String, [f64; NUM_ACTIONS]>,
    cards: (usize, usize),
    history: &str,
) -> f64 {
    if is_terminal(history) {
        return terminal_value_p1(history, cards);
    }
    let player = history.len() % 2;
    let card = if player == 0 { cards.0 } else { cards.1 };
    let probs = strategy
        .get(&info_set(card, history))
        .copied()
        .unwrap_or([0.5, 0.5]);
    let mut value = 0.0;
    for (a, &p) in probs.iter().enumerate() {
        value += p * eval_node(strategy, cards, &extend(history, a));
    }
    value
}

pub fn evaluate(strategy: &BTreeMap<String, [f64; NUM_ACTIONS]>) -> f64 {
    deals()
        .iter()
        .map(|&cards| eval_node(strategy, cards, "") / 6.0)
        .sum()
}

pub fn solve(iterations: u64) -> KuhnSolution {
    let mut nodes: HashMap<String, Node> = HashMap::new();
    for _ in 0..iterations {
        for &cards in deals().iter() {
            cfr(&mut nodes, cards, "", 1.0, 1.0);
        }
    }

    let mut strategy: BTreeMap<String, [f64; NUM_ACTIONS]> = BTreeMap::new();
    for (key, node) in &nodes {
        let avg = normalize(&node.strategy_sum);
        strategy.insert(key.clone(), [avg[0], avg[1]]);
    }

    let value = evaluate(&strategy);
    KuhnSolution { value, strategy }
}

pub struct Kuhn;

#[derive(Clone)]
pub struct KuhnState {
    dealt: bool,
    cards: (usize, usize),
    history: String,
}

impl Game for Kuhn {
    type State = KuhnState;

    fn root(&self) -> KuhnState {
        KuhnState {
            dealt: false,
            cards: (0, 0),
            history: String::new(),
        }
    }

    fn node(&self, s: &KuhnState) -> GameNode<KuhnState> {
        if !s.dealt {
            let outcomes = deals()
                .iter()
                .map(|&cards| {
                    (
                        1.0 / 6.0,
                        KuhnState {
                            dealt: true,
                            cards,
                            history: String::new(),
                        },
                    )
                })
                .collect();
            return GameNode::Chance(outcomes);
        }
        if is_terminal(&s.history) {
            return GameNode::Terminal(terminal_value_p1(&s.history, s.cards));
        }
        let player = s.history.len() % 2;
        let card = if player == 0 { s.cards.0 } else { s.cards.1 };

        let infoset = format!("{card}:{}", s.history);
        let actions = SYMBOLS
            .iter()
            .map(|&sym| {
                let mut history = String::with_capacity(s.history.len() + 1);
                history.push_str(&s.history);
                history.push(sym);
                (
                    sym,
                    KuhnState {
                        dealt: true,
                        cards: s.cards,
                        history,
                    },
                )
            })
            .collect();
        GameNode::Decision {
            player,
            infoset,
            actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_classification() {
        assert!(is_terminal("pp"));
        assert!(is_terminal("pbb"));
        assert!(!is_terminal("pb"));
        assert!(!is_terminal("p"));
    }

    #[test]
    fn value_matches_known_kuhn_value() {
        let sol = solve(50_000);
        assert!(
            (sol.value - (-1.0 / 18.0)).abs() < 5e-3,
            "value = {} (expected ~ -0.0556)",
            sol.value
        );

        assert_eq!(sol.strategy.len(), 12);
        for probs in sol.strategy.values() {
            assert!((probs[0] + probs[1] - 1.0).abs() < 1e-9);
        }
    }
}
