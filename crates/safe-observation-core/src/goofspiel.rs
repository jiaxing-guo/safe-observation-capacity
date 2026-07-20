use std::cmp::Ordering;

use crate::game::{Game, Node};

pub const N: usize = 3;

const fn prize(round: usize) -> i32 {
    (N - round) as i32
}

fn card_char(v: usize) -> char {
    debug_assert!((1..=9).contains(&v));
    char::from(b'0' + v as u8)
}

#[derive(Clone)]
pub struct GoofspielState {
    round: usize,

    hand: [u16; 2],

    pending0: Option<usize>,

    to_act: usize,

    diff: i32,

    hist: String,
}

pub struct Goofspiel;

impl Goofspiel {
    fn infoset(player: usize, history: &str) -> String {
        format!("{player}|{history}")
    }
}

impl Game for Goofspiel {
    type State = GoofspielState;

    fn root(&self) -> GoofspielState {
        let full = (1u16 << N) - 1;
        GoofspielState {
            round: 0,
            hand: [full, full],
            pending0: None,
            to_act: 0,
            diff: 0,
            hist: String::new(),
        }
    }

    fn node(&self, s: &GoofspielState) -> Node<GoofspielState> {
        if s.round >= N {
            return Node::Terminal(s.diff as f64);
        }

        let player = s.to_act;

        let infoset = Self::infoset(player, &s.hist);
        let hand = s.hand[player];

        let mut actions = Vec::new();
        for bit in 0..N {
            if hand & (1u16 << bit) == 0 {
                continue;
            }
            let card = bit + 1;
            let mut next = s.clone();
            next.hand[player] &= !(1u16 << bit);

            if player == 0 {
                next.pending0 = Some(card);
                next.to_act = 1;
            } else {
                let p0_bid = s.pending0.expect("player 0 must bid before player 1");
                let p1_bid = card;
                next.diff += match p0_bid.cmp(&p1_bid) {
                    Ordering::Greater => prize(s.round),
                    Ordering::Less => -prize(s.round),
                    Ordering::Equal => 0,
                };
                next.hist.push(card_char(p0_bid));
                next.hist.push(card_char(p1_bid));
                next.pending0 = None;
                next.to_act = 0;
                next.round = s.round + 1;
            }
            actions.push((card_char(card), next));
        }

        Node::Decision {
            player,
            infoset,
            actions,
        }
    }
}

pub fn compile_goofspiel(player: usize) -> crate::sequence_form::SequenceForm {
    crate::sequence_form::compile(&Goofspiel, player)
}

pub fn build_goofspiel() -> crate::payoff::PayoffMatrix {
    crate::payoff::build(&Goofspiel, &compile_goofspiel(0), &compile_goofspiel(1))
}

pub fn solve_blueprint_goofspiel() -> crate::lp::BlueprintSolution {
    crate::lp::solve_blueprint(
        &compile_goofspiel(0),
        &compile_goofspiel(1),
        &build_goofspiel(),
    )
}

pub fn sizes(player: usize) -> (usize, usize) {
    let sf = compile_goofspiel(player);
    (sf.num_sequences(), sf.num_infosets())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::{best_response_p1, safety_verify, solve_blueprint};
    use crate::payoff::PayoffMatrix;

    #[test]
    fn prizes_are_descending_and_sum_to_triangular() {
        let total: i32 = (0..N).map(prize).sum();
        assert_eq!(total, (N * (N + 1) / 2) as i32);
        assert_eq!(prize(0), N as i32);
        assert_eq!(prize(N - 1), 1);
    }

    #[test]
    fn sequence_form_sizes_are_symmetric() {
        assert_eq!(sizes(0), (58, 46));
        assert_eq!(sizes(1), (58, 46));
    }

    #[test]
    fn blueprint_value_is_zero_by_symmetry() {
        let v = solve_blueprint_goofspiel().value;
        assert!(v.abs() < 1e-6, "Goofspiel value {v} != 0");
    }

    #[test]
    fn blueprint_is_an_exact_nash_zero_exploitability() {
        let sf0 = compile_goofspiel(0);
        let sf1 = compile_goofspiel(1);
        let a = build_goofspiel();

        let p0 = solve_blueprint(&sf0, &sf1, &a);
        let v0 = p0.value;

        let b = PayoffMatrix {
            nrows: a.ncols,
            ncols: a.nrows,
            entries: a.entries.iter().map(|&(r, c, v)| (c, r, -v)).collect(),
        };
        let p1 = solve_blueprint(&sf1, &sf0, &b);
        let v1 = p1.value;

        assert!(
            (v0 + v1).abs() < 1e-6,
            "v0 = {v0}, v1 = {v1} (not a saddle point)"
        );

        let safety = safety_verify(&sf1, &a, &p0.realization);
        assert!(
            (safety.value - v0).abs() < 1e-6,
            "min_y x*^T A y = {} != v0 = {v0}",
            safety.value
        );
        let br = best_response_p1(&sf0, &a, &p1.realization);
        assert!(
            (br.value - v0).abs() < 1e-6,
            "max_x x^T A y* = {} != v0 = {v0}",
            br.value
        );

        assert!(sf0.constraint_residual(&p0.realization) < 1e-6);
        assert!(sf1.constraint_residual(&p1.realization) < 1e-6);
    }

    #[test]
    fn uniform_vs_uniform_has_value_zero_by_symmetry() {
        let sf0 = compile_goofspiel(0);
        let sf1 = compile_goofspiel(1);
        let a = build_goofspiel();
        let x_unif = sf0.realization_from_behavior(&std::collections::HashMap::new());
        let y_unif = sf1.realization_from_behavior(&std::collections::HashMap::new());
        let v = a.bilinear(&x_unif, &y_unif);
        assert!(v.abs() < 1e-9, "uniform-vs-uniform value {v} != 0");
    }
}
