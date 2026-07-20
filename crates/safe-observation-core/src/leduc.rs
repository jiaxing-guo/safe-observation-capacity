use crate::game::{Game, Node};

#[derive(Clone, Copy)]
pub struct LeducRules {
    num_ranks: usize,
    num_suits: usize,
    raise_cap: u8,
    bet0: u32,
    bet1: u32,
    ante: u32,
    rank_chars: &'static [char],
}

impl LeducRules {
    pub const fn standard() -> Self {
        Self {
            num_ranks: 3,
            num_suits: 2,
            raise_cap: 2,
            bet0: 2,
            bet1: 4,
            ante: 1,
            rank_chars: &['J', 'Q', 'K'],
        }
    }

    pub const fn big() -> Self {
        Self {
            num_ranks: 5,
            num_suits: 2,
            raise_cap: 3,
            bet0: 2,
            bet1: 4,
            ante: 1,
            rank_chars: &['9', 'T', 'J', 'Q', 'K'],
        }
    }

    const fn num_cards(&self) -> usize {
        self.num_ranks * self.num_suits
    }

    fn rank(&self, card: usize) -> usize {
        card / self.num_suits
    }

    fn rank_char(&self, card: usize) -> char {
        self.rank_chars[self.rank(card)]
    }

    fn round_bet(&self, round: u8) -> u32 {
        if round == 0 {
            self.bet0
        } else {
            self.bet1
        }
    }

    fn winner(&self, c0: usize, c1: usize, public: usize) -> Option<usize> {
        let (r0, r1, rp) = (self.rank(c0), self.rank(c1), self.rank(public));
        let p0_pair = r0 == rp;
        let p1_pair = r1 == rp;

        if p0_pair && !p1_pair {
            return Some(0);
        }
        if p1_pair && !p0_pair {
            return Some(1);
        }
        match r0.cmp(&r1) {
            std::cmp::Ordering::Greater => Some(0),
            std::cmp::Ordering::Less => Some(1),
            std::cmp::Ordering::Equal => None,
        }
    }
}

#[derive(Clone)]
pub struct LeducState {
    p0: Option<usize>,
    p1: Option<usize>,
    public: Option<usize>,
    round: u8,
    committed: [u32; 2],
    to_act: usize,
    raises: u8,
    acted: u8,
    closed: bool,
    folder: Option<usize>,
    hist0: String,
    hist1: String,
}

pub struct Leduc;

pub struct LeducBig;

impl LeducRules {
    fn infoset_key(&self, s: &LeducState, player: usize) -> String {
        let own = if player == 0 {
            s.p0.unwrap()
        } else {
            s.p1.unwrap()
        };
        let public = match s.public {
            Some(c) => self.rank_char(c),
            None => '-',
        };
        format!("{}|{}|{}|{}", self.rank_char(own), public, s.hist0, s.hist1)
    }

    fn push_action(&self, n: &mut LeducState, ch: char) {
        if n.round == 0 {
            n.hist0.push(ch);
        } else {
            n.hist1.push(ch);
        }
        n.acted += 1;
    }

    fn apply_check(&self, s: &LeducState) -> LeducState {
        let mut n = s.clone();
        let was_acted = n.acted;
        self.push_action(&mut n, 'c');
        if was_acted >= 1 {
            n.closed = true;
        } else {
            n.to_act = 1 - n.to_act;
        }
        n
    }

    fn apply_call(&self, s: &LeducState) -> LeducState {
        let mut n = s.clone();
        let i = n.to_act;
        n.committed[i] = n.committed[1 - i];
        self.push_action(&mut n, 'c');
        n.closed = true;
        n
    }

    fn apply_raise(&self, s: &LeducState, bet: u32) -> LeducState {
        let mut n = s.clone();
        let i = n.to_act;
        n.committed[i] = n.committed[1 - i] + bet;
        n.raises += 1;
        self.push_action(&mut n, 'r');
        n.to_act = 1 - i;
        n
    }

    fn apply_fold(&self, s: &LeducState) -> LeducState {
        let mut n = s.clone();
        let i = n.to_act;
        self.push_action(&mut n, 'f');
        n.folder = Some(i);
        n
    }

    fn legal_actions(&self, s: &LeducState) -> Vec<(char, LeducState)> {
        let i = s.to_act;
        let to_call = s.committed[1 - i] - s.committed[i];
        let bet = self.round_bet(s.round);
        let mut out = Vec::with_capacity(3);
        if to_call == 0 {
            out.push(('c', self.apply_check(s)));
            if s.raises < self.raise_cap {
                out.push(('r', self.apply_raise(s, bet)));
            }
        } else {
            out.push(('f', self.apply_fold(s)));
            out.push(('c', self.apply_call(s)));
            if s.raises < self.raise_cap {
                out.push(('r', self.apply_raise(s, bet)));
            }
        }
        out
    }

    fn showdown_value(&self, s: &LeducState) -> f64 {
        match self.winner(s.p0.unwrap(), s.p1.unwrap(), s.public.unwrap()) {
            Some(0) => s.committed[1] as f64,
            Some(_) => -(s.committed[0] as f64),
            None => 0.0,
        }
    }
}

impl LeducRules {
    fn root(&self) -> LeducState {
        LeducState {
            p0: None,
            p1: None,
            public: None,
            round: 0,
            committed: [0, 0],
            to_act: 0,
            raises: 0,
            acted: 0,
            closed: false,
            folder: None,
            hist0: String::new(),
            hist1: String::new(),
        }
    }

    fn node(&self, s: &LeducState) -> Node<LeducState> {
        let num_cards = self.num_cards();

        if s.p0.is_none() {
            let prob = 1.0 / (num_cards * (num_cards - 1)) as f64;
            let mut outcomes = Vec::with_capacity(num_cards * (num_cards - 1));
            for c0 in 0..num_cards {
                for c1 in 0..num_cards {
                    if c0 != c1 {
                        let mut next = s.clone();
                        next.p0 = Some(c0);
                        next.p1 = Some(c1);
                        next.committed = [self.ante, self.ante];
                        outcomes.push((prob, next));
                    }
                }
            }
            return Node::Chance(outcomes);
        }

        if let Some(f) = s.folder {
            let value = if f == 0 {
                -(s.committed[0] as f64)
            } else {
                s.committed[1] as f64
            };
            return Node::Terminal(value);
        }

        if s.closed {
            if s.round == 0 {
                let used = [s.p0.unwrap(), s.p1.unwrap()];
                let remaining: Vec<usize> = (0..num_cards).filter(|c| !used.contains(c)).collect();
                let prob = 1.0 / remaining.len() as f64;
                let outcomes = remaining
                    .into_iter()
                    .map(|c| {
                        let mut next = s.clone();
                        next.public = Some(c);
                        next.round = 1;
                        next.closed = false;
                        next.raises = 0;
                        next.acted = 0;
                        next.to_act = 0;
                        (prob, next)
                    })
                    .collect();
                return Node::Chance(outcomes);
            }
            return Node::Terminal(self.showdown_value(s));
        }

        let player = s.to_act;
        let infoset = self.infoset_key(s, player);
        let actions = self.legal_actions(s);
        Node::Decision {
            player,
            infoset,
            actions,
        }
    }
}

impl Game for Leduc {
    type State = LeducState;

    fn root(&self) -> LeducState {
        LeducRules::standard().root()
    }

    fn node(&self, s: &LeducState) -> Node<LeducState> {
        LeducRules::standard().node(s)
    }
}

impl Game for LeducBig {
    type State = LeducState;

    fn root(&self) -> LeducState {
        LeducRules::big().root()
    }

    fn node(&self, s: &LeducState) -> Node<LeducState> {
        LeducRules::big().node(s)
    }
}

pub fn compile_leduc(player: usize) -> crate::sequence_form::SequenceForm {
    crate::sequence_form::compile(&Leduc, player)
}

pub fn build_leduc() -> crate::payoff::PayoffMatrix {
    crate::payoff::build(&Leduc, &compile_leduc(0), &compile_leduc(1))
}

pub fn solve_blueprint_leduc() -> crate::lp::BlueprintSolution {
    crate::lp::solve_blueprint(&compile_leduc(0), &compile_leduc(1), &build_leduc())
}

pub fn blueprint_realization(player: usize) -> Vec<f64> {
    let sf0 = compile_leduc(0);
    let sf1 = compile_leduc(1);
    let a = build_leduc();
    if player == 0 {
        crate::lp::solve_blueprint(&sf0, &sf1, &a).realization
    } else {
        let b = crate::payoff::PayoffMatrix {
            nrows: a.ncols,
            ncols: a.nrows,
            entries: a.entries.iter().map(|&(r, c, v)| (c, r, -v)).collect(),
        };
        crate::lp::solve_blueprint(&sf1, &sf0, &b).realization
    }
}

pub fn best_response_leduc(y: &[f64]) -> crate::lp::BestResponseResult {
    crate::lp::best_response_p1(&compile_leduc(0), &build_leduc(), y)
}

pub fn sizes(player: usize) -> (usize, usize) {
    let sf = compile_leduc(player);
    (sf.num_sequences(), sf.num_infosets())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::{best_response_p1, safety_verify, solve_blueprint};
    use crate::payoff::PayoffMatrix;

    const LEDUC_VALUE: f64 = -0.0856064240780;

    #[test]
    fn winner_pairs_and_ranks() {
        let r = LeducRules::standard();
        assert_eq!(r.winner(0, 2, 1), Some(0));
        assert_eq!(r.winner(2, 4, 3), Some(0));
        assert_eq!(r.winner(0, 2, 5), Some(1));
        assert_eq!(r.winner(4, 0, 3), Some(0));
        assert_eq!(r.winner(0, 1, 3), None);
    }

    #[test]
    fn sequence_form_sizes_match_literature() {
        assert_eq!(sizes(0), (337, 144));
        assert_eq!(sizes(1), (337, 144));
    }

    #[test]
    fn blueprint_value_matches_literature() {
        let v = solve_blueprint_leduc().value;
        assert!(
            (v - LEDUC_VALUE).abs() < 1e-6,
            "Leduc value {v} != literature {LEDUC_VALUE}"
        );
    }

    #[test]
    fn blueprint_is_an_exact_nash_zero_exploitability() {
        let sf0 = compile_leduc(0);
        let sf1 = compile_leduc(1);
        let a = build_leduc();

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
            "min_y x*^T A y = {}",
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
}
