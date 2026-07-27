//! Censored chain algorithms for safe observation. See supplementary Reproducibility for its role in the release workflow.

use crate::game::{Game, Node};

#[derive(Clone)]
/// Stores state for chain state.
pub struct ChainState {
    s0: Option<usize>,

    s1: Option<usize>,

    committed: [u32; 2],

    street: usize,

    to_act: usize,

    folder: Option<usize>,

    hist: String,
}

#[derive(Clone)]
/// Stores state for chain game.
pub struct ChainGame {
    num_types: usize,
    depth: usize,
    ante: u32,
    bet: u32,
}

/// Implements operations for `ChainGame`.
impl ChainGame {
    /// Constructs a new value from the supplied configuration.
    pub fn new(depth: usize, num_types: usize) -> Self {
        ChainGame {
            num_types,
            depth,
            ante: 1,
            bet: 1,
        }
    }

    /// Computes type char.
    fn type_char(&self, t: usize) -> char {
        std::char::from_digit(t as u32, 36).unwrap_or('?')
    }

    /// Computes information set.
    fn infoset(&self, s: &ChainState, player: usize) -> String {
        let own = if player == 0 {
            s.s0.expect("dealt")
        } else {
            s.s1.expect("dealt")
        };
        format!("{}|{}", self.type_char(own), s.hist)
    }

    /// Compute the showdown value.
    fn showdown_value(&self, s: &ChainState) -> f64 {
        match s.s0.expect("dealt").cmp(&s.s1.expect("dealt")) {
            std::cmp::Ordering::Greater => s.committed[1] as f64,
            std::cmp::Ordering::Less => -(s.committed[0] as f64),
            std::cmp::Ordering::Equal => 0.0,
        }
    }
}

/// Implements operations for `ChainGame`.
impl Game for ChainGame {
    /// Aliases the type used for state.
    type State = ChainState;

    /// Return the initial game state.
    fn root(&self) -> ChainState {
        ChainState {
            s0: None,
            s1: None,
            committed: [self.ante, self.ante],
            street: 0,
            to_act: 0,
            folder: None,
            hist: String::new(),
        }
    }

    /// Expand a state into its chance, decision, or terminal game node.
    fn node(&self, s: &ChainState) -> Node<ChainState> {
        if s.s0.is_none() {
            let n = self.num_types;
            let prob = 1.0 / (n * n) as f64;
            let mut outcomes = Vec::with_capacity(n * n);
            for a in 0..n {
                for b in 0..n {
                    let mut next = s.clone();
                    next.s0 = Some(a);
                    next.s1 = Some(b);
                    outcomes.push((prob, next));
                }
            }
            return Node::Chance(outcomes);
        }

        if let Some(f) = s.folder {
            let value = if f == 1 {
                s.committed[1] as f64
            } else {
                -(s.committed[0] as f64)
            };
            return Node::Terminal(value);
        }

        if s.street >= self.depth {
            return Node::Terminal(self.showdown_value(s));
        }

        let player = s.to_act;
        let infoset = self.infoset(s, player);

        if player == 0 {
            let mut check = s.clone();
            check.hist.push('c');
            check.hist.push('/');
            check.street += 1;
            check.to_act = 0;

            let mut bets = s.clone();
            bets.committed[0] += self.bet;
            bets.hist.push('b');
            bets.to_act = 1;

            Node::Decision {
                player: 0,
                infoset,
                actions: vec![('c', check), ('b', bets)],
            }
        } else {
            let mut fold = s.clone();
            fold.hist.push('f');
            fold.folder = Some(1);

            let mut call = s.clone();
            call.committed[1] += self.bet;
            call.hist.push('c');
            call.hist.push('/');
            call.street += 1;
            call.to_act = 0;

            Node::Decision {
                player: 1,
                infoset,
                actions: vec![('f', fold), ('c', call)],
            }
        }
    }
}

/// Computes chain game.
pub fn chain_game(suffix: &str) -> Option<ChainGame> {
    let body = suffix.strip_prefix('_')?;
    let mut depth: Option<usize> = None;
    let mut types: Option<usize> = None;
    for part in body.split('_') {
        if let Some(d) = part.strip_prefix('d') {
            depth = Some(d.parse().ok()?);
        } else if let Some(k) = part.strip_prefix('k') {
            types = Some(k.parse().ok()?);
        } else {
            return None;
        }
    }
    let depth = depth?;
    let types = types?;
    if depth == 0 || types == 0 || types > 36 {
        return None;
    }
    Some(ChainGame::new(depth, types))
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;
    use crate::lp::{best_response_p1, safety_verify, solve_blueprint};
    use crate::payoff::{build, PayoffMatrix};
    use crate::sequence_form::compile;

    /// Computes Nash saddle holds.
    fn nash_saddle_holds(game: &ChainGame) {
        let sf0 = compile(game, 0);
        let sf1 = compile(game, 1);
        let a = build(game, &sf0, &sf1);

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
    /// Verifies that depth1 is an exact Nash.
    fn depth1_is_an_exact_nash() {
        nash_saddle_holds(&ChainGame::new(1, 4));
    }

    #[test]
    /// Verifies that depth2 is an exact Nash.
    fn depth2_is_an_exact_nash() {
        nash_saddle_holds(&ChainGame::new(2, 4));
    }

    #[test]
    /// Verifies that depth3 is an exact Nash.
    fn depth3_is_an_exact_nash() {
        nash_saddle_holds(&ChainGame::new(3, 3));
    }

    #[test]
    /// Verifies that parses game strings.
    fn parses_game_strings() {
        assert!(chain_game("_d3_k4").is_some());
        assert!(chain_game("_d1_k2").is_some());
        assert!(chain_game("_k4_d3").is_some());
        assert!(chain_game("").is_none());
        assert!(chain_game("_d0_k4").is_none());
        assert!(chain_game("_d3").is_none());
        assert!(chain_game("_x3_k4").is_none());
    }

    #[test]
    /// Verifies that has censored fold and round structure.
    fn has_censored_fold_and_round_structure() {
        let game = ChainGame::new(2, 3);
        let sf1 = compile(&game, 1);
        let mut saw_fold_infoset = false;
        let mut saw_round_separator = false;
        for info in &sf1.info_sets {
            let hist = info.label.split('|').nth(1).unwrap_or("");
            if hist.ends_with('b') {
                saw_fold_infoset = true;
            }
            if hist.contains('/') {
                saw_round_separator = true;
            }
        }
        assert!(saw_fold_infoset, "expected a P2 fold information set");
        assert!(
            saw_round_separator,
            "expected '/' round separators in deep play"
        );
    }
}
