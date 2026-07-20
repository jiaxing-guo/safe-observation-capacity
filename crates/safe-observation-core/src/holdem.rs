use crate::game::{Game, Node};

#[derive(Clone, Copy)]
pub struct HoldemRules {
    num_strengths: usize,

    ante: u32,

    stack: u32,

    raise_cap: u8,
}

impl HoldemRules {
    pub const fn river_toy() -> Self {
        Self {
            num_strengths: 3,
            ante: 1,
            stack: 6,
            raise_cap: 2,
        }
    }

    pub const fn river_small() -> Self {
        Self {
            num_strengths: 4,
            ante: 1,
            stack: 10,
            raise_cap: 3,
        }
    }

    fn strength_char(&self, s: usize) -> char {
        debug_assert!(s < 10);
        char::from(b'0' + s as u8)
    }
}

#[derive(Clone)]
pub struct HoldemState {
    pub(crate) s0: Option<usize>,
    pub(crate) s1: Option<usize>,

    pub(crate) committed: [u32; 2],
    pub(crate) to_act: usize,

    pub(crate) raises: u8,

    pub(crate) acted: u8,
    pub(crate) closed: bool,
    pub(crate) folder: Option<usize>,
    pub(crate) hist: String,

    pub(crate) street: u8,

    pub(crate) river: Option<u8>,
}

pub struct HoldemToy;

pub struct HoldemSmall;

impl HoldemRules {
    fn infoset_key(&self, s: &HoldemState, player: usize) -> String {
        let own = if player == 0 {
            s.s0.unwrap()
        } else {
            s.s1.unwrap()
        };
        format!("{}|{}", self.strength_char(own), s.hist)
    }

    fn push_action(&self, n: &mut HoldemState, ch: char) {
        n.hist.push(ch);
        n.acted += 1;
    }

    fn apply_check(&self, s: &HoldemState) -> HoldemState {
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

    fn apply_call(&self, s: &HoldemState) -> HoldemState {
        let mut n = s.clone();
        let i = n.to_act;
        n.committed[i] = n.committed[1 - i];
        self.push_action(&mut n, 'c');
        n.closed = true;
        n
    }

    fn apply_raise(&self, s: &HoldemState, target: u32, ch: char) -> HoldemState {
        let mut n = s.clone();
        let i = n.to_act;
        n.committed[i] = target;
        n.raises += 1;
        self.push_action(&mut n, ch);
        n.to_act = 1 - i;
        n
    }

    fn apply_fold(&self, s: &HoldemState) -> HoldemState {
        let mut n = s.clone();
        let i = n.to_act;
        self.push_action(&mut n, 'f');
        n.folder = Some(i);
        n
    }

    pub(crate) fn legal_actions(&self, s: &HoldemState) -> Vec<(char, HoldemState)> {
        let i = s.to_act;
        let to_call = s.committed[1 - i] - s.committed[i];
        let pot = s.committed[0] + s.committed[1];
        let mut out = Vec::with_capacity(4);

        if to_call == 0 {
            out.push(('c', self.apply_check(s)));
            if s.raises < self.raise_cap && s.committed[i] < self.stack {
                let target = (s.committed[i] + pot).min(self.stack);
                if target < self.stack {
                    out.push(('p', self.apply_raise(s, target, 'p')));
                }
                out.push(('a', self.apply_raise(s, self.stack, 'a')));
            }
        } else {
            out.push(('f', self.apply_fold(s)));
            out.push(('c', self.apply_call(s)));
            if s.raises < self.raise_cap && self.stack > s.committed[1 - i] {
                let target = (3 * s.committed[1 - i]).min(self.stack);
                if target < self.stack {
                    out.push(('p', self.apply_raise(s, target, 'p')));
                }
                out.push(('a', self.apply_raise(s, self.stack, 'a')));
            }
        }
        out
    }

    fn showdown_value(&self, s: &HoldemState) -> f64 {
        match s.s0.unwrap().cmp(&s.s1.unwrap()) {
            std::cmp::Ordering::Greater => s.committed[1] as f64,
            std::cmp::Ordering::Less => -(s.committed[0] as f64),
            std::cmp::Ordering::Equal => 0.0,
        }
    }

    pub(crate) fn root(&self) -> HoldemState {
        HoldemState {
            s0: None,
            s1: None,
            committed: [self.ante, self.ante],
            to_act: 0,
            raises: 0,
            acted: 0,
            closed: false,
            folder: None,
            hist: String::new(),
            street: 0,
            river: None,
        }
    }

    fn node(&self, s: &HoldemState) -> Node<HoldemState> {
        if s.s0.is_none() {
            let n = self.num_strengths;
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
            let value = if f == 0 {
                -(s.committed[0] as f64)
            } else {
                s.committed[1] as f64
            };
            return Node::Terminal(value);
        }

        if s.closed {
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

impl Game for HoldemToy {
    type State = HoldemState;
    fn root(&self) -> HoldemState {
        HoldemRules::river_toy().root()
    }
    fn node(&self, s: &HoldemState) -> Node<HoldemState> {
        HoldemRules::river_toy().node(s)
    }
}

impl Game for HoldemSmall {
    type State = HoldemState;
    fn root(&self) -> HoldemState {
        HoldemRules::river_small().root()
    }
    fn node(&self, s: &HoldemState) -> Node<HoldemState> {
        HoldemRules::river_small().node(s)
    }
}

pub fn sizes<G: Game<State = HoldemState>>(game: &G, player: usize) -> (usize, usize) {
    let sf = crate::sequence_form::compile(game, player);
    (sf.num_sequences(), sf.num_infosets())
}

use crate::hand_eval::{card_str, evaluate7};

pub struct RiverEndgame {
    rules: HoldemRules,
    range0: Vec<[u8; 2]>,
    range1: Vec<[u8; 2]>,
    score0: Vec<u32>,
    score1: Vec<u32>,

    deals: Vec<(usize, usize)>,
}

pub fn full_river_range(board: &[u8; 5]) -> Vec<[u8; 2]> {
    let avail: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
    let mut out = Vec::with_capacity(avail.len() * (avail.len() - 1) / 2);
    for i in 0..avail.len() {
        for j in (i + 1)..avail.len() {
            out.push([avail[i], avail[j]]);
        }
    }
    out
}

impl RiverEndgame {
    pub fn new(
        rules: HoldemRules,
        board: [u8; 5],
        range0: Vec<[u8; 2]>,
        range1: Vec<[u8; 2]>,
    ) -> Self {
        let score = |hole: &[u8; 2]| -> u32 {
            let seven = [
                hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
            ];
            evaluate7(&seven)
        };
        let score0: Vec<u32> = range0.iter().map(score).collect();
        let score1: Vec<u32> = range1.iter().map(score).collect();
        let mut deals = Vec::new();
        for (i, h0) in range0.iter().enumerate() {
            for (j, h1) in range1.iter().enumerate() {
                if h0[0] != h1[0] && h0[0] != h1[1] && h0[1] != h1[0] && h0[1] != h1[1] {
                    deals.push((i, j));
                }
            }
        }
        Self {
            rules,
            range0,
            range1,
            score0,
            score1,
            deals,
        }
    }

    pub fn full(rules: HoldemRules, board: [u8; 5]) -> Self {
        let range = full_river_range(&board);
        Self::new(rules, board, range.clone(), range)
    }

    pub fn num_deals(&self) -> usize {
        self.deals.len()
    }

    pub(crate) fn rules(&self) -> HoldemRules {
        self.rules
    }

    pub(crate) fn range(&self, player: usize) -> &[[u8; 2]] {
        if player == 0 {
            &self.range0
        } else {
            &self.range1
        }
    }

    pub(crate) fn scores(&self, player: usize) -> &[u32] {
        if player == 0 {
            &self.score0
        } else {
            &self.score1
        }
    }

    fn infoset_key(&self, s: &HoldemState, player: usize) -> String {
        let hole = if player == 0 {
            self.range0[s.s0.unwrap()]
        } else {
            self.range1[s.s1.unwrap()]
        };
        format!("{}{}|{}", card_str(hole[0]), card_str(hole[1]), s.hist)
    }

    fn showdown_value(&self, s: &HoldemState) -> f64 {
        match self.score0[s.s0.unwrap()].cmp(&self.score1[s.s1.unwrap()]) {
            std::cmp::Ordering::Greater => s.committed[1] as f64,
            std::cmp::Ordering::Less => -(s.committed[0] as f64),
            std::cmp::Ordering::Equal => 0.0,
        }
    }
}

impl Game for RiverEndgame {
    type State = HoldemState;

    fn root(&self) -> HoldemState {
        self.rules.root()
    }

    fn node(&self, s: &HoldemState) -> Node<HoldemState> {
        if s.s0.is_none() {
            let prob = 1.0 / self.deals.len() as f64;
            let outcomes = self
                .deals
                .iter()
                .map(|&(i, j)| {
                    let mut next = s.clone();
                    next.s0 = Some(i);
                    next.s1 = Some(j);
                    (prob, next)
                })
                .collect();
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
            return Node::Terminal(self.showdown_value(s));
        }

        let player = s.to_act;
        let infoset = self.infoset_key(s, player);
        let actions = self.rules.legal_actions(s);
        Node::Decision {
            player,
            infoset,
            actions,
        }
    }

    fn sample_chance(&self, s: &HoldemState, rng: &mut dyn FnMut() -> f64) -> Option<HoldemState> {
        if s.s0.is_some() {
            return None;
        }
        let n = self.deals.len();
        let idx = ((rng() * n as f64) as usize).min(n - 1);
        let (i, j) = self.deals[idx];
        let mut next = s.clone();
        next.s0 = Some(i);
        next.s1 = Some(j);
        Some(next)
    }
}

pub fn canonical_holdem() -> RiverEndgame {
    let board = [
        crate::hand_eval::card(12, 3),
        crate::hand_eval::card(11, 3),
        crate::hand_eval::card(10, 1),
        crate::hand_eval::card(9, 0),
        crate::hand_eval::card(7, 2),
    ];
    river_endgame_on(board)
}

fn ten_highest_range(board: &[u8; 5]) -> Vec<[u8; 2]> {
    let mut cards = Vec::with_capacity(10);
    'outer: for r in (0..13u8).rev() {
        for s in 0..4u8 {
            let c = crate::hand_eval::card(r, s);
            if !board.contains(&c) {
                cards.push(c);
                if cards.len() == 10 {
                    break 'outer;
                }
            }
        }
    }
    let mut range = Vec::with_capacity(45);
    for i in 0..cards.len() {
        for j in (i + 1)..cards.len() {
            range.push([cards[i], cards[j]]);
        }
    }
    range
}

fn river_endgame_on(board: [u8; 5]) -> RiverEndgame {
    let range = ten_highest_range(&board);
    RiverEndgame::new(HoldemRules::river_small(), board, range.clone(), range)
}

pub fn holdem_variant(name: &str) -> Option<RiverEndgame> {
    let c = crate::hand_eval::card;
    let board = match name {
        "paired" => [c(11, 3), c(11, 2), c(10, 1), c(5, 0), c(0, 3)],
        "monotone" => [c(12, 3), c(8, 3), c(5, 3), c(2, 3), c(0, 3)],
        "dry" => [c(11, 1), c(6, 0), c(3, 2), c(1, 3), c(0, 0)],
        "wet" => [c(9, 2), c(8, 2), c(7, 0), c(6, 1), c(2, 3)],
        "low" => [c(6, 1), c(4, 0), c(2, 2), c(1, 3), c(0, 0)],
        _ => return None,
    };
    Some(river_endgame_on(board))
}

pub fn compile_holdem(player: usize) -> crate::sequence_form::SequenceForm {
    crate::sequence_form::compile(&canonical_holdem(), player)
}

pub fn build_holdem() -> crate::payoff::PayoffMatrix {
    let game = canonical_holdem();
    crate::payoff::build(&game, &compile_holdem(0), &compile_holdem(1))
}

use crate::hand_eval::rank_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiverDeal {
    Exact,

    Bucketed(usize),
}

pub struct TurnRiverEndgame {
    rules: HoldemRules,
    turn_board: [u8; 4],
    range0: Vec<[u8; 2]>,
    range1: Vec<[u8; 2]>,
    deal_mode: RiverDeal,

    avail_river: Vec<u8>,

    buckets: Vec<(f64, u8)>,

    deals: Vec<(usize, usize)>,
}

impl TurnRiverEndgame {
    pub fn new(
        rules: HoldemRules,
        turn_board: [u8; 4],
        range0: Vec<[u8; 2]>,
        range1: Vec<[u8; 2]>,
        deal_mode: RiverDeal,
    ) -> Self {
        let avail_river: Vec<u8> = (0..52u8).filter(|c| !turn_board.contains(c)).collect();
        let buckets = match deal_mode {
            RiverDeal::Exact => Vec::new(),
            RiverDeal::Bucketed(k) => bucket_river(&avail_river, k),
        };
        let mut deals = Vec::new();
        for (i, h0) in range0.iter().enumerate() {
            for (j, h1) in range1.iter().enumerate() {
                if h0[0] != h1[0] && h0[0] != h1[1] && h0[1] != h1[0] && h0[1] != h1[1] {
                    deals.push((i, j));
                }
            }
        }
        Self {
            rules,
            turn_board,
            range0,
            range1,
            deal_mode,
            avail_river,
            buckets,
            deals,
        }
    }

    fn hole(&self, s: &HoldemState, player: usize) -> [u8; 2] {
        if player == 0 {
            self.range0[s.s0.unwrap()]
        } else {
            self.range1[s.s1.unwrap()]
        }
    }

    fn infoset_key(&self, s: &HoldemState, player: usize) -> String {
        let h = self.hole(s, player);
        format!("{}{}|{}", card_str(h[0]), card_str(h[1]), s.hist)
    }

    fn seven(&self, hole: [u8; 2], river: u8) -> [u8; 7] {
        let b = self.turn_board;
        [hole[0], hole[1], b[0], b[1], b[2], b[3], river]
    }

    fn showdown_value(&self, s: &HoldemState) -> f64 {
        let river = s.river.unwrap();
        let r0 = evaluate7(&self.seven(self.hole(s, 0), river));
        let r1 = evaluate7(&self.seven(self.hole(s, 1), river));
        match r0.cmp(&r1) {
            std::cmp::Ordering::Greater => s.committed[1] as f64,
            std::cmp::Ordering::Less => -(s.committed[0] as f64),
            std::cmp::Ordering::Equal => 0.0,
        }
    }

    fn river_state(&self, s: &HoldemState, card: u8) -> HoldemState {
        let mut n = s.clone();
        n.river = Some(card);
        n.street = 1;
        n.closed = false;
        n.acted = 0;
        n.raises = 0;
        n.to_act = 0;
        n.hist.push('/');
        n.hist.push_str(&card_str(card));
        n
    }

    pub fn num_deals(&self) -> usize {
        self.deals.len()
    }
}

fn bucket_river(avail: &[u8], k: usize) -> Vec<(f64, u8)> {
    let n = avail.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let mut sorted = avail.to_vec();
    sorted.sort_by_key(|&c| (rank_of(c), c));
    let mut out = Vec::with_capacity(k);
    let base = n / k;
    let rem = n % k;
    let mut start = 0usize;
    for b in 0..k {
        let len = base + usize::from(b < rem);
        if len == 0 {
            continue;
        }
        let chunk = &sorted[start..start + len];
        let rep = *chunk.iter().max_by_key(|&&c| (rank_of(c), c)).unwrap();
        out.push((len as f64 / n as f64, rep));
        start += len;
    }
    out
}

impl Game for TurnRiverEndgame {
    type State = HoldemState;

    fn root(&self) -> HoldemState {
        self.rules.root()
    }

    fn node(&self, s: &HoldemState) -> Node<HoldemState> {
        if s.s0.is_none() {
            let prob = 1.0 / self.deals.len() as f64;
            let outcomes = self
                .deals
                .iter()
                .map(|&(i, j)| {
                    let mut next = s.clone();
                    next.s0 = Some(i);
                    next.s1 = Some(j);
                    (prob, next)
                })
                .collect();
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

        if s.closed && s.street == 0 {
            let outcomes: Vec<(f64, HoldemState)> = match self.deal_mode {
                RiverDeal::Exact => {
                    let h0 = self.hole(s, 0);
                    let h1 = self.hole(s, 1);
                    let live: Vec<u8> = self
                        .avail_river
                        .iter()
                        .copied()
                        .filter(|&c| c != h0[0] && c != h0[1] && c != h1[0] && c != h1[1])
                        .collect();
                    let prob = 1.0 / live.len() as f64;
                    live.into_iter()
                        .map(|c| (prob, self.river_state(s, c)))
                        .collect()
                }
                RiverDeal::Bucketed(_) => self
                    .buckets
                    .iter()
                    .map(|&(p, rep)| (p, self.river_state(s, rep)))
                    .collect(),
            };
            return Node::Chance(outcomes);
        }

        if s.closed {
            return Node::Terminal(self.showdown_value(s));
        }

        let player = s.to_act;
        let infoset = self.infoset_key(s, player);
        let actions = self.rules.legal_actions(s);
        Node::Decision {
            player,
            infoset,
            actions,
        }
    }

    fn sample_chance(&self, s: &HoldemState, rng: &mut dyn FnMut() -> f64) -> Option<HoldemState> {
        if s.s0.is_some() {
            return None;
        }
        let n = self.deals.len();
        let idx = ((rng() * n as f64) as usize).min(n - 1);
        let (i, j) = self.deals[idx];
        let mut next = s.clone();
        next.s0 = Some(i);
        next.s1 = Some(j);
        Some(next)
    }
}

pub enum HoldemGame {
    River(RiverEndgame),

    TurnRiver(TurnRiverEndgame),
}

impl Game for HoldemGame {
    type State = HoldemState;
    fn root(&self) -> HoldemState {
        match self {
            HoldemGame::River(g) => g.root(),
            HoldemGame::TurnRiver(g) => g.root(),
        }
    }
    fn node(&self, s: &HoldemState) -> Node<HoldemState> {
        match self {
            HoldemGame::River(g) => g.node(s),
            HoldemGame::TurnRiver(g) => g.node(s),
        }
    }
    fn sample_chance(&self, s: &HoldemState, rng: &mut dyn FnMut() -> f64) -> Option<HoldemState> {
        match self {
            HoldemGame::River(g) => g.sample_chance(s, rng),
            HoldemGame::TurnRiver(g) => g.sample_chance(s, rng),
        }
    }
}

fn canonical_turn_board() -> [u8; 4] {
    let c = crate::hand_eval::card;
    [c(12, 3), c(11, 3), c(10, 1), c(9, 0)]
}

fn ten_highest_range4(turn_board: &[u8; 4]) -> Vec<[u8; 2]> {
    let mut cards = Vec::with_capacity(10);
    'outer: for r in (0..13u8).rev() {
        for s in 0..4u8 {
            let c = crate::hand_eval::card(r, s);
            if !turn_board.contains(&c) {
                cards.push(c);
                if cards.len() == 10 {
                    break 'outer;
                }
            }
        }
    }
    let mut range = Vec::with_capacity(45);
    for i in 0..cards.len() {
        for j in (i + 1)..cards.len() {
            range.push([cards[i], cards[j]]);
        }
    }
    range
}

pub fn canonical_turn_river(deal_mode: RiverDeal) -> TurnRiverEndgame {
    let tb = canonical_turn_board();
    let range = ten_highest_range4(&tb);
    TurnRiverEndgame::new(
        HoldemRules::river_small(),
        tb,
        range.clone(),
        range,
        deal_mode,
    )
}

pub fn turn_river_game(suffix: &str) -> Option<TurnRiverEndgame> {
    let mode = if suffix.is_empty() {
        RiverDeal::Exact
    } else if let Some(k) = suffix.strip_prefix("_b") {
        RiverDeal::Bucketed(k.parse().ok()?)
    } else {
        return None;
    };
    Some(canonical_turn_river(mode))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::{best_response_p1, safety_verify, solve_blueprint};
    use crate::payoff::{build, PayoffMatrix};
    use crate::sequence_form::compile;

    fn nash_saddle_holds<G: Game<State = HoldemState>>(game: &G) {
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
    fn toy_is_an_exact_nash_zero_exploitability() {
        nash_saddle_holds(&HoldemToy);
    }

    #[test]
    fn small_is_an_exact_nash_zero_exploitability() {
        nash_saddle_holds(&HoldemSmall);
    }

    #[test]
    fn sizes_are_well_formed_and_symmetric() {
        let toy = sizes(&HoldemToy, 0);
        assert_eq!(toy, sizes(&HoldemToy, 1));
        assert!(toy.0 > 10 && toy.1 > 5, "toy too small: {toy:?}");
        let small = sizes(&HoldemSmall, 0);
        assert!(small.0 > toy.0, "river_small must be larger than river_toy");
    }

    #[test]
    fn fcpa_action_menu_is_correct_at_the_root_decision() {
        let rules = HoldemRules::river_toy();
        let mut s = rules.root();

        if let Node::Chance(outcomes) = rules.node(&s) {
            s = outcomes[0].1.clone();
        }
        match rules.node(&s) {
            Node::Decision {
                player, actions, ..
            } => {
                assert_eq!(player, 0);
                let chars: Vec<char> = actions.iter().map(|(c, _)| *c).collect();
                assert_eq!(chars, vec!['c', 'p', 'a'], "expected check / pot / all-in");
            }
            _ => panic!("expected a decision node after the deal"),
        }
    }

    #[test]
    fn pot_bet_then_facing_player_can_fold_call_or_raise() {
        let rules = HoldemRules::river_toy();
        let mut s = rules.root();
        if let Node::Chance(outcomes) = rules.node(&s) {
            s = outcomes[0].1.clone();
        }

        if let Node::Decision { actions, .. } = rules.node(&s) {
            let (_, next) = actions.iter().find(|(c, _)| *c == 'p').unwrap();
            s = next.clone();
        }
        match rules.node(&s) {
            Node::Decision {
                player, actions, ..
            } => {
                assert_eq!(player, 1);
                let chars: Vec<char> = actions.iter().map(|(c, _)| *c).collect();
                assert!(chars.contains(&'f'), "must be able to fold");
                assert!(chars.contains(&'c'), "must be able to call");
                assert!(chars.contains(&'a'), "must be able to shove");
            }
            _ => panic!("expected a decision node facing the bet"),
        }
    }

    #[test]
    fn fold_forfeits_committed_chips() {
        let rules = HoldemRules::river_toy();
        let mut s = rules.root();
        if let Node::Chance(outcomes) = rules.node(&s) {
            s = outcomes[0].1.clone();
        }

        if let Node::Decision { actions, .. } = rules.node(&s) {
            s = actions.iter().find(|(c, _)| *c == 'p').unwrap().1.clone();
        }

        if let Node::Decision { actions, .. } = rules.node(&s) {
            s = actions.iter().find(|(c, _)| *c == 'f').unwrap().1.clone();
        }
        match rules.node(&s) {
            Node::Terminal(v) => {
                assert_eq!(v, 1.0, "fold payoff should be +ante to player 0");
            }
            _ => panic!("expected a terminal after the fold"),
        }
    }

    use crate::hand_eval::card;
    use crate::holdem::{full_river_range, RiverEndgame};

    fn small_river() -> RiverEndgame {
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
        RiverEndgame::new(HoldemRules::river_toy(), board, range0, range1)
    }

    #[test]
    fn real_card_river_is_an_exact_nash_zero_exploitability() {
        nash_saddle_holds(&small_river());
    }

    #[test]
    fn showdown_uses_the_real_hand_evaluator() {
        let game = small_river();

        let root = game.root();
        let deal = match game.node(&root) {
            Node::Chance(outcomes) => outcomes
                .into_iter()
                .map(|(_, st)| st)
                .find(|st| st.s0 == Some(0) && st.s1 == Some(0))
                .expect("Ts4s vs Ac7c deal must be legal"),
            _ => panic!("root must deal"),
        };

        let mut s = deal;
        for _ in 0..2 {
            if let Node::Decision { actions, .. } = game.node(&s) {
                s = actions.iter().find(|(c, _)| *c == 'c').unwrap().1.clone();
            }
        }
        match game.node(&s) {
            Node::Terminal(v) => assert!(v > 0.0, "spade flush should beat a pair of aces"),
            _ => panic!("expected showdown terminal"),
        }
    }

    #[test]
    fn full_river_range_has_1081_combos() {
        let board = [
            card(12, 3),
            card(11, 3),
            card(10, 1),
            card(9, 0),
            card(7, 2),
        ];
        assert_eq!(full_river_range(&board).len(), 1081);

        let game = RiverEndgame::full(HoldemRules::river_toy(), board);

        assert!(game.num_deals() > 1_000_000, "deals = {}", game.num_deals());
    }

    #[test]
    fn sample_chance_is_bit_identical_to_node_deal() {
        let game = small_river();
        let root = game.root();
        let outcomes = match game.node(&root) {
            Node::Chance(o) => o,
            _ => panic!("root must be a chance deal"),
        };
        let n = outcomes.len();
        for k in 0..200u32 {
            let u = (k as f64 + 0.5) / 200.0;
            let mut draw = || u;
            let sampled = game
                .sample_chance(&root, &mut draw)
                .expect("root is chance");
            let expect_idx = ((u * n as f64) as usize).min(n - 1);
            let expected = &outcomes[expect_idx].1;
            assert_eq!(sampled.s0, expected.s0, "u={u}");
            assert_eq!(sampled.s1, expected.s1, "u={u}");
        }

        let dealt = outcomes[0].1.clone();
        let mut draw = || 0.5;
        assert!(game.sample_chance(&dealt, &mut draw).is_none());
    }

    fn small_turn_river(mode: RiverDeal) -> TurnRiverEndgame {
        let tb = [card(12, 3), card(11, 3), card(10, 1), card(9, 0)];
        let range0 = vec![[card(8, 3), card(2, 3)], [card(7, 0), card(7, 1)]];
        let range1 = vec![[card(12, 0), card(5, 0)], [card(3, 2), card(4, 2)]];
        TurnRiverEndgame::new(HoldemRules::river_toy(), tb, range0, range1, mode)
    }

    #[test]
    fn turn_river_bucketed_is_an_exact_nash_zero_exploitability() {
        nash_saddle_holds(&small_turn_river(RiverDeal::Bucketed(2)));
    }

    #[test]
    fn turn_river_exact_is_an_exact_nash_zero_exploitability() {
        nash_saddle_holds(&small_turn_river(RiverDeal::Exact));
    }

    #[test]
    fn turn_check_check_deals_a_public_river_then_resumes_betting() {
        let game = small_turn_river(RiverDeal::Bucketed(2));
        let mut s = match game.node(&game.root()) {
            Node::Chance(o) => o[0].1.clone(),
            _ => panic!("root must deal"),
        };
        for _ in 0..2 {
            if let Node::Decision { actions, .. } = game.node(&s) {
                s = actions.iter().find(|(c, _)| *c == 'c').unwrap().1.clone();
            } else {
                panic!("expected a turn betting decision");
            }
        }
        match game.node(&s) {
            Node::Chance(outcomes) => {
                assert_eq!(outcomes.len(), 2, "Bucketed(2) deals two river outcomes");
                let r = outcomes[0].1.clone();
                assert_eq!(r.street, 1, "river successor is street 1");
                assert!(r.river.is_some(), "river card assigned");
                assert!(
                    matches!(game.node(&r), Node::Decision { .. }),
                    "street-1 betting resumes after the river"
                );
            }
            _ => panic!("turn close without a fold must deal a river, not terminate"),
        }
    }

    #[test]
    fn turn_fold_ends_the_hand_with_no_river_dealt() {
        let game = small_turn_river(RiverDeal::Bucketed(2));
        let mut s = match game.node(&game.root()) {
            Node::Chance(o) => o[0].1.clone(),
            _ => panic!("root must deal"),
        };

        if let Node::Decision { actions, .. } = game.node(&s) {
            s = actions.iter().find(|(c, _)| *c == 'p').unwrap().1.clone();
        }

        if let Node::Decision { actions, .. } = game.node(&s) {
            s = actions.iter().find(|(c, _)| *c == 'f').unwrap().1.clone();
        }
        match game.node(&s) {
            Node::Terminal(_) => assert!(s.river.is_none(), "a turn fold deals no river"),
            _ => panic!("a turn fold must terminate the hand"),
        }
    }

    #[test]
    fn river_bucketing_shrinks_the_tree_versus_exact() {
        let exact = sizes(&small_turn_river(RiverDeal::Exact), 0);
        let bucketed = sizes(&small_turn_river(RiverDeal::Bucketed(2)), 0);
        assert!(
            bucketed.0 < exact.0,
            "bucketed {bucketed:?} must be smaller than exact {exact:?}"
        );
    }

    #[test]
    fn two_street_is_larger_than_one_street() {
        let one = sizes(&HoldemSmall, 0);
        let two = sizes(&small_turn_river(RiverDeal::Bucketed(2)), 0);
        assert!(
            two.0 > one.0,
            "two-street {two:?} must exceed one-street {one:?}"
        );
    }
}
