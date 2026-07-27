//! Hand eval algorithms for safe observation. See supplementary Reproducibility for its role in the release workflow.

/// Assign average ranks while preserving tied values.
const RANKS: [char; 13] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];

/// Defines the suits constant.
const SUITS: [char; 4] = ['c', 'd', 'h', 's'];

#[inline]
/// Computes rank of.
pub fn rank_of(card: u8) -> u8 {
    card % 13
}

#[inline]
/// Computes suit of.
pub fn suit_of(card: u8) -> u8 {
    card / 13
}

#[inline]
/// Computes card.
pub fn card(rank: u8, suit: u8) -> u8 {
    debug_assert!(rank < 13 && suit < 4);
    suit * 13 + rank
}

/// Computes card str.
pub fn card_str(card: u8) -> String {
    let mut s = String::with_capacity(2);
    s.push(RANKS[rank_of(card) as usize]);
    s.push(SUITS[suit_of(card) as usize]);
    s
}

/// Computes straight high.
fn straight_high(rank_mask: u16) -> Option<u8> {
    let ext = (rank_mask << 1) | ((rank_mask >> 12) & 1);
    for high in (4..=13).rev() {
        let run = 0b1_1111u16 << (high - 4);
        if ext & run == run {
            return Some((high - 1) as u8);
        }
    }
    None
}

/// Computes pack.
fn pack(category: u32, tiebreakers: &[u8]) -> u32 {
    let mut s = category;
    for i in 0..5 {
        let t = tiebreakers.get(i).copied().unwrap_or(0) as u32;
        s = (s << 4) | t;
    }
    s
}

/// Computes top ranks.
fn top_ranks(mask: u16, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for r in (0..13).rev() {
        if mask & (1 << r) != 0 {
            out.push(r as u8);
            if out.len() == n {
                break;
            }
        }
    }
    out
}

/// Computes top ranks excluding.
fn top_ranks_excluding(mask: u16, skip: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for r in (0..13).rev() {
        if skip.contains(&(r as u8)) {
            continue;
        }
        if mask & (1 << r) != 0 {
            out.push(r as u8);
            if out.len() == n {
                break;
            }
        }
    }
    out
}

/// Defines the high card constant.
const HIGH_CARD: u32 = 0;
/// Defines the pair constant.
const PAIR: u32 = 1;
/// Defines the two pair constant.
const TWO_PAIR: u32 = 2;
/// Defines the trips constant.
const TRIPS: u32 = 3;
/// Defines the straight constant.
const STRAIGHT: u32 = 4;
/// Persist the accumulated experiment rows to the output file.
const FLUSH: u32 = 5;
/// Defines the full house constant.
const FULL_HOUSE: u32 = 6;
/// Defines the quads constant.
const QUADS: u32 = 7;
/// Defines the straight flush constant.
const STRAIGHT_FLUSH: u32 = 8;

/// Computes evaluate7.
pub fn evaluate7(cards: &[u8]) -> u32 {
    debug_assert_eq!(cards.len(), 7, "evaluate7 needs exactly 7 cards");
    let mut rank_count = [0u8; 13];
    let mut suit_rank_mask = [0u16; 4];
    let mut rank_mask = 0u16;
    for &c in cards {
        let r = rank_of(c) as usize;
        let s = suit_of(c) as usize;
        rank_count[r] += 1;
        suit_rank_mask[s] |= 1 << r;
        rank_mask |= 1 << r;
    }

    let flush_suit = (0..4).find(|&s| suit_rank_mask[s].count_ones() >= 5);

    if let Some(s) = flush_suit {
        if let Some(high) = straight_high(suit_rank_mask[s]) {
            return pack(STRAIGHT_FLUSH, &[high]);
        }
    }

    let mut quads = Vec::new();
    let mut trips = Vec::new();
    let mut pairs = Vec::new();
    for r in (0..13).rev() {
        match rank_count[r] {
            4 => quads.push(r as u8),
            3 => trips.push(r as u8),
            2 => pairs.push(r as u8),
            _ => {}
        }
    }

    if let Some(&q) = quads.first() {
        let kicker = top_ranks_excluding(rank_mask, &[q], 1);
        return pack(QUADS, &[q, kicker[0]]);
    }

    if let Some(&t) = trips.first() {
        let pair = if trips.len() >= 2 {
            Some(trips[1])
        } else {
            pairs.first().copied()
        };
        if let Some(p) = pair {
            return pack(FULL_HOUSE, &[t, p]);
        }
    }

    if let Some(s) = flush_suit {
        return pack(FLUSH, &top_ranks(suit_rank_mask[s], 5));
    }

    if let Some(high) = straight_high(rank_mask) {
        return pack(STRAIGHT, &[high]);
    }

    if let Some(&t) = trips.first() {
        let k = top_ranks_excluding(rank_mask, &[t], 2);
        return pack(TRIPS, &[t, k[0], k[1]]);
    }

    if pairs.len() >= 2 {
        let (hi, lo) = (pairs[0], pairs[1]);
        let kicker = top_ranks_excluding(rank_mask, &[hi, lo], 1);
        return pack(TWO_PAIR, &[hi, lo, kicker[0]]);
    }

    if let Some(&p) = pairs.first() {
        let k = top_ranks_excluding(rank_mask, &[p], 3);
        return pack(PAIR, &[p, k[0], k[1], k[2]]);
    }

    pack(HIGH_CARD, &top_ranks(rank_mask, 5))
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;

    /// Computes hand.
    fn hand(spec: &[(u8, u8)]) -> Vec<u8> {
        spec.iter().map(|&(r, s)| card(r, s)).collect()
    }

    #[test]
    /// Verifies that category ordering is correct.
    fn category_ordering_is_correct() {
        let high = hand(&[(12, 0), (10, 1), (8, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        let pair = hand(&[(12, 0), (12, 1), (8, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        let two_pair = hand(&[(12, 0), (12, 1), (8, 2), (8, 3), (3, 0), (1, 1), (0, 2)]);
        let trips = hand(&[(12, 0), (12, 1), (12, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        let straight = hand(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (10, 1), (11, 2)]);
        let flush = hand(&[(12, 0), (10, 0), (8, 0), (5, 0), (3, 0), (1, 1), (0, 2)]);
        let full = hand(&[(12, 0), (12, 1), (12, 2), (8, 3), (8, 0), (1, 1), (0, 2)]);
        let quads = hand(&[(12, 0), (12, 1), (12, 2), (12, 3), (8, 0), (1, 1), (0, 2)]);
        let sf = hand(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (10, 1), (11, 2)]);

        let scores = [
            evaluate7(&high),
            evaluate7(&pair),
            evaluate7(&two_pair),
            evaluate7(&trips),
            evaluate7(&straight),
            evaluate7(&flush),
            evaluate7(&full),
            evaluate7(&quads),
            evaluate7(&sf),
        ];
        for w in scores.windows(2) {
            assert!(w[0] < w[1], "category order broken: {} !< {}", w[0], w[1]);
        }
    }

    #[test]
    /// Verifies that higher pair beats lower pair.
    fn higher_pair_beats_lower_pair() {
        let aces = hand(&[(12, 0), (12, 1), (8, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        let kings = hand(&[(11, 0), (11, 1), (8, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        assert!(evaluate7(&aces) > evaluate7(&kings));
    }

    #[test]
    /// Verifies that pair kicker breaks ties.
    fn pair_kicker_breaks_ties() {
        let ace_kick = hand(&[(8, 0), (8, 1), (12, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        let king_kick = hand(&[(8, 0), (8, 1), (11, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        assert!(evaluate7(&ace_kick) > evaluate7(&king_kick));
    }

    #[test]
    /// Verifies that wheel is a straight below six high.
    fn wheel_is_a_straight_below_six_high() {
        let wheel = hand(&[(12, 0), (0, 1), (1, 2), (2, 3), (3, 0), (10, 1), (11, 2)]);
        let six_high = hand(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (10, 1), (11, 2)]);
        let ws = evaluate7(&wheel);
        let ss = evaluate7(&six_high);
        assert_eq!(ws >> 20, STRAIGHT, "wheel must be a straight");
        assert!(ws < ss, "wheel must be the lowest straight");
    }

    #[test]
    /// Verifies that broadway is the top straight.
    fn broadway_is_the_top_straight() {
        let broadway = hand(&[(8, 0), (9, 1), (10, 2), (11, 3), (12, 0), (1, 1), (0, 2)]);
        assert_eq!(evaluate7(&broadway) >> 20, STRAIGHT);

        assert_eq!((evaluate7(&broadway) >> 16) & 0xF, 12);
    }

    #[test]
    /// Verifies that full house uses higher trip as the three.
    fn full_house_uses_higher_trip_as_the_three() {
        let two_trips = hand(&[(7, 0), (7, 1), (7, 2), (6, 3), (6, 0), (6, 1), (0, 2)]);
        let s = evaluate7(&two_trips);
        assert_eq!(s >> 20, FULL_HOUSE);
        assert_eq!((s >> 16) & 0xF, 7, "trip rank should be the nines");
        assert_eq!((s >> 12) & 0xF, 6, "pair rank should be the eights");
    }

    #[test]
    /// Verifies that flush beats a lower flush by top card.
    fn flush_beats_a_lower_flush_by_top_card() {
        let ace_flush = hand(&[(12, 0), (9, 0), (7, 0), (5, 0), (2, 0), (1, 1), (0, 2)]);
        let king_flush = hand(&[(11, 0), (9, 0), (7, 0), (5, 0), (2, 0), (1, 1), (0, 2)]);
        assert!(evaluate7(&ace_flush) > evaluate7(&king_flush));
    }

    #[test]
    /// Verifies that identical hands tie.
    fn identical_hands_tie() {
        let a = hand(&[(12, 0), (12, 1), (8, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        let b = hand(&[(12, 0), (12, 1), (8, 2), (5, 3), (3, 0), (1, 1), (0, 2)]);
        assert_eq!(evaluate7(&a), evaluate7(&b));
    }

    #[test]
    /// Verifies that card helpers round trip.
    fn card_helpers_round_trip() {
        for c in 0..52u8 {
            assert_eq!(card(rank_of(c), suit_of(c)), c);
        }
        assert_eq!(card_str(card(12, 3)), "As");
        assert_eq!(card_str(card(8, 0)), "Tc");
    }
}
