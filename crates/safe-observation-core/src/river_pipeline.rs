use std::collections::HashMap;

use crate::confidence::{self, ConfidenceSet};
use crate::river_range::{PubNode, RangeGame};
use crate::sequence_form::SequenceForm;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z = z ^ (z >> 31);
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n - 1)
    }
}

pub fn overfold_opponent(
    rg: &RangeGame,
    base: &HashMap<String, Vec<f64>>,
    boost: f64,
) -> HashMap<String, Vec<f64>> {
    let mut out = base.clone();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let Some(fold_at) = children.iter().position(|&(a, _)| a == 'f') else {
            continue;
        };
        for k in 0..rg.combos(1) {
            let label = rg.label(1, k, hist);
            let m = children.len();
            let dist = out.entry(label).or_insert_with(|| vec![1.0 / m as f64; m]);
            let f = dist[fold_at];
            let new_f = f + boost * (1.0 - f);
            let scale = if 1.0 - f > 0.0 {
                (1.0 - new_f) / (1.0 - f)
            } else {
                0.0
            };
            for (a, p) in dist.iter_mut().enumerate() {
                *p = if a == fold_at { new_f } else { *p * scale };
            }
        }
    }
    out
}

pub fn grouped_pin_confidence(
    rg: &RangeGame,
    sf1: &SequenceForm,
    counts: &SimCounts,
    probe_policy: &HashMap<String, Vec<f64>>,
    quantiles: usize,
    delta: f64,
) -> ConfidenceSet {
    let weights = reach_weights(rg, probe_policy);
    let children: HashMap<&str, &Vec<(char, usize)>> = sf1
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), &i.children))
        .collect();
    let n1 = rg.combos(1);
    let scores = rg.game.scores(1);
    let mut order: Vec<usize> = (0..n1).collect();
    order.sort_by_key(|&k| scores[k]);
    let mut group_of = vec![0usize; n1];
    for (rank, &k) in order.iter().enumerate() {
        group_of[k] = (rank * quantiles / n1).min(quantiles - 1);
    }

    let candidates: usize = rg
        .tree
        .nodes
        .iter()
        .filter_map(|n| match n {
            PubNode::Decision {
                player: 1,
                children,
                ..
            } => Some(children.iter().filter(|&&(a, _)| a != 'f').count() * quantiles),
            _ => None,
        })
        .sum();
    let delta_row = delta / (2.0 * candidates.max(1) as f64);

    let empty = vec![0u64; 8];
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut meta: Vec<(String, usize)> = Vec::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children: node_children,
        } = node
        else {
            continue;
        };
        for (ai, &(a, _)) in node_children.iter().enumerate() {
            if a == 'f' {
                continue;
            }
            for g in 0..quantiles {
                let mut members: Vec<(usize, f64)> = Vec::new();
                let mut c = 0u64;
                let mut w_total = 0.0;
                for (k, &gk) in group_of.iter().enumerate() {
                    if gk != g {
                        continue;
                    }
                    let label = rg.label(1, k, hist);
                    let Some(&w) = weights.get(&label) else {
                        continue;
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    let Some(kids) = children.get(label.as_str()) else {
                        continue;
                    };
                    members.push((kids[ai].1, w));
                    w_total += w;
                    c += counts.revealed.get(&label).unwrap_or(&empty)[ai];
                }
                if members.is_empty() {
                    continue;
                }
                let p = c as f64 / counts.hands as f64;
                let eps = bernstein(c, counts.hands, delta_row);
                let (lo, hi) = ((p - eps).max(0.0), p + eps);
                let key = format!("{hist}|grp{g}");
                if hi < w_total {
                    let row = h.len();
                    entries.extend(members.iter().map(|&(col, w)| (row, col, w)));
                    h.push(hi);
                    meta.push((key.clone(), ai));
                }
                if lo > 0.0 {
                    let row = h.len();
                    entries.extend(members.iter().map(|&(col, w)| (row, col, -w)));
                    h.push(-lo);
                    meta.push((key, ai));
                }
            }
        }
    }
    confidence::build_linear(sf1, entries, h, meta)
}

pub fn reach_invariant_leak(
    rg: &RangeGame,
    base: &HashMap<String, Vec<f64>>,
    deploy: &HashMap<String, Vec<f64>>,
    shift: f64,
) -> HashMap<String, Vec<f64>> {
    let mut out = base.clone();
    let weights = reach_weights(rg, deploy);
    let scores = rg.game.scores(1);
    let mut order: Vec<usize> = (0..rg.combos(1)).collect();
    order.sort_by_key(|&k| scores[k]);
    let strong: std::collections::HashSet<usize> =
        order[order.len() / 2..].iter().copied().collect();

    let mut stack: Vec<(usize, bool)> = vec![(0, false)];
    while let Some((id, p1_acted)) = stack.pop() {
        let PubNode::Decision {
            player,
            hist,
            children,
        } = &rg.tree.nodes[id]
        else {
            continue;
        };
        if *player != 1 {
            for &(_a, child) in children {
                stack.push((child, p1_acted));
            }
            continue;
        }
        for &(_a, child) in children {
            stack.push((child, true));
        }
        if p1_acted {
            continue;
        }
        let fold_at = children.iter().position(|&(a, _)| a == 'f');
        let call_at = children.iter().position(|&(a, _)| a == 'c');
        let (Some(fold_at), Some(call_at)) = (fold_at, call_at) else {
            continue;
        };

        if !matches!(rg.tree.nodes[children[call_at].1], PubNode::Showdown { .. }) {
            continue;
        }
        let m = children.len();

        let mut weak_fold_w = 0.0;
        let mut strong_call_w = 0.0;
        for k in 0..rg.combos(1) {
            let w = weights.get(&rg.label(1, k, hist)).copied().unwrap_or(0.0);
            let dist = out
                .entry(rg.label(1, k, hist))
                .or_insert_with(|| vec![1.0 / m as f64; m]);
            if strong.contains(&k) {
                strong_call_w += w * dist[call_at];
            } else {
                weak_fold_w += w * dist[fold_at];
            }
        }
        if weak_fold_w <= 0.0 || strong_call_w <= 0.0 {
            continue;
        }

        let s = shift;
        let t = (s * weak_fold_w / strong_call_w).min(1.0);
        let s = t * strong_call_w / weak_fold_w;
        for k in 0..rg.combos(1) {
            let dist = out.get_mut(&rg.label(1, k, hist)).expect("inserted above");
            if strong.contains(&k) {
                let d = t * dist[call_at];
                dist[call_at] -= d;
                dist[fold_at] += d;
            } else {
                let d = s * dist[fold_at];
                dist[fold_at] -= d;
                dist[call_at] += d;
            }
        }
    }
    out
}

pub fn revealed_call_grouped_opponent(
    rg: &RangeGame,
    base: &HashMap<String, Vec<f64>>,
    deploy: &HashMap<String, Vec<f64>>,
    shift: f64,
) -> HashMap<String, Vec<f64>> {
    let mut out = base.clone();
    let weights = reach_weights(rg, deploy);
    let scores = rg.game.scores(1);
    let mut order: Vec<usize> = (0..rg.combos(1)).collect();
    order.sort_by_key(|&k| scores[k]);
    let strong: std::collections::HashSet<usize> =
        order[order.len() / 2..].iter().copied().collect();

    let mut stack: Vec<(usize, bool)> = vec![(0, false)];
    while let Some((id, p1_acted)) = stack.pop() {
        let PubNode::Decision {
            player,
            hist,
            children,
        } = &rg.tree.nodes[id]
        else {
            continue;
        };
        if *player != 1 {
            for &(_a, child) in children {
                stack.push((child, p1_acted));
            }
            continue;
        }
        for &(_a, child) in children {
            stack.push((child, true));
        }
        let fold_at = children.iter().position(|&(a, _)| a == 'f');
        let call_at = children.iter().position(|&(a, _)| a == 'c');
        let (Some(fold_at), Some(call_at)) = (fold_at, call_at) else {
            continue;
        };
        if p1_acted {
            continue;
        }

        let m = children.len();
        let mut moved = 0.0;
        let mut weak_mass = 0.0;
        for k in 0..rg.combos(1) {
            let w = weights.get(&rg.label(1, k, hist)).copied().unwrap_or(0.0);
            let dist = out
                .entry(rg.label(1, k, hist))
                .or_insert_with(|| vec![1.0 / m as f64; m]);
            if strong.contains(&k) {
                let d = shift * dist[fold_at];
                dist[fold_at] -= d;
                dist[call_at] += d;
                moved += w * d;
            } else {
                weak_mass += w * dist[call_at];
            }
        }
        if weak_mass > 0.0 {
            let scale = (moved / weak_mass).min(1.0);
            for k in 0..rg.combos(1) {
                if strong.contains(&k) {
                    continue;
                }
                let dist = out.get_mut(&rg.label(1, k, hist)).expect("inserted above");
                let d = scale * dist[call_at];
                dist[call_at] -= d;
                dist[fold_at] += d;
            }
        }
    }
    out
}

pub fn revealed_call_opponent(
    rg: &RangeGame,
    base: &HashMap<String, Vec<f64>>,
    shift: f64,
) -> HashMap<String, Vec<f64>> {
    let mut out = base.clone();
    let scores = rg.game.scores(1);
    let mut order: Vec<usize> = (0..rg.combos(1)).collect();
    order.sort_by_key(|&k| scores[k]);
    let strong: std::collections::HashSet<usize> =
        order[order.len() / 2..].iter().copied().collect();

    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let Some(fold_at) = children.iter().position(|&(a, _)| a == 'f') else {
            continue;
        };
        let Some(call_at) = children.iter().position(|&(a, _)| a == 'c') else {
            continue;
        };
        let m = children.len();

        let mut moved = 0.0;
        let mut weak_call_mass = 0.0;
        for k in 0..rg.combos(1) {
            let label = rg.label(1, k, hist);
            let dist = out.entry(label).or_insert_with(|| vec![1.0 / m as f64; m]);
            if strong.contains(&k) {
                let d = shift * dist[fold_at];
                dist[fold_at] -= d;
                dist[call_at] += d;
                moved += d;
            } else {
                weak_call_mass += dist[call_at];
            }
        }

        if weak_call_mass > 0.0 {
            let scale = (moved / weak_call_mass).min(1.0);
            for k in 0..rg.combos(1) {
                if strong.contains(&k) {
                    continue;
                }
                let label = rg.label(1, k, hist);
                let dist = out.get_mut(&label).expect("inserted in pass 1");
                let d = scale * dist[call_at];
                dist[call_at] -= d;
                dist[fold_at] += d;
            }
        }
    }
    out
}

pub struct SimCounts {
    pub public: HashMap<String, (u64, Vec<u64>)>,

    pub revealed: HashMap<String, Vec<u64>>,

    pub hands: u64,
}

pub fn simulate_counts(
    rg: &RangeGame,
    b0: &HashMap<String, Vec<f64>>,
    b1: &HashMap<String, Vec<f64>>,
    n: u64,
    seed: u64,
) -> SimCounts {
    let mut rng = SplitMix64(seed);
    let mut public: HashMap<String, (u64, Vec<u64>)> = HashMap::new();
    let mut revealed: HashMap<String, Vec<u64>> = HashMap::new();
    let (n0, n1) = (rg.combos(0), rg.combos(1));

    for _ in 0..n {
        let (i, j) = loop {
            let i = rng.below(n0);
            let j = rng.below(n1);
            if rg.compatible(i, j) {
                break (i, j);
            }
        };
        let mut node_id = 0usize;
        let mut pending: Vec<(String, usize)> = Vec::new();
        loop {
            match &rg.tree.nodes[node_id] {
                PubNode::Fold { .. } => break,
                PubNode::Showdown { .. } => {
                    for (label, action) in &pending {
                        let m = revealed.entry(label.clone()).or_insert_with(|| vec![0; 8]);
                        m[*action] += 1;
                    }
                    break;
                }
                PubNode::Decision {
                    player,
                    hist,
                    children,
                } => {
                    let k = if *player == 0 { i } else { j };
                    let behavior = if *player == 0 { b0 } else { b1 };
                    let label = rg.label(*player, k, hist);
                    let m = children.len();
                    let dist = behavior.get(&label);
                    let u = rng.next_f64();
                    let mut acc = 0.0;
                    let mut chosen = m - 1;
                    for a in 0..m {
                        acc += dist.map_or(1.0 / m as f64, |d| d[a]);
                        if u < acc {
                            chosen = a;
                            break;
                        }
                    }
                    if *player == 1 {
                        let entry = public
                            .entry(hist.clone())
                            .or_insert_with(|| (0, vec![0; m]));
                        entry.0 += 1;
                        entry.1[chosen] += 1;
                        pending.push((label, chosen));
                    }
                    node_id = children[chosen].1;
                }
            }
        }
    }
    SimCounts {
        public,
        revealed,
        hands: n,
    }
}

fn hoeffding(count: u64, delta: f64) -> f64 {
    if count == 0 {
        return 1.0;
    }
    ((1.0 / (2.0 * count as f64)) * (2.0 / delta).ln()).sqrt()
}

fn bernstein(c: u64, n: u64, delta: f64) -> f64 {
    let n_f = n as f64;
    let p = c as f64 / n_f;
    let log_term = (3.0 / delta).ln();
    (2.0 * p * (1.0 - p) * log_term / n_f).sqrt() + 3.0 * log_term / n_f
}

pub fn reach_weights(rg: &RangeGame, b0: &HashMap<String, Vec<f64>>) -> HashMap<String, f64> {
    let mut weights = HashMap::new();
    let n0 = rg.combos(0);
    let mut stack: Vec<(usize, Vec<f64>)> = vec![(0, vec![1.0; n0])];
    while let Some((id, reach)) = stack.pop() {
        let PubNode::Decision {
            player,
            hist,
            children,
        } = &rg.tree.nodes[id]
        else {
            continue;
        };
        if *player == 1 {
            let masses = rg.fold_masses(1, &reach);
            for (k, m) in masses.iter().enumerate() {
                weights.insert(rg.label(1, k, hist), m * rg.deal_weight());
            }
            for &(_a, child) in children {
                stack.push((child, reach.clone()));
            }
        } else {
            for (ai, &(_a, child)) in children.iter().enumerate() {
                let mut r = reach.clone();
                let m = children.len();
                for (k, w) in r.iter_mut().enumerate() {
                    let p = b0
                        .get(&rg.label(0, k, hist))
                        .map_or(1.0 / m as f64, |d| d[ai]);
                    *w *= p;
                }
                stack.push((child, r));
            }
        }
    }
    weights
}

pub fn public_confidence(
    rg: &RangeGame,
    sf1: &SequenceForm,
    counts: &SimCounts,
    b0: &HashMap<String, Vec<f64>>,
    delta: f64,
) -> ConfidenceSet {
    let weights = reach_weights(rg, b0);
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    let mut intervals: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1, hist, ..
        } = node
        else {
            continue;
        };
        groups.insert(
            hist.clone(),
            (0..rg.combos(1)).map(|k| rg.label(1, k, hist)).collect(),
        );
        if let Some((visits, acts)) = counts.public.get(hist) {
            let w = hoeffding(*visits, delta);
            intervals.insert(
                hist.clone(),
                acts.iter()
                    .map(|&c| {
                        let p = c as f64 / (*visits).max(1) as f64;
                        ((p - w).max(0.0), (p + w).min(1.0))
                    })
                    .collect(),
            );
        }
    }
    confidence::build_public(sf1, &groups, &intervals, &weights)
}

pub fn pin_confidence(
    rg: &RangeGame,
    sf1: &SequenceForm,
    counts: &SimCounts,
    probe_policy: &HashMap<String, Vec<f64>>,
    delta: f64,
) -> ConfidenceSet {
    let weights = reach_weights(rg, probe_policy);

    let children: HashMap<&str, Vec<char>> = sf1
        .info_sets
        .iter()
        .map(|i| {
            (
                i.label.as_str(),
                i.children.iter().map(|&(a, _)| a).collect(),
            )
        })
        .collect();
    let empty = vec![0u64; 8];

    let candidates: usize = weights
        .iter()
        .filter(|(_, &w)| w > 0.0)
        .filter_map(|(label, _)| children.get(label.as_str()))
        .map(|acts| acts.iter().filter(|&&a| a != 'f').count())
        .sum();
    let delta_row = delta / (2.0 * candidates.max(1) as f64);
    let mut boxes: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for (label, &w) in &weights {
        if w <= 0.0 {
            continue;
        }
        let Some(actions) = children.get(label.as_str()) else {
            continue;
        };
        let acts = counts.revealed.get(label).unwrap_or(&empty);
        let bounds: Vec<(f64, f64)> = actions
            .iter()
            .enumerate()
            .map(|(ai, &a)| {
                if a == 'f' {
                    return (0.0, 1.0);
                }
                let c = acts[ai];
                let eps = bernstein(c, counts.hands, delta_row);
                let p = c as f64 / counts.hands as f64;
                let lo = ((p - eps) / w).max(0.0);
                let hi = ((p + eps) / w).min(1.0);
                if hi - lo >= 1.0 {
                    (0.0, 1.0)
                } else {
                    (lo, hi)
                }
            })
            .collect();
        boxes.insert(label.clone(), bounds);
    }
    confidence::build_boxes(sf1, &boxes)
}

pub fn sampled_leak_opponent(
    rg: &RangeGame,
    base: &HashMap<String, Vec<f64>>,
    strength: f64,
) -> HashMap<String, Vec<f64>> {
    sampled_leak_opponent_salted(rg, base, strength, 0)
}

pub fn sampled_leak_opponent_salted(
    rg: &RangeGame,
    base: &HashMap<String, Vec<f64>>,
    strength: f64,
    salt: u64,
) -> HashMap<String, Vec<f64>> {
    let mix = salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut out = base.clone();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let m = children.len();
        for k in 0..rg.combos(1) {
            let label = rg.label(1, k, hist);

            let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ mix;
            for b in label.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            if h % 100 >= 55 {
                continue;
            }

            let mut hh: u64 = 0xcbf2_9ce4_8422_2325 ^ mix;
            for b in hist.bytes() {
                hh ^= b as u64;
                hh = hh.wrapping_mul(0x0000_0100_0000_01b3);
            }
            let fold_at = children.iter().position(|&(a, _)| a == 'f');
            let call_at = children.iter().position(|&(a, _)| a == 'c');
            let dir = match (fold_at, call_at) {
                (Some(fi), Some(ci)) => {
                    if (hh >> 3) & 1 == 0 {
                        fi
                    } else {
                        ci
                    }
                }
                (Some(fi), None) => fi,
                (None, Some(ci)) => ci,
                (None, None) => ((hh >> 8) as usize) % m,
            };
            let s = strength * (0.25 + 0.75 * (((h >> 16) % 100) as f64 / 100.0));
            let dist = out.entry(label).or_insert_with(|| vec![1.0 / m as f64; m]);
            for (i, v) in dist.iter_mut().enumerate() {
                *v = (1.0 - s) * *v + if i == dir { s } else { 0.0 };
            }
        }
    }
    out
}

pub fn merge_counts(a: &SimCounts, b: &SimCounts) -> SimCounts {
    let mut public = a.public.clone();
    for (k, (v, acts)) in &b.public {
        let e = public
            .entry(k.clone())
            .or_insert_with(|| (0, vec![0; acts.len()]));
        e.0 += v;
        for (i, c) in acts.iter().enumerate() {
            if e.1.len() <= i {
                e.1.resize(i + 1, 0);
            }
            e.1[i] += c;
        }
    }
    let mut revealed = a.revealed.clone();
    for (k, acts) in &b.revealed {
        let e = revealed
            .entry(k.clone())
            .or_insert_with(|| vec![0; acts.len()]);
        for (i, c) in acts.iter().enumerate() {
            if e.len() <= i {
                e.resize(i + 1, 0);
            }
            e[i] += c;
        }
    }
    SimCounts {
        public,
        revealed,
        hands: a.hands + b.hands,
    }
}

pub fn passive_pin_confidence(
    rg: &RangeGame,
    sf1: &SequenceForm,
    counts: &SimCounts,
    b0: &HashMap<String, Vec<f64>>,
    delta: f64,
) -> ConfidenceSet {
    let weights = reach_weights(rg, b0);
    let empty = vec![0u64; 8];

    let mut closing: HashMap<&str, Vec<usize>> = HashMap::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let idx: Vec<usize> = children
            .iter()
            .enumerate()
            .filter(|&(_, &(a, child))| {
                a != 'f' && matches!(rg.tree.nodes[child], PubNode::Showdown { .. })
            })
            .map(|(ai, _)| ai)
            .collect();
        if !idx.is_empty() {
            closing.insert(hist.as_str(), idx);
        }
    }
    let candidates: usize = weights
        .iter()
        .filter(|(_, &w)| w > 0.0)
        .filter_map(|(label, _)| {
            let hist = label.split('|').nth(1)?;
            closing.get(hist).map(|v| v.len())
        })
        .sum();
    let delta_row = delta / (2.0 * candidates.max(1) as f64);
    let children: HashMap<&str, &Vec<(char, usize)>> = sf1
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), &i.children))
        .collect();
    let mut boxes: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for (label, &w) in &weights {
        if w <= 0.0 {
            continue;
        }
        let Some(hist) = label.split('|').nth(1) else {
            continue;
        };
        let Some(close_idx) = closing.get(hist) else {
            continue;
        };
        let Some(kids) = children.get(label.as_str()) else {
            continue;
        };
        let acts = counts.revealed.get(label).unwrap_or(&empty);
        let bounds: Vec<(f64, f64)> = kids
            .iter()
            .enumerate()
            .map(|(ai, _)| {
                if !close_idx.contains(&ai) {
                    return (0.0, 1.0);
                }
                let c = acts[ai];
                let eps = bernstein(c, counts.hands, delta_row);
                let p = c as f64 / counts.hands as f64;
                let lo = ((p - eps) / w).max(0.0);
                let hi = ((p + eps) / w).min(1.0);
                if hi - lo >= 1.0 {
                    (0.0, 1.0)
                } else {
                    (lo, hi)
                }
            })
            .collect();
        boxes.insert(label.clone(), bounds);
    }
    confidence::build_boxes(sf1, &boxes)
}

pub fn passive_grouped_confidence(
    rg: &RangeGame,
    sf1: &SequenceForm,
    counts: &SimCounts,
    b0: &HashMap<String, Vec<f64>>,
    quantiles: usize,
    delta: f64,
) -> ConfidenceSet {
    let weights = reach_weights(rg, b0);
    let children: HashMap<&str, &Vec<(char, usize)>> = sf1
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), &i.children))
        .collect();
    let n1 = rg.combos(1);
    let scores = rg.game.scores(1);
    let mut order: Vec<usize> = (0..n1).collect();
    order.sort_by_key(|&k| scores[k]);
    let mut group_of = vec![0usize; n1];
    for (rank, &k) in order.iter().enumerate() {
        group_of[k] = (rank * quantiles / n1).min(quantiles - 1);
    }
    let candidates: usize = rg
        .tree
        .nodes
        .iter()
        .filter_map(|n| match n {
            PubNode::Decision {
                player: 1,
                children,
                ..
            } => Some(
                children
                    .iter()
                    .filter(|&&(a, child)| {
                        a != 'f' && matches!(rg.tree.nodes[child], PubNode::Showdown { .. })
                    })
                    .count()
                    * quantiles,
            ),
            _ => None,
        })
        .sum();
    let delta_row = delta / (2.0 * candidates.max(1) as f64);
    let empty = vec![0u64; 8];
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut meta: Vec<(String, usize)> = Vec::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children: node_children,
        } = node
        else {
            continue;
        };
        for (ai, &(a, child)) in node_children.iter().enumerate() {
            if a == 'f' || !matches!(rg.tree.nodes[child], PubNode::Showdown { .. }) {
                continue;
            }
            for g in 0..quantiles {
                let mut members: Vec<(usize, f64)> = Vec::new();
                let mut c = 0u64;
                let mut w_total = 0.0;
                for (k, &gk) in group_of.iter().enumerate() {
                    if gk != g {
                        continue;
                    }
                    let label = rg.label(1, k, hist);
                    let Some(&w) = weights.get(&label) else {
                        continue;
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    let Some(kids) = children.get(label.as_str()) else {
                        continue;
                    };
                    members.push((kids[ai].1, w));
                    w_total += w;
                    c += counts.revealed.get(&label).unwrap_or(&empty)[ai];
                }
                if members.is_empty() {
                    continue;
                }
                let p = c as f64 / counts.hands as f64;
                let eps = bernstein(c, counts.hands, delta_row);
                let (lo, hi) = ((p - eps).max(0.0), p + eps);
                let key = format!("{hist}|passgrp{g}");
                if hi < w_total {
                    let row = h.len();
                    entries.extend(members.iter().map(|&(col, w)| (row, col, w)));
                    h.push(hi);
                    meta.push((key.clone(), ai));
                }
                if lo > 0.0 {
                    let row = h.len();
                    entries.extend(members.iter().map(|&(col, w)| (row, col, -w)));
                    h.push(-lo);
                    meta.push((key, ai));
                }
            }
        }
    }
    confidence::build_linear(sf1, entries, h, meta)
}

pub fn call_down_policy(rg: &RangeGame) -> HashMap<String, Vec<f64>> {
    let mut out = HashMap::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 0,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let call_at = children
            .iter()
            .position(|&(a, _)| a == 'c')
            .expect("check/call is always legal in the fcpa engine");
        let mut dist = vec![0.0; children.len()];
        dist[call_at] = 1.0;
        for k in 0..rg.combos(0) {
            out.insert(rg.label(0, k, hist), dist.clone());
        }
    }
    out
}

pub fn allin_then_call_policy(rg: &RangeGame) -> HashMap<String, Vec<f64>> {
    let mut out = HashMap::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 0,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let facing_bet = children.iter().any(|&(a, _)| a == 'f');
        let at = if facing_bet {
            children
                .iter()
                .position(|&(a, _)| a == 'c')
                .expect("call is always legal facing a bet")
        } else {
            children
                .iter()
                .position(|&(a, _)| a == 'a')
                .or_else(|| children.iter().position(|&(a, _)| a == 'c'))
                .expect("all-in or check is always legal unopened")
        };
        let mut dist = vec![0.0; children.len()];
        dist[at] = 1.0;
        for k in 0..rg.combos(0) {
            out.insert(rg.label(0, k, hist), dist.clone());
        }
    }
    out
}

pub fn bet_then_call_policy(rg: &RangeGame) -> HashMap<String, Vec<f64>> {
    let mut out = HashMap::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 0,
            hist,
            children,
        } = node
        else {
            continue;
        };

        let facing_bet = children.iter().any(|&(a, _)| a == 'f');
        let at = if facing_bet {
            children
                .iter()
                .position(|&(a, _)| a == 'c')
                .expect("call is always legal facing a bet")
        } else {
            children
                .iter()
                .position(|&(a, _)| a == 'p')
                .or_else(|| children.iter().position(|&(a, _)| a == 'c'))
                .expect("bet or check is always legal unopened")
        };
        let mut dist = vec![0.0; children.len()];
        dist[at] = 1.0;
        for k in 0..rg.combos(0) {
            out.insert(rg.label(0, k, hist), dist.clone());
        }
    }
    out
}

pub fn population_pins(
    rg: &RangeGame,
    sf1: &SequenceForm,
    probe_policy: &HashMap<String, Vec<f64>>,
    y_star: &[f64],
) -> ConfidenceSet {
    let weights = reach_weights(rg, probe_policy);
    let mut boxes: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for info in &sf1.info_sets {
        let Some(&w) = weights.get(&info.label) else {
            continue;
        };
        if w <= 0.0 {
            continue;
        }
        boxes.insert(
            info.label.clone(),
            info.children
                .iter()
                .map(|&(a, child)| {
                    if a == 'f' {
                        (0.0, 1.0)
                    } else {
                        ((y_star[child] - 1e-6).max(0.0), y_star[child] + 1e-6)
                    }
                })
                .collect(),
        );
    }
    confidence::build_boxes(sf1, &boxes)
}

pub fn population_grouped(
    rg: &RangeGame,
    sf1: &SequenceForm,
    probe_policy: &HashMap<String, Vec<f64>>,
    y_star: &[f64],
    quantiles: usize,
) -> ConfidenceSet {
    let weights = reach_weights(rg, probe_policy);
    let children: HashMap<&str, &Vec<(char, usize)>> = sf1
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), &i.children))
        .collect();
    let n1 = rg.combos(1);
    let scores = rg.game.scores(1);
    let mut order: Vec<usize> = (0..n1).collect();
    order.sort_by_key(|&k| scores[k]);
    let mut group_of = vec![0usize; n1];
    for (rank, &k) in order.iter().enumerate() {
        group_of[k] = (rank * quantiles / n1).min(quantiles - 1);
    }
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut meta: Vec<(String, usize)> = Vec::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children: node_children,
        } = node
        else {
            continue;
        };
        for (ai, &(a, _)) in node_children.iter().enumerate() {
            if a == 'f' {
                continue;
            }
            for g in 0..quantiles {
                let mut members: Vec<(usize, f64)> = Vec::new();
                let mut s_true = 0.0;
                for (k, &gk) in group_of.iter().enumerate() {
                    if gk != g {
                        continue;
                    }
                    let label = rg.label(1, k, hist);
                    let Some(&w) = weights.get(&label) else {
                        continue;
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    let Some(kids) = children.get(label.as_str()) else {
                        continue;
                    };
                    members.push((kids[ai].1, w));
                    s_true += w * y_star[kids[ai].1];
                }
                if members.is_empty() {
                    continue;
                }
                let key = format!("{hist}|grp{g}");
                let row = h.len();
                entries.extend(members.iter().map(|&(col, w)| (row, col, w)));
                h.push(s_true + 1e-6);
                meta.push((key.clone(), ai));
                let row = h.len();
                entries.extend(members.iter().map(|&(col, w)| (row, col, -w)));
                h.push(-(s_true - 1e-6));
                meta.push((key, ai));
            }
        }
    }
    confidence::build_linear(sf1, entries, h, meta)
}

pub fn population_public(
    rg: &RangeGame,
    sf1: &SequenceForm,
    b0: &HashMap<String, Vec<f64>>,
    y_star: &[f64],
) -> ConfidenceSet {
    let weights = reach_weights(rg, b0);
    let by_label: HashMap<&str, &crate::sequence_form::InfoSet> = sf1
        .info_sets
        .iter()
        .map(|i| (i.label.as_str(), i))
        .collect();
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    let mut intervals: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for node in &rg.tree.nodes {
        let PubNode::Decision {
            player: 1,
            hist,
            children,
        } = node
        else {
            continue;
        };
        let labels: Vec<String> = (0..rg.combos(1)).map(|k| rg.label(1, k, hist)).collect();
        let mut num = vec![0.0; children.len()];
        let mut den = 0.0;
        for label in &labels {
            let (Some(&w), Some(info)) = (weights.get(label), by_label.get(label.as_str())) else {
                continue;
            };
            den += w * y_star[info.parent_seq];
            for (ai, &(_a, child)) in info.children.iter().enumerate() {
                num[ai] += w * y_star[child];
            }
        }
        if den <= 0.0 {
            continue;
        }

        intervals.insert(
            hist.clone(),
            num.iter().map(|n| (n / den, n / den)).collect(),
        );
        groups.insert(hist.clone(), labels);
    }
    confidence::build_public(sf1, &groups, &intervals, &weights)
}

pub fn run_river_experiment(leak: &str, n_hands: u64, quantiles: usize, seed: u64, out_dir: &str) {
    use crate::best_response::treeplex_opt;
    use crate::hand_eval::card;
    use crate::holdem::{HoldemRules, RiverEndgame};
    use crate::payoff_oracle::{PayoffOracle, RangeOracle};
    use crate::river_solve::{RangeCfr, Variant};
    use crate::robust_cuts::{robust_response_cuts, CutParams};
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
    let sf0 = rg.compile_sequence_form(0);
    let sf1 = rg.compile_sequence_form(1);
    let oracle = RangeOracle::new(&rg, &sf0, &sf1);

    let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
    for _ in 0..2000 {
        cfr.iterate();
    }
    let b0 = cfr.average_behavior(0);
    let x_bp = sf0.realization_from_behavior(&b0);
    let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
    let rho = 0.5;
    let v_target = v_ref - rho;

    let base1 = cfr.average_behavior(1);
    let b1_leak = match leak {
        "control" => base1.clone(),
        "overfold" => overfold_opponent(&rg, &base1, 0.35),
        "revealed_call" => revealed_call_opponent(&rg, &base1, 0.6),
        "reach_invariant" => reach_invariant_leak(&rg, &base1, &b0, 0.6),
        "sampled" => sampled_leak_opponent(&rg, &base1, 0.8),
        _ => unreachable!("unknown leak family"),
    };
    let y_leak = sf1.realization_from_behavior(&b1_leak);
    let ceiling = treeplex_opt(&sf0, oracle.a_y(&y_leak), true).value;
    println!("full river cell [{leak}]: v_ref {v_ref:.6}  unsafe BR ceiling vs leak {ceiling:.6}");

    let probes: Vec<HashMap<String, Vec<f64>>> = match leak {
        "revealed_call" => vec![call_down_policy(&rg), bet_then_call_policy(&rg)],
        "reach_invariant" => vec![
            call_down_policy(&rg),
            bet_then_call_policy(&rg),
            allin_then_call_policy(&rg),
        ],
        _ => vec![call_down_policy(&rg)],
    };
    let t0 = Instant::now();
    let deploy = simulate_counts(&rg, &b0, &b1_leak, n_hands, 1000 * seed + 41);
    let per_batch = n_hands / probes.len() as u64;
    let batches: Vec<SimCounts> = probes
        .iter()
        .enumerate()
        .map(|(i, p)| simulate_counts(&rg, p, &b1_leak, per_batch, 1000 * seed + 42 + i as u64))
        .collect();
    println!(
        "simulate 1e5 deploy + {} x {per_batch} probe hands: {:?}",
        probes.len(),
        t0.elapsed()
    );

    let t0 = Instant::now();
    let cpub_only = public_confidence(&rg, &sf1, &deploy, &b0, 1e-3);
    let pin_sets = || {
        probes
            .iter()
            .zip(&batches)
            .map(|(p, s)| pin_confidence(&rg, &sf1, s, p, 1e-3))
    };
    let grp_sets = || {
        probes
            .iter()
            .zip(&batches)
            .map(|(p, s)| grouped_pin_confidence(&rg, &sf1, s, p, quantiles, 1e-3))
    };
    let pin_rows: usize = pin_sets().map(|c| c.nrows).sum();
    let grp_rows: usize = grp_sets().map(|c| c.nrows).sum();
    let cid = pin_sets().fold(
        public_confidence(&rg, &sf1, &deploy, &b0, 1e-3),
        ConfidenceSet::intersect,
    );
    let cid_grp = grp_sets().fold(
        pin_sets().fold(
            public_confidence(&rg, &sf1, &deploy, &b0, 1e-3),
            ConfidenceSet::intersect,
        ),
        ConfidenceSet::intersect,
    );
    println!(
        "build sets: {:?}  (C_pub {} rows; pins {} rows; grouped {} rows; C_id {} rows; C_id+grp {} rows)",
        t0.elapsed(),
        cpub_only.nrows,
        pin_rows,
        grp_rows,
        cid.nrows,
        cid_grp.nrows
    );
    assert!(
        cid_grp.max_violation(&y_leak) <= 5e-3,
        "true leak opponent outside the built set: {}",
        cid_grp.max_violation(&y_leak)
    );

    let a_y_leak = oracle.a_y(&y_leak);
    let realized = |x: &[f64]| -> f64 { x.iter().zip(&a_y_leak).map(|(a, b)| a * b).sum() };
    println!("blueprint realized vs leak: {:.6}", realized(&x_bp));

    let mut arms: Vec<(&str, f64, f64, f64, f64, usize, bool)> = Vec::new();
    let params = CutParams {
        max_wall_s: 600.0,
        ..CutParams::default()
    };
    for (name, conf) in [
        ("C_pub", &cpub_only),
        ("C_id", &cid),
        ("C_id+grp", &cid_grp),
    ] {
        let t0 = Instant::now();
        let sol = robust_response_cuts(&oracle, &sf0, &sf1, conf, v_target, &x_bp, &params);
        let wall = t0.elapsed().as_secs_f64();
        let real = realized(&sol.realization);
        println!(
            "{name} ({} rows): {wall:.1}s  iters {}  certified {:.6}  bound {:.6}  floor {:.6}  realized {real:.6}  converged {}",
            conf.nrows,
            sol.iters,
            sol.certified_value,
            sol.master_bound,
            sol.floor_value,
            sol.converged
        );
        for th in [1e-3, 1e-4, 1e-5] {
            if let Some(row) = sol.trace.iter().find(|r| r.master_ub - r.best_lb <= th) {
                println!(
                    "  gap<{th:.0e} at iter {} (wall {:.1}s)",
                    row.iter, row.wall_s
                );
            }
        }
        arms.push((
            name,
            sol.certified_value,
            real,
            sol.floor_value,
            wall,
            sol.iters,
            sol.converged,
        ));
    }

    let routed = arms
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("two arms");
    println!(
        "routed: {} certified {:.6} realized {:.6}  |  cell price = sim 0.1s + sets 0.02s + solves {:.1}s",
        routed.0,
        routed.1,
        routed.2,
        arms.iter().map(|a| a.4).sum::<f64>(),
    );

    let arm_json: Vec<String> = arms
        .iter()
        .map(|(name, cert, real, floor, wall, iters, conv)| {
            format!(
                "{{\"arm\": \"{name}\", \"certified\": {cert:.9}, \"realized\": {real:.9}, \
                 \"floor\": {floor:.9}, \"wall_s\": {wall:.3}, \"iters\": {iters}, \
                 \"converged\": {conv}}}"
            )
        })
        .collect();
    let json = format!(
        "{{\"game\": \"holdem_river_full\", \"leak\": \"{leak}\", \"seed\": {seed}, \"n_hands\": {n_hands}, \
         \"quantiles\": {quantiles}, \
         \"rho\": {rho}, \"v_ref\": {v_ref:.9}, \"v_target\": {v_target:.9}, \
         \"bp_realized\": {:.9}, \"ceiling\": {ceiling:.9}, \
         \"rows\": {{\"cpub\": {}, \"pins\": {pin_rows}, \"grouped\": {grp_rows}, \
         \"cid\": {}, \"cid_grp\": {}}}, \
         \"arms\": [{}]}}\n",
        realized(&x_bp),
        cpub_only.nrows,
        cid.nrows,
        cid_grp.nrows,
        arm_json.join(", "),
    );
    std::fs::write(
        format!("{out_dir}/full_river_cell_{leak}_n{n_hands}_k{quantiles}_s{seed}.json"),
        json,
    )
    .expect("write checkpoint");
}

pub fn run_passive_river_experiment(
    leak: &str,
    n_hands: u64,
    quantiles: usize,
    seed: u64,
    out_dir: &str,
) {
    use crate::best_response::treeplex_opt;
    use crate::hand_eval::card;
    use crate::holdem::{HoldemRules, RiverEndgame};
    use crate::payoff_oracle::{PayoffOracle, RangeOracle};
    use crate::river_solve::{RangeCfr, Variant};
    use crate::robust_cuts::{robust_response_cuts, CutParams};
    use std::time::Instant;

    let ckpt = format!("{out_dir}/full_river_passive_{leak}_n{n_hands}_k{quantiles}_s{seed}.json");
    if std::fs::metadata(&ckpt)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
    {
        println!("passive cell already done: {ckpt}");
        return;
    }

    let board = [
        card(12, 3),
        card(11, 3),
        card(10, 1),
        card(9, 0),
        card(7, 2),
    ];
    let game = RiverEndgame::full(HoldemRules::river_small(), board);
    let rg = RangeGame::new(&game);
    let sf0 = rg.compile_sequence_form(0);
    let sf1 = rg.compile_sequence_form(1);
    let oracle = RangeOracle::new(&rg, &sf0, &sf1);

    let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
    for _ in 0..2000 {
        cfr.iterate();
    }
    let b0 = cfr.average_behavior(0);
    let x_bp = sf0.realization_from_behavior(&b0);
    let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
    let rho = 0.5;
    let v_target = v_ref - rho;

    let base1 = cfr.average_behavior(1);
    let b1_leak = match leak {
        "control" => base1.clone(),
        "overfold" => overfold_opponent(&rg, &base1, 0.35),
        "revealed_call" => revealed_call_opponent(&rg, &base1, 0.6),
        "reach_invariant" => reach_invariant_leak(&rg, &base1, &b0, 0.6),
        "sampled" => sampled_leak_opponent(&rg, &base1, 0.8),
        _ => unreachable!("unknown leak family"),
    };
    let y_leak = sf1.realization_from_behavior(&b1_leak);
    println!("full river passive cell [{leak}] n={n_hands} seed={seed}: v_ref {v_ref:.6}");

    let t0 = Instant::now();
    let deploy = simulate_counts(&rg, &b0, &b1_leak, n_hands, 1000 * seed + 41);
    println!("simulate {n_hands} deploy hands: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let cpub = public_confidence(&rg, &sf1, &deploy, &b0, 1e-3);
    let pins = passive_pin_confidence(&rg, &sf1, &deploy, &b0, 1e-3);
    let grp = passive_grouped_confidence(&rg, &sf1, &deploy, &b0, quantiles, 1e-3);
    let (pin_rows, grp_rows) = (pins.nrows, grp.nrows);
    let conf = grp.intersect(pins).intersect(cpub);
    println!(
        "build passive sets: {:?}  (pins {pin_rows} rows; grouped {grp_rows} rows; total {} rows)",
        t0.elapsed(),
        conf.nrows
    );

    assert!(
        conf.max_violation(&y_leak) <= 5e-3,
        "true leak opponent outside the passive set: {}",
        conf.max_violation(&y_leak)
    );

    let a_y_leak = oracle.a_y(&y_leak);
    let realized = |x: &[f64]| -> f64 { x.iter().zip(&a_y_leak).map(|(a, b)| a * b).sum() };

    let params = CutParams {
        max_wall_s: 600.0,
        ..CutParams::default()
    };
    let t0 = Instant::now();
    let sol =
        try_robust(|| robust_response_cuts(&oracle, &sf0, &sf1, &conf, v_target, &x_bp, &params));
    let wall = t0.elapsed().as_secs_f64();
    let Some(sol) = sol else {
        let json = format!(
            "{{\"game\": \"holdem_river_full\", \"leak\": \"{leak}\", \"seed\": {seed}, \
             \"n_hands\": {n_hands}, \"quantiles\": {quantiles}, \"rho\": {rho}, \
             \"v_ref\": {v_ref:.9}, \"arms\": [{{\"arm\": \"C_id_passive\", \
             \"certified\": null, \"degenerate\": true, \"wall_s\": {wall:.3}}}]}}\n"
        );
        std::fs::write(&ckpt, json).expect("write checkpoint");
        println!("C_id_passive: DEGENERATE (all retry rungs); recorded");
        return;
    };
    let real = realized(&sol.realization);
    println!(
        "C_id_passive ({} rows): {wall:.1}s  iters {}  certified {:.6}  bound {:.6}  floor {:.6}  realized {real:.6}  converged {}",
        conf.nrows, sol.iters, sol.certified_value, sol.master_bound, sol.floor_value, sol.converged
    );

    let json = format!(
        "{{\"game\": \"holdem_river_full\", \"leak\": \"{leak}\", \"seed\": {seed}, \"n_hands\": {n_hands}, \
         \"quantiles\": {quantiles}, \
         \"rho\": {rho}, \"v_ref\": {v_ref:.9}, \"v_target\": {v_target:.9}, \
         \"bp_realized\": {:.9}, \
         \"rows\": {{\"cpub\": {}, \"passive_pins\": {pin_rows}, \"passive_grouped\": {grp_rows}, \
         \"total\": {}}}, \
         \"arms\": [{{\"arm\": \"C_id_passive\", \"certified\": {:.9}, \"realized\": {real:.9}, \
         \"floor\": {:.9}, \"wall_s\": {wall:.3}, \"iters\": {}, \"converged\": {}}}]}}\n",
        realized(&x_bp),
        conf.nrows - pin_rows - grp_rows,
        conf.nrows,
        sol.certified_value,
        sol.floor_value,
        sol.iters,
        sol.converged,
    );
    std::fs::write(&ckpt, json).expect("write checkpoint");
}

fn try_robust(
    f: impl FnOnce() -> crate::robust_cuts::CutSolution,
) -> Option<crate::robust_cuts::CutSolution> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok()
}

pub fn run_population_river_experiment(class: &str, param: u64, salt: u64, out_dir: &str) {
    use crate::best_response::treeplex_opt;
    use crate::hand_eval::card;
    use crate::holdem::{HoldemRules, RiverEndgame};
    use crate::payoff_oracle::{PayoffOracle, RangeOracle};
    use crate::river_solve::{RangeCfr, Variant};
    use crate::robust_cuts::{linear_objective_cuts, robust_response_cuts, CutParams};
    use std::time::Instant;

    let ckpt = format!("{out_dir}/zoo_holdem_river_full_{class}_p{param}_s{salt}.json");
    if std::fs::metadata(&ckpt)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
    {
        println!("zoo cell already done: {ckpt}");
        return;
    }

    let board = [
        card(12, 3),
        card(11, 3),
        card(10, 1),
        card(9, 0),
        card(7, 2),
    ];
    let game = RiverEndgame::full(HoldemRules::river_small(), board);
    let rg = RangeGame::new(&game);
    let sf0 = rg.compile_sequence_form(0);
    let sf1 = rg.compile_sequence_form(1);
    let oracle = RangeOracle::new(&rg, &sf0, &sf1);

    let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
    for _ in 0..2000 {
        cfr.iterate();
    }
    let b0 = cfr.average_behavior(0);
    let x_bp = sf0.realization_from_behavior(&b0);
    let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
    let rho = 0.5;
    let v_target = v_ref - rho;
    let base1 = cfr.average_behavior(1);

    let b1 = match class {
        "sampled" => sampled_leak_opponent_salted(&rg, &base1, param as f64 / 100.0, salt),
        "weakcfr" => {
            let mut weak = RangeCfr::new(&rg, Variant::CfrPlus);
            for _ in 0..param {
                weak.iterate();
            }
            weak.average_behavior(1)
        }
        _ => unreachable!("unknown zoo class"),
    };
    let y = sf1.realization_from_behavior(&b1);
    let ceiling = treeplex_opt(&sf0, oracle.a_y(&y), true).value;
    println!("zoo [{class} p={param} s={salt}]: v_ref {v_ref:.6}  ceiling {ceiling:.6}");

    let params = CutParams {
        max_wall_s: 600.0,
        ..CutParams::default()
    };
    let t0 = Instant::now();
    let vs = linear_objective_cuts(
        &oracle,
        &sf0,
        &sf1,
        &oracle.a_y(&y),
        v_target,
        &x_bp,
        &params,
    );
    let v_safe = vs.certified_value;
    println!(
        "V_safe = {v_safe:.6} (bound {:.6}, {:?})",
        vs.master_bound,
        t0.elapsed()
    );

    let pub_set = population_public(&rg, &sf1, &b0, &y);
    let t0 = Instant::now();
    let r_pub = try_robust(|| {
        robust_response_cuts(&oracle, &sf0, &sf1, &pub_set, v_target, &x_bp, &params)
    });
    match &r_pub {
        Some(r) => println!(
            "R_pub = {:.6} (bound {:.6}, {} rows, {:?})",
            r.certified_value,
            r.master_bound,
            pub_set.nrows,
            t0.elapsed()
        ),
        None => println!("R_pub: degenerate (all retry rungs)"),
    }

    let probes = [
        call_down_policy(&rg),
        bet_then_call_policy(&rg),
        allin_then_call_policy(&rg),
    ];
    let grp = probes
        .iter()
        .map(|p| population_grouped(&rg, &sf1, p, &y, 4))
        .fold(
            population_public(&rg, &sf1, &b0, &y),
            ConfidenceSet::intersect,
        );
    let t0 = Instant::now();
    let r_grp =
        try_robust(|| robust_response_cuts(&oracle, &sf0, &sf1, &grp, v_target, &x_bp, &params));
    match &r_grp {
        Some(r) => println!(
            "R_grp = {:.6} (bound {:.6}, {} rows, {:?})",
            r.certified_value,
            r.master_bound,
            grp.nrows,
            t0.elapsed()
        ),
        None => println!("R_grp: degenerate (all retry rungs)"),
    }

    let fmt = |r: &Option<crate::robust_cuts::CutSolution>| -> String {
        match r {
            Some(r) => format!(
                "{:.9}, \"bound\": {:.9}, \"converged\": {}",
                r.certified_value, r.master_bound, r.converged
            ),
            None => "null, \"bound\": null, \"converged\": false".to_string(),
        }
    };
    let json = format!(
        "{{\"game\": \"holdem_river_full\", \"class\": \"{class}\", \"param\": {param}, \
         \"salt\": {salt}, \"rho\": {rho}, \"v_ref\": {v_ref:.9}, \"ceiling\": {ceiling:.9}, \
         \"v_safe\": {v_safe:.9}, \"v_safe_bound\": {:.9}, \
         \"r_pub\": {{\"value\": {}}}, \
         \"r_grp\": {{\"value\": {}}}}}\n",
        vs.master_bound,
        fmt(&r_pub),
        fmt(&r_grp),
    );
    std::fs::write(&ckpt, json).expect("write checkpoint");
}

pub fn run_drift_river_experiment(
    pair: &str,
    n_hands: u64,
    quantiles: usize,
    seed: u64,
    out_dir: &str,
) {
    use crate::best_response::treeplex_opt;
    use crate::hand_eval::card;
    use crate::holdem::{HoldemRules, RiverEndgame};
    use crate::payoff_oracle::{PayoffOracle, RangeOracle};
    use crate::river_solve::{RangeCfr, Variant};
    use crate::robust_cuts::{robust_response_cuts, CutParams};
    use std::time::Instant;

    let ckpt = format!("{out_dir}/full_river_drift_{pair}_n{n_hands}_k{quantiles}_s{seed}.json");
    if std::fs::metadata(&ckpt)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
    {
        println!("drift cell already done: {ckpt}");
        return;
    }

    let board = [
        card(12, 3),
        card(11, 3),
        card(10, 1),
        card(9, 0),
        card(7, 2),
    ];
    let game = RiverEndgame::full(HoldemRules::river_small(), board);
    let rg = RangeGame::new(&game);
    let sf0 = rg.compile_sequence_form(0);
    let sf1 = rg.compile_sequence_form(1);
    let oracle = RangeOracle::new(&rg, &sf0, &sf1);

    let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
    for _ in 0..2000 {
        cfr.iterate();
    }
    let b0 = cfr.average_behavior(0);
    let x_bp = sf0.realization_from_behavior(&b0);
    let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
    let rho = 0.5;
    let v_target = v_ref - rho;
    let base1 = cfr.average_behavior(1);

    let (b1_a, b1_b) = match pair {
        "twin_off" => (reach_invariant_leak(&rg, &base1, &b0, 0.6), base1.clone()),
        "twin_on" => (base1.clone(), reach_invariant_leak(&rg, &base1, &b0, 0.6)),
        "overfold_shift" => (
            overfold_opponent(&rg, &base1, 0.15),
            overfold_opponent(&rg, &base1, 0.5),
        ),
        _ => unreachable!("unknown drift pair"),
    };
    let y1 = sf1.realization_from_behavior(&b1_a);
    let y2 = sf1.realization_from_behavior(&b1_b);
    println!("full river drift cell [{pair}] n={n_hands} seed={seed}: v_ref {v_ref:.6}");

    let probes = [
        call_down_policy(&rg),
        bet_then_call_policy(&rg),
        allin_then_call_policy(&rg),
    ];
    let half = n_hands / 2;
    let per_batch = half / probes.len() as u64;
    let t0 = Instant::now();
    let dep1 = simulate_counts(&rg, &b0, &b1_a, half, 1000 * seed + 41);
    let dep2 = simulate_counts(&rg, &b0, &b1_b, half, 1000 * seed + 43);
    let dep_all = merge_counts(&dep1, &dep2);
    let pr1: Vec<SimCounts> = probes
        .iter()
        .enumerate()
        .map(|(i, p)| simulate_counts(&rg, p, &b1_a, per_batch, 1000 * seed + 50 + i as u64))
        .collect();
    let pr2: Vec<SimCounts> = probes
        .iter()
        .enumerate()
        .map(|(i, p)| simulate_counts(&rg, p, &b1_b, per_batch, 1000 * seed + 60 + i as u64))
        .collect();
    let pr_all: Vec<SimCounts> = pr1
        .iter()
        .zip(&pr2)
        .map(|(a, b)| merge_counts(a, b))
        .collect();
    println!("simulate two-phase streams: {:?}", t0.elapsed());

    let build = |dep: &SimCounts, prs: &[SimCounts]| -> ConfidenceSet {
        let mut c = public_confidence(&rg, &sf1, dep, &b0, 1e-3);
        for (p, s) in probes.iter().zip(prs) {
            c = c
                .intersect(pin_confidence(&rg, &sf1, s, p, 1e-3))
                .intersect(grouped_pin_confidence(&rg, &sf1, s, p, quantiles, 1e-3));
        }
        c
    };
    let arms: Vec<(&str, ConfidenceSet)> = vec![
        ("static", build(&dep_all, &pr_all)),
        ("windowed", build(&dep2, &pr2)),
        ("cpub", public_confidence(&rg, &sf1, &dep_all, &b0, 1e-3)),
    ];

    let a_y1 = oracle.a_y(&y1);
    let a_y2 = oracle.a_y(&y2);
    let params = CutParams {
        max_wall_s: 600.0,
        ..CutParams::default()
    };
    let mut rows: Vec<String> = Vec::new();
    for (name, conf) in &arms {
        let viol2 = conf.max_violation(&y2);
        let viol1 = conf.max_violation(&y1);
        let t0 = Instant::now();
        let sol = try_robust(|| {
            robust_response_cuts(&oracle, &sf0, &sf1, conf, v_target, &x_bp, &params)
        });
        let wall = t0.elapsed().as_secs_f64();
        match &sol {
            Some(sol) => {
                let real2: f64 = sol.realization.iter().zip(&a_y2).map(|(a, b)| a * b).sum();
                let real1: f64 = sol.realization.iter().zip(&a_y1).map(|(a, b)| a * b).sum();
                println!(
                    "{name} ({} rows): certified {:.6}  realized_vs_y2 {real2:.6}  floor {:.6}  \
                     viol(y2) {viol2:.2e}  viol(y1) {viol1:.2e}  ({wall:.1}s, conv {})",
                    conf.nrows, sol.certified_value, sol.floor_value, sol.converged
                );
                rows.push(format!(
                    "{{\"arm\": \"{name}\", \"rows\": {}, \"certified\": {:.9}, \
                     \"realized_vs_y2\": {real2:.9}, \"realized_vs_y1\": {real1:.9}, \
                     \"floor\": {:.9}, \"viol_y2\": {viol2:.3e}, \"viol_y1\": {viol1:.3e}, \
                     \"wall_s\": {wall:.3}, \"converged\": {}}}",
                    conf.nrows, sol.certified_value, sol.floor_value, sol.converged
                ));
            }
            None => {
                println!(
                    "{name} ({} rows): DEGENERATE (all retry rungs)  viol(y2) {viol2:.2e}  viol(y1) {viol1:.2e}",
                    conf.nrows
                );
                rows.push(format!(
                    "{{\"arm\": \"{name}\", \"rows\": {}, \"certified\": null, \
                     \"realized_vs_y2\": null, \"realized_vs_y1\": null, \
                     \"floor\": null, \"viol_y2\": {viol2:.3e}, \"viol_y1\": {viol1:.3e}, \
                     \"wall_s\": {wall:.3}, \"degenerate\": true}}",
                    conf.nrows
                ));
            }
        }
    }

    let json = format!(
        "{{\"game\": \"holdem_river_full\", \"pair\": \"{pair}\", \"seed\": {seed}, \
         \"n_hands\": {n_hands}, \"quantiles\": {quantiles}, \"rho\": {rho}, \
         \"v_ref\": {v_ref:.9}, \"arms\": [{}]}}\n",
        rows.join(", ")
    );
    std::fs::write(&ckpt, json).expect("write checkpoint");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holdem::canonical_holdem;
    use crate::river_range::RangeGame;
    use crate::river_solve::{RangeCfr, Variant};

    #[test]
    #[ignore = "full river pipeline pricing; run explicitly in release mode"]
    fn full_river_pipeline_robust_pricing() {
        run_river_experiment(
            "overfold",
            100_000,
            16,
            1,
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
        );
    }

    #[test]
    #[ignore = "full river revealed-call cell; run explicitly in release mode"]
    fn full_river_revealed_call_experiment() {
        run_river_experiment(
            "revealed_call",
            100_000,
            16,
            1,
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
        );
    }

    #[test]
    #[ignore = "N-ladder point; run explicitly in release mode"]
    fn full_river_ladder_revealed_n1e5() {
        for k in [4, 64] {
            run_river_experiment(
                "revealed_call",
                100_000,
                k,
                1,
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
            );
        }
    }
    #[test]
    #[ignore = "N-ladder point; run explicitly in release mode"]
    fn full_river_ladder_revealed_n1e6() {
        for k in [4, 16, 64] {
            run_river_experiment(
                "revealed_call",
                1_000_000,
                k,
                1,
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
            );
        }
    }
    #[test]
    #[ignore = "N-ladder point; run explicitly in release mode"]
    fn full_river_ladder_revealed_n1e7() {
        for k in [4, 16, 64] {
            run_river_experiment(
                "revealed_call",
                10_000_000,
                k,
                1,
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
            );
        }
    }
    #[test]
    #[ignore = "N-ladder point; run explicitly in release mode"]
    fn full_river_ladder_overfold() {
        for n in [1_000_000, 10_000_000] {
            run_river_experiment(
                "overfold",
                n,
                16,
                1,
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
            );
        }
    }

    #[test]
    #[ignore = "N-ladder point; run explicitly in release mode"]
    fn full_river_ladder_reach_invariant() {
        for n in [100_000, 1_000_000, 10_000_000] {
            run_river_experiment(
                "reach_invariant",
                n,
                4,
                1,
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../results"),
            );
        }
    }

    #[test]
    #[ignore = "full river residual computation; run explicitly in release mode"]
    fn full_river_residual_revealed_call() {
        full_river_residual("revealed_call");
    }

    #[test]
    #[ignore = "full river grouped residual; run explicitly in release mode"]
    fn full_river_residual_revealed_call_grouped() {
        full_river_residual("revealed_call_grouped");
    }

    #[test]
    #[ignore = "full river reach-invariant residual; run explicitly in release mode"]
    fn full_river_residual_reach_invariant() {
        full_river_residual("reach_invariant");
    }

    #[test]
    #[ignore = "full river reach-invariant 2-probe residual; run explicitly in release mode"]
    fn full_river_residual_reach_invariant_2probe() {
        full_river_residual("reach_invariant_2probe");
    }

    fn full_river_residual(leak: &'static str) {
        use crate::best_response::treeplex_opt;
        use crate::hand_eval::card;
        use crate::holdem::{HoldemRules, RiverEndgame};
        use crate::payoff_oracle::{PayoffOracle, RangeOracle};
        use crate::robust_cuts::{linear_objective_cuts, robust_response_cuts, CutParams};
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
        let sf0 = rg.compile_sequence_form(0);
        let sf1 = rg.compile_sequence_form(1);
        let oracle = RangeOracle::new(&rg, &sf0, &sf1);
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..2000 {
            cfr.iterate();
        }
        let b0 = cfr.average_behavior(0);
        let x_bp = sf0.realization_from_behavior(&b0);
        let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
        let rho = 0.5;
        let v_target = v_ref - rho;
        let b1_leak = match leak {
            "revealed_call" => revealed_call_opponent(&rg, &cfr.average_behavior(1), 0.6),
            "revealed_call_grouped" => {
                revealed_call_grouped_opponent(&rg, &cfr.average_behavior(1), &b0, 0.6)
            }

            "reach_invariant" | "reach_invariant_2probe" => {
                reach_invariant_leak(&rg, &cfr.average_behavior(1), &b0, 0.6)
            }
            _ => unreachable!(),
        };
        let y_leak = sf1.realization_from_behavior(&b1_leak);

        let params = CutParams::default();

        let ckpt_path = format!(
            "{}/../../results/full_river_residual_{leak}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let (done_names, prior_rows, prior_v, prior_best_rm) =
            match std::fs::read_to_string(&ckpt_path) {
                Ok(s) => {
                    let v = s
                        .split("\"v_safe\": ")
                        .nth(1)
                        .and_then(|t| t.split(',').next())
                        .and_then(|t| t.trim().parse::<f64>().ok());
                    let mut names: Vec<String> = Vec::new();
                    let mut rows: Vec<String> = Vec::new();
                    let mut best = f64::NEG_INFINITY;
                    if let Some(arr) = s.split("\"designs\": [").nth(1) {
                        let arr = arr.rsplit_once(']').map(|(a, _)| a).unwrap_or(arr);
                        for obj in arr.split("}, {").filter(|o| !o.trim().is_empty()) {
                            let inner = obj.trim().trim_matches(|c| c == '{' || c == '}');
                            if let Some(n) = inner
                                .split("\"design\": \"")
                                .nth(1)
                                .and_then(|t| t.split('"').next())
                            {
                                names.push(n.to_string());
                                rows.push(format!("{{{inner}}}"));
                            }
                            if let Some(rm) = inner
                                .split("\"r_m\": ")
                                .nth(1)
                                .and_then(|t| t.split(',').next())
                                .and_then(|t| t.trim().parse::<f64>().ok())
                            {
                                best = best.max(rm);
                            }
                        }
                    }
                    (names, rows, v, best)
                }
                Err(_) => (Vec::new(), Vec::new(), None, f64::NEG_INFINITY),
            };
        if !done_names.is_empty() {
            println!(
                "resuming: {} designs from checkpoint {done_names:?}",
                done_names.len()
            );
        }

        let t0 = Instant::now();
        let v_safe_val = match prior_v {
            Some(v) => {
                println!("safe BR V = {v:.6} (resumed from checkpoint)");
                v
            }
            None => {
                let v_safe = linear_objective_cuts(
                    &oracle,
                    &sf0,
                    &sf1,
                    &oracle.a_y(&y_leak),
                    v_target,
                    &x_bp,
                    &params,
                );
                println!(
                    "safe BR V = {:.6} (bound {:.6}, {:?})",
                    v_safe.certified_value,
                    v_safe.master_bound,
                    t0.elapsed()
                );
                v_safe.certified_value
            }
        };

        let probes: Vec<HashMap<String, Vec<f64>>> = match leak {
            "reach_invariant" => vec![
                call_down_policy(&rg),
                bet_then_call_policy(&rg),
                allin_then_call_policy(&rg),
            ],
            _ => vec![call_down_policy(&rg), bet_then_call_policy(&rg)],
        };
        let pop_pub = || population_public(&rg, &sf1, &b0, &y_leak);
        let rows: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(prior_rows);

        let flush = |rows: &[String]| {
            let json = format!(
                "{{\"game\": \"holdem_river_full\", \"leak\": \"{leak}\", \
                 \"v_ref\": {v_ref:.9}, \"rho\": {rho}, \"v_safe\": {v_safe_val:.9}, \
                 \"designs\": [{}]}}\n",
                rows.join(", ")
            );
            std::fs::write(&ckpt_path, json).expect("write checkpoint");
        };
        let best_solved_rm = std::cell::Cell::new(prior_best_rm);
        let run = |name: &str, conf: ConfidenceSet| {
            if done_names.iter().any(|n| n == name) {
                println!("{name}: resumed from checkpoint");
                return;
            }
            let t0 = Instant::now();
            let sol = robust_response_cuts(&oracle, &sf0, &sf1, &conf, v_target, &x_bp, &params);
            best_solved_rm.set(best_solved_rm.get().max(sol.certified_value));
            let delta = v_safe_val - sol.certified_value;
            println!(
                "{name} ({} rows): R_M = {:.6} (bound {:.6})  Delta^- = {delta:.6}  ({:?}, iters {}, converged {})",
                conf.nrows,
                sol.certified_value,
                sol.master_bound,
                t0.elapsed(),
                sol.iters,
                sol.converged
            );
            rows.borrow_mut().push(format!(
                "{{\"design\": \"{name}\", \"rows\": {}, \"r_m\": {:.9}, \"r_bound\": {:.9}, \
                 \"delta_minus\": {delta:.9}, \"converged\": {}}}",
                conf.nrows, sol.certified_value, sol.master_bound, sol.converged
            ));
            flush(&rows.borrow());
        };
        run("public_only", pop_pub());
        let ks: &[usize] = if leak.starts_with("reach_invariant") {
            &[4]
        } else {
            &[4, 16, 64]
        };
        for &k in ks {
            let grp = probes
                .iter()
                .map(|p| population_grouped(&rg, &sf1, p, &y_leak, k))
                .reduce(ConfidenceSet::intersect)
                .expect("at least one probe");
            run(&format!("grouped_k{k}"), pop_pub().intersect(grp));
        }
        let pins = probes
            .iter()
            .map(|p| population_pins(&rg, &sf1, p, &y_leak))
            .reduce(ConfidenceSet::intersect)
            .expect("at least one probe");

        let percombo = pop_pub().intersect(pins);
        let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run("percombo", percombo);
        }))
        .is_ok();
        if !solved {
            let best_rm = best_solved_rm.get();
            let delta = v_safe_val - best_rm;
            println!(
                "percombo: LP degenerate on all retry rungs; monotone bracket R_M >= {best_rm:.6}, Delta^- <= {delta:.6}"
            );
            rows.borrow_mut().push(format!(
                "{{\"design\": \"percombo\", \"rows\": null, \"r_m_lower\": {best_rm:.9}, \
                 \"delta_minus_upper\": {delta:.9}, \"bracketed_by_monotonicity\": true}}"
            ));
            flush(&rows.borrow());
        }
        let _ = rows.into_inner();
    }

    #[test]
    #[ignore = "full river kappa blueprint-shift overlay; run explicitly in release mode"]
    fn full_river_kappa_blueprint_shift() {
        use crate::best_response::treeplex_opt;
        use crate::hand_eval::card;
        use crate::holdem::{HoldemRules, RiverEndgame};
        use crate::payoff_oracle::{PayoffOracle, RangeOracle};
        use crate::robust_cuts::{linear_objective_cuts, CutParams};

        let board = [
            card(12, 3),
            card(11, 3),
            card(10, 1),
            card(9, 0),
            card(7, 2),
        ];
        let game = RiverEndgame::full(HoldemRules::river_small(), board);
        let rg = RangeGame::new(&game);
        let sf0 = rg.compile_sequence_form(0);
        let sf1 = rg.compile_sequence_form(1);
        let oracle = RangeOracle::new(&rg, &sf0, &sf1);

        let mut plans = Vec::new();
        for iters in [50u32, 300, 2000] {
            let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
            for _ in 0..iters {
                cfr.iterate();
            }
            let x = sf0.realization_from_behavior(&cfr.average_behavior(0));
            let v = treeplex_opt(&sf1, oracle.at_x(&x), false).value;
            println!("blueprint iters {iters}: exact v_ref {v:.6}");
            plans.push((iters, x, v));
        }
        let (_, x_anchor, v_anchor) = plans.last().cloned().expect("anchor");

        let (node_id, hist) = rg
            .tree
            .nodes
            .iter()
            .enumerate()
            .find_map(|(id, n)| match n {
                PubNode::Decision {
                    player: 1,
                    hist,
                    children,
                } if children.iter().any(|&(a, _)| a == 'f') => Some((id, hist.clone())),
                _ => None,
            })
            .expect("a facing-a-bet opponent state exists");
        let n1 = rg.combos(1);
        let scores = rg.game.scores(1);
        let mut order: Vec<usize> = (0..n1).collect();
        order.sort_by_key(|&k| scores[k]);
        let targets: Vec<usize> = [0.02, 0.25, 0.5, 0.75, 0.98]
            .iter()
            .map(|q| order[((q * n1 as f64) as usize).min(n1 - 1)])
            .collect();

        let params = CutParams::default();
        let mut max_dev: f64 = 0.0;
        let mut rows: Vec<String> = Vec::new();
        for &k in &targets {
            let hk = rg.game.range(1)[k];
            let mut c = vec![0.0; sf0.num_sequences()];
            for (i, hi) in rg.game.range(0).iter().enumerate() {
                if hi[0] != hk[0] && hi[0] != hk[1] && hi[1] != hk[0] && hi[1] != hk[1] {
                    c[oracle.seq_at(0, node_id, i)] += rg.deal_weight();
                }
            }
            for &(q_iters, ref x_q, v_q) in &plans[..2] {
                let eps = v_anchor - v_q;
                for rho in [0.1, 0.25, 0.4] {
                    let a = linear_objective_cuts(&oracle, &sf0, &sf1, &c, v_q - rho, x_q, &params);
                    let b = linear_objective_cuts(
                        &oracle,
                        &sf0,
                        &sf1,
                        &c,
                        v_anchor - (rho + eps),
                        &x_anchor,
                        &params,
                    );
                    let dev = (a.certified_value - b.certified_value).abs();
                    max_dev = max_dev.max(dev);
                    println!(
                        "target k={k} q={q_iters} rho={rho}: kappa(weak anchor) {:.6} vs kappa(strong, rho+eps) {:.6}  dev {dev:.2e}",
                        a.certified_value, b.certified_value
                    );
                    rows.push(format!(
                        "{{\"target\": {k}, \"bp_iters\": {q_iters}, \"rho\": {rho}, \
                         \"eps\": {eps:.9}, \"kappa_weak\": {:.9}, \"kappa_shifted\": {:.9}}}",
                        a.certified_value, b.certified_value
                    ));

                    let json = format!(
                        "{{\"game\": \"holdem_river_full\", \"state\": \"{hist}\", \
                         \"v_anchor\": {v_anchor:.9}, \"max_dev\": {max_dev:.3e}, \"points\": [{}]}}\n",
                        rows.join(", ")
                    );
                    std::fs::write(
                        concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/../../results/full_river_kappa_shift.json"
                        ),
                        json,
                    )
                    .expect("write checkpoint");
                }
            }
        }
        println!("blueprint-shift overlay: max deviation {max_dev:.2e} (state {hist})");
        assert!(
            max_dev <= 1e-4,
            "blueprint-shift overlay deviates: {max_dev}"
        );
    }

    #[test]
    fn revealed_call_grouped_blinds_public_on_compact_river() {
        use crate::best_response::treeplex_opt;
        use crate::payoff_oracle::{PayoffOracle, RangeOracle};
        use crate::robust_cuts::{linear_objective_cuts, robust_response_cuts, CutParams};
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = rg.compile_sequence_form(0);
        let sf1 = rg.compile_sequence_form(1);
        let oracle = RangeOracle::new(&rg, &sf0, &sf1);
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..400 {
            cfr.iterate();
        }
        let b0 = cfr.average_behavior(0);
        let x_bp = sf0.realization_from_behavior(&b0);
        let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
        let v_target = v_ref - 0.5;
        let b1 = revealed_call_grouped_opponent(&rg, &cfr.average_behavior(1), &b0, 0.6);
        let y = sf1.realization_from_behavior(&b1);
        let params = CutParams::default();
        let v_safe = linear_objective_cuts(
            &oracle,
            &sf0,
            &sf1,
            &oracle.a_y(&y),
            v_target,
            &x_bp,
            &params,
        )
        .certified_value;
        let pub_set = population_public(&rg, &sf1, &b0, &y);

        assert!(
            pub_set.max_violation(&y) <= 1e-6,
            "grouped public coverage: {}",
            pub_set.max_violation(&y)
        );
        let r_pub = robust_response_cuts(&oracle, &sf0, &sf1, &pub_set, v_target, &x_bp, &params)
            .certified_value;
        let delta_pub = v_safe - r_pub;

        println!(
            "grouped compact river: V_safe {v_safe:.6}  R_pub {r_pub:.6}  Delta_pub {delta_pub:.6}  ({:.0}% public-blind)",
            100.0 * delta_pub / v_safe.max(1e-9)
        );

        assert!(
            v_safe <= 1e-6 || delta_pub >= 0.4 * v_safe,
            "grouped opponent not public-blind: Delta_pub {delta_pub} vs V_safe {v_safe}"
        );
    }

    #[test]
    fn reach_invariant_leak_blinds_public_on_compact_river() {
        use crate::best_response::treeplex_opt;
        use crate::payoff_oracle::{PayoffOracle, RangeOracle};
        use crate::robust_cuts::{linear_objective_cuts, robust_response_cuts, CutParams};
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf0 = rg.compile_sequence_form(0);
        let sf1 = rg.compile_sequence_form(1);
        let oracle = RangeOracle::new(&rg, &sf0, &sf1);
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..400 {
            cfr.iterate();
        }
        let b0 = cfr.average_behavior(0);
        let x_bp = sf0.realization_from_behavior(&b0);
        let v_ref = treeplex_opt(&sf1, oracle.at_x(&x_bp), false).value;
        let v_target = v_ref - 0.5;
        let b1 = reach_invariant_leak(&rg, &cfr.average_behavior(1), &b0, 0.6);
        let y = sf1.realization_from_behavior(&b1);
        let params = CutParams::default();

        let pub_set = population_public(&rg, &sf1, &b0, &y);
        let pub_base = population_public(
            &rg,
            &sf1,
            &b0,
            &sf1.realization_from_behavior(&cfr.average_behavior(1)),
        );
        assert!(
            pub_set.max_violation(&y) <= 1e-6,
            "coverage: {}",
            pub_set.max_violation(&y)
        );

        let y_base = sf1.realization_from_behavior(&cfr.average_behavior(1));
        assert!(
            pub_set.max_violation(&y_base) <= 1e-6,
            "base not in leak's public fiber (not reach-invariant): {}",
            pub_set.max_violation(&y_base)
        );
        let _ = pub_base;

        let v_safe = linear_objective_cuts(
            &oracle,
            &sf0,
            &sf1,
            &oracle.a_y(&y),
            v_target,
            &x_bp,
            &params,
        )
        .certified_value;
        let r_pub = robust_response_cuts(&oracle, &sf0, &sf1, &pub_set, v_target, &x_bp, &params)
            .certified_value;
        let delta_pub = v_safe - r_pub;
        println!(
            "reach-invariant compact river: V_safe {v_safe:.6}  R_pub {r_pub:.6}  Delta_pub {delta_pub:.6}  ({:.0}% public-blind)",
            100.0 * delta_pub / v_safe.max(1e-9)
        );
        assert!(
            v_safe <= 1e-4 || delta_pub >= 0.85 * v_safe,
            "reach-invariant leak not public-blind: Delta_pub {delta_pub} vs V_safe {v_safe}"
        );
    }

    #[test]
    fn population_sets_feasible_on_compact_river() {
        use crate::robust_cuts::inner_min_lp;
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf1 = rg.compile_sequence_form(1);
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..300 {
            cfr.iterate();
        }
        let b0 = cfr.average_behavior(0);
        let b1 = revealed_call_opponent(&rg, &cfr.average_behavior(1), 0.6);
        let y = sf1.realization_from_behavior(&b1);
        let cd = call_down_policy(&rg);
        let btc = bet_then_call_policy(&rg);

        let pub_set = population_public(&rg, &sf1, &b0, &y);
        println!(
            "pub {} rows, viol {:.3e}",
            pub_set.nrows,
            pub_set.max_violation(&y)
        );
        assert!(
            pub_set.max_violation(&y) <= 2e-6,
            "population public violates: {}",
            pub_set.max_violation(&y)
        );
        let pins =
            population_pins(&rg, &sf1, &cd, &y).intersect(population_pins(&rg, &sf1, &btc, &y));
        assert!(
            pins.max_violation(&y) <= 2e-6,
            "population pins violate: {}",
            pins.max_violation(&y)
        );
        let grp = population_grouped(&rg, &sf1, &cd, &y, 4)
            .intersect(population_grouped(&rg, &sf1, &btc, &y, 4));
        assert!(
            grp.max_violation(&y) <= 2e-6,
            "population grouped violates: {}",
            grp.max_violation(&y)
        );

        let zero = vec![0.0; sf1.num_sequences()];
        println!("inner: pub only");
        let _ = inner_min_lp(&sf1, &pub_set, &zero);
        println!("inner: pub ok");
        let _ = inner_min_lp(
            &sf1,
            &population_public(&rg, &sf1, &b0, &y).intersect(grp),
            &zero,
        );
        let _ = inner_min_lp(
            &sf1,
            &population_public(&rg, &sf1, &b0, &y).intersect(pins),
            &zero,
        );
    }

    #[test]
    fn pipeline_sets_cover_truth_on_compact_river() {
        let game = canonical_holdem();
        let rg = RangeGame::new(&game);
        let sf1 = rg.compile_sequence_form(1);
        let mut cfr = RangeCfr::new(&rg, Variant::CfrPlus);
        for _ in 0..500 {
            cfr.iterate();
        }
        let b0 = cfr.average_behavior(0);
        let b1 = overfold_opponent(&rg, &cfr.average_behavior(1), 0.4);
        let y_true = sf1.realization_from_behavior(&b1);

        let counts = simulate_counts(&rg, &b0, &b1, 200_000, 2026);
        let cpub = public_confidence(&rg, &sf1, &counts, &b0, 1e-3);
        let probe_policy = call_down_policy(&rg);
        let probe = simulate_counts(&rg, &probe_policy, &b1, 200_000, 2027);
        let pins = pin_confidence(&rg, &sf1, &probe, &probe_policy, 1e-3);
        assert!(cpub.nrows > 0, "public set has no rows");
        assert!(pins.nrows > 0, "pin set has no rows");
        let both = cpub.intersect(pins);
        let viol = both.max_violation(&y_true);
        assert!(
            viol <= 5e-3,
            "true opponent violates the built set by {viol}"
        );
    }
}
