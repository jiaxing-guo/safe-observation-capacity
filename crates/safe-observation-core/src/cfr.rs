pub fn regret_matching(regret_sum: &[f64]) -> Vec<f64> {
    let positive: Vec<f64> = regret_sum.iter().map(|&r| r.max(0.0)).collect();
    let total: f64 = positive.iter().sum();
    if total > 0.0 {
        positive.iter().map(|p| p / total).collect()
    } else {
        let n = regret_sum.len().max(1);
        vec![1.0 / n as f64; regret_sum.len()]
    }
}

pub fn normalize(weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        weights.iter().map(|w| w / total).collect()
    } else {
        let n = weights.len().max(1);
        vec![1.0 / n as f64; weights.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_when_no_positive_regret() {
        let s = regret_matching(&[0.0, -1.0]);
        assert!((s[0] - 0.5).abs() < 1e-12);
        assert!((s[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn proportional_to_positive_regret() {
        let s = regret_matching(&[3.0, 1.0]);
        assert!((s[0] - 0.75).abs() < 1e-12);
        assert!((s[1] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn normalize_sums_to_one() {
        let s = normalize(&[2.0, 2.0]);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
