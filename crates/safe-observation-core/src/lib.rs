//! Core algorithms for safe observation capacity. See supplementary Reproducibility for its role in the release workflow.

/// Contains best response implementation details.
pub mod best_response;
/// Contains censored chain implementation details.
pub mod censored_chain;
/// Contains CFR implementation details.
pub mod cfr;
/// Contains confidence implementation details.
pub mod confidence;
/// Contains game implementation details.
pub mod game;
/// Contains goofspiel implementation details.
pub mod goofspiel;
/// Contains hand eval implementation details.
pub mod hand_eval;
/// Contains holdem implementation details.
pub mod holdem;
/// Contains Kuhn implementation details.
pub mod kuhn;
/// Contains Leduc implementation details.
pub mod leduc;
/// Contains linear program implementation details.
pub mod lp;
/// Contains payoff implementation details.
pub mod payoff;
/// Contains payoff oracle implementation details.
pub mod payoff_oracle;
/// Contains probe implementation details.
pub mod probe;
/// Contains river pipeline implementation details.
pub mod river_pipeline;
/// Contains river range implementation details.
pub mod river_range;
/// Contains river solve implementation details.
pub mod river_solve;
/// Contains robust cuts implementation details.
pub mod robust_cuts;
/// Contains robust lagrangian implementation details.
pub mod robust_lagrangian;
/// Contains sequence form implementation details.
pub mod sequence_form;
/// Contains sim implementation details.
pub mod sim;
/// Contains solver implementation details.
pub mod solver;

/// Return the package version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
/// Contains regression tests for this module.
mod tests {
    use super::*;

    #[test]
    /// Verifies that version is nonempty.
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
