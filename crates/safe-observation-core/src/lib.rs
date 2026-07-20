pub mod best_response;
pub mod censored_chain;
pub mod cfr;
pub mod confidence;
pub mod game;
pub mod goofspiel;
pub mod hand_eval;
pub mod holdem;
pub mod kuhn;
pub mod leduc;
pub mod lp;
pub mod payoff;
pub mod payoff_oracle;
pub mod probe;
pub mod river_pipeline;
pub mod river_range;
pub mod river_solve;
pub mod robust_cuts;
pub mod robust_lagrangian;
pub mod sequence_form;
pub mod sim;
pub mod solver;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
