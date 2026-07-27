//! Game algorithms for safe observation. See Preliminaries and Problem Setup.

/// Expand a state into its chance, decision, or terminal game node.
pub enum Node<S> {
    Terminal(f64),

    Chance(Vec<(f64, S)>),

    Decision {
        player: usize,
        infoset: String,
        actions: Vec<(char, S)>,
    },
}

/// Defines the interface for game.
pub trait Game {
    /// Aliases the type used for state.
    type State: Clone;

    /// Return the initial game state.
    fn root(&self) -> Self::State;

    /// Expand a state into its chance, decision, or terminal game node.
    fn node(&self, state: &Self::State) -> Node<Self::State>;

    /// Sample chance.
    fn sample_chance(
        &self,
        state: &Self::State,
        rng: &mut dyn FnMut() -> f64,
    ) -> Option<Self::State> {
        let _ = (state, rng);
        None
    }
}
