pub enum Node<S> {
    Terminal(f64),

    Chance(Vec<(f64, S)>),

    Decision {
        player: usize,
        infoset: String,
        actions: Vec<(char, S)>,
    },
}

pub trait Game {
    type State: Clone;

    fn root(&self) -> Self::State;

    fn node(&self, state: &Self::State) -> Node<Self::State>;

    fn sample_chance(
        &self,
        state: &Self::State,
        rng: &mut dyn FnMut() -> f64,
    ) -> Option<Self::State> {
        let _ = (state, rng);
        None
    }
}
