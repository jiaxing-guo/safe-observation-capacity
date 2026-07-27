"""Native primitives for safe-observation experiments. See supplementary Reproducibility for its role in the release workflow."""

try:
    import safe_observation_native as _native
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The compiled extension `safe_observation_native` is not available. Build it with "
        "`uv sync` (recommended) or "
        "`uv run maturin develop --manifest-path crates/safe-observation-python/Cargo.toml`."
    ) from exc


version = _native.version

solve_kuhn = _native.solve_kuhn


blueprint_mccfr = _native.blueprint_mccfr

sequence_form = _native.sequence_form

sequence_form_sizes = _native.sequence_form_sizes

blueprint_lp = _native.blueprint_lp

blueprint_realization = _native.blueprint_realization

safety_verify = _native.safety_verify

best_response = _native.best_response

safety_constrained_best_response = _native.safety_constrained_best_response

restricted_nash_response = _native.restricted_nash_response

robust_safe_response = _native.robust_safe_response

robust_safe_response_public = _native.robust_safe_response_public

robust_safe_response_public_cutting_plane = (
    _native.robust_safe_response_public_cutting_plane
)


robust_safe_response_envelope = _native.robust_safe_response_envelope


robust_safe_response_obs = _native.robust_safe_response_obs


robust_safe_response_linear = _native.robust_safe_response_linear

robust_safe_response_linear_cutting_plane = (
    _native.robust_safe_response_linear_cutting_plane
)


opponent_reach_weights = _native.opponent_reach_weights


agent_showdown_reach = _native.agent_showdown_reach

probe_coefficients = _native.probe_coefficients

robust_safe_response_probe = _native.robust_safe_response_probe


confidence_guarded_point_probe = _native.confidence_guarded_point_probe

confidence_sensitivity = _native.confidence_sensitivity

simulate = _native.simulate


simulate_showdown = _native.simulate_showdown

PayoffMatrix = _native.PayoffMatrix

ConfidenceSet = _native.ConfidenceSet

__all__ = [
    "version",
    "solve_kuhn",
    "blueprint_mccfr",
    "sequence_form",
    "sequence_form_sizes",
    "blueprint_lp",
    "blueprint_realization",
    "safety_verify",
    "best_response",
    "safety_constrained_best_response",
    "restricted_nash_response",
    "robust_safe_response",
    "robust_safe_response_public",
    "robust_safe_response_public_cutting_plane",
    "robust_safe_response_envelope",
    "robust_safe_response_obs",
    "robust_safe_response_linear",
    "robust_safe_response_linear_cutting_plane",
    "opponent_reach_weights",
    "agent_showdown_reach",
    "probe_coefficients",
    "robust_safe_response_probe",
    "confidence_sensitivity",
    "simulate",
    "PayoffMatrix",
    "ConfidenceSet",
]
