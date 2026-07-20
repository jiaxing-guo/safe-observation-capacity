""

from pathlib import Path

import pytest

from safe_observation.experiments.config import (
    ConfigRun,
    load_config,
    run_config,
    summarize,
)
from safe_observation.opponents import opponent_from_spec

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_opponent_from_spec_passes_params():
    opp = opponent_from_spec({"type": "static_biased", "bet_prob": 0.2})
    assert opp.name == "static_biased"

    assert all(probs == [0.8, 0.2] for probs in opp.behavior.values())


def test_opponent_from_spec_requires_type():
    with pytest.raises(ValueError):
        opponent_from_spec({"bet_prob": 0.2})


def test_opponent_from_spec_unknown_type():
    with pytest.raises(ValueError):
        opponent_from_spec({"type": "nope"})


def test_shipped_configs_exist():
    assert (CONFIGS / "kuhn" / "blueprint.toml").exists()
    assert (CONFIGS / "kuhn" / "static_opponent.toml").exists()
    assert (CONFIGS / "kuhn" / "safety_ablation.toml").exists()


def test_kuhn_static_config_matches_run_parameters():

    cfg = load_config(CONFIGS / "kuhn" / "static_opponent.toml")
    assert cfg["experiment"]["kind"] == "online_replicated"
    assert cfg["opponent"] == {"type": "static_biased", "bet_prob": 0.1}
    assert cfg["run"]["seeds"] == [42, 43, 44, 45, 46]
    assert cfg["run"]["rounds"] == 300
    assert cfg["run"]["episodes_per_round"] == 100


def test_run_blueprint_config(tmp_path):
    run = run_config(
        {
            "experiment": {"kind": "blueprint", "name": "bp"},
            "solver": {"method": "lp"},
            "output": {"dir": str(tmp_path)},
        }
    )
    assert isinstance(run, ConfigRun)
    assert run.kind == "blueprint"
    assert run.results["value_player1"] == pytest.approx(-1.0 / 18.0, abs=1e-9)
    assert "blueprint value" in summarize(run)


def test_run_online_replicated_config_no_figures(tmp_path):
    run = run_config(
        {
            "experiment": {"kind": "online_replicated", "name": "online_test"},
            "opponent": {"type": "static_biased", "bet_prob": 0.1},
            "run": {"rounds": 8, "episodes_per_round": 40, "seeds": [42, 43]},
            "output": {"dir": str(tmp_path), "figures": False},
        }
    )
    assert run.kind == "online_replicated"
    assert run.results["safety_preserved"]
    assert run.results["exploitation_gain_mean"] > 0.0
    assert run.figures == {}
    assert "gain" in summarize(run)


def test_run_ablation_config(tmp_path):
    run = run_config(
        {
            "experiment": {"kind": "ablation", "name": "abl"},
            "opponent": {"type": "static_biased", "bet_prob": 0.1},
            "run": {"rounds": 6, "episodes_per_round": 30, "seeds": [42]},
            "ablation": {
                "deltas": [0.05, 0.2],
                "eps_safes": [0.0],
                "methods": ["hoeffding"],
            },
            "output": {"dir": str(tmp_path), "figures": False},
        }
    )
    assert run.kind == "ablation"
    assert set(run.results) == {"delta", "eps_safe", "method"}
    assert "ablation over" in summarize(run)


def test_figures_override_renders(tmp_path):
    pytest.importorskip("matplotlib")
    run = run_config(
        {
            "experiment": {"kind": "online_replicated", "name": "online_test"},
            "opponent": {"type": "static_biased", "bet_prob": 0.1},
            "run": {"rounds": 6, "episodes_per_round": 30, "seeds": [42, 43]},
            "output": {"dir": str(tmp_path), "figures": False},
        },
        figures=True,
    )
    assert set(run.figures) == {"value", "safety", "ci_shrinkage", "robust_vs_br"}
    for path in run.figures.values():
        assert path.exists()


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        run_config({"experiment": {"kind": "bogus"}})
