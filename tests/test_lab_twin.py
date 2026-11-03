"""The generic receiver twin keeps its synthetic inputs apart (base demand, added demand, leakage) and responds to their sum. Equal totals
must give the same plant response while the decomposition stays distinct and correctly labelled. No recorded data is needed."""
import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.explorer import Explorer
from backend.app.lab import Lab, TwinRun
from backend.app.runtime import load_runtime_config
from ml.control.baseline import BaselineControllerConfig, simulate_baseline


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return Lab(Explorer(tmp_path_factory.mktemp("no_index")))  # the twin never reads the recorded index


def run(lab, **kw):
    return lab.twin(TwinRun(**{"hours": 1, "seed": 3, "onset_minutes": 20, "magnitude_kg_s": 0.02, **kw}))


def arr(r, part, key):
    return np.array(r[part][key], dtype=float)


def test_no_change_adds_neither_demand_nor_leakage(lab):
    r = run(lab, scenario="none")
    assert not arr(r, "inputs", "added_demand_kg_s").any() and not arr(r, "inputs", "leak_kg_s").any()
    assert np.allclose(arr(r, "inputs", "total_outflow_kg_s"), arr(r, "inputs", "base_demand_kg_s"))
    assert r["magnitude_kg_s"] == 0 and r["onset_seconds"] is None and r["summary"]["leak_kg"] == r["summary"]["added_demand_kg"] == 0


@pytest.mark.parametrize(("scenario", "own", "other"), [("demand_step", "added_demand_kg_s", "leak_kg_s"), ("leak_onset", "leak_kg_s", "added_demand_kg_s")])
def test_each_scenario_changes_only_its_own_stream_from_its_onset(lab, scenario, own, other):
    r = run(lab, scenario=scenario, duration_minutes=15)
    t, mine = np.array(r["time_seconds"]), arr(r, "inputs", own)
    on = (t > 20 * 60) & (t <= 35 * 60)
    assert not arr(r, "inputs", other).any()
    assert np.allclose(mine[on], 0.02) and not mine[~on].any()
    assert r["onset_seconds"] == 1200 and r["end_seconds"] == 2100


def test_total_outflow_is_the_sum_of_its_parts(lab):
    for scenario in ("none", "demand_step", "leak_onset"):
        r = run(lab, scenario=scenario)
        parts = arr(r, "inputs", "base_demand_kg_s") + arr(r, "inputs", "added_demand_kg_s") + arr(r, "inputs", "leak_kg_s")
        assert np.allclose(parts, arr(r, "inputs", "total_outflow_kg_s"), atol=2e-6)
        s = r["summary"]
        assert s["total_outflow_kg"] == pytest.approx(s["base_demand_kg"] + s["added_demand_kg"] + s["leak_kg"], rel=1e-12)
    leak = run(lab, scenario="leak_onset")
    assert leak["summary"]["leak_kg"] == pytest.approx(0.02 * 40 * 60)  # 40 minutes at 0.02 kg/s


def test_nothing_differs_from_the_unchanged_run_before_onset(lab):
    base, leak = run(lab, scenario="none"), run(lab, scenario="leak_onset")
    before = np.array(leak["time_seconds"]) <= 20 * 60
    for key in ("pressure_bar_g", "compressor_flow_kg_s", "power_kw"):
        assert np.array_equal(arr(base, "response", key)[before], arr(leak, "response", key)[before])
    assert arr(leak, "response", "compressor_flow_kg_s")[~before].sum() > arr(base, "response", "compressor_flow_kg_s")[~before].sum()


def test_equal_totals_give_the_same_plant_response_and_different_ground_truth(lab):
    demand, leak = run(lab, scenario="demand_step"), run(lab, scenario="leak_onset")
    assert demand["response"] == leak["response"]
    for k in ("pressure_min", "pressure_max", "energy_kwh", "safety_violations", "starts", "total_outflow_kg"):
        assert demand["summary"][k] == leak["summary"][k], k
    assert demand["inputs"]["total_outflow_kg_s"] == leak["inputs"]["total_outflow_kg_s"]
    assert demand["inputs"]["added_demand_kg_s"] == leak["inputs"]["leak_kg_s"] != demand["inputs"]["leak_kg_s"]


def test_driving_the_controller_with_total_outflow_equals_its_own_leak_input():
    """Justifies the adapter: the frozen controller given demand D and a constant leak L behaves as with demand D + L and no leak."""
    rt = load_runtime_config()
    p = rt.pressure
    cfg = BaselineControllerConfig(target_bar_g=p["target_bar_g"], lower_band_bar_g=p["lower_band_bar_g"], upper_band_bar_g=p["upper_band_bar_g"],
                                   safety_min_bar_g=p["safety_min_bar_g"], safety_max_bar_g=p["safety_max_bar_g"], pressure_gain_kg_s_per_bar=0.025)
    demand = list(0.09 * (1 + 0.02 * np.random.default_rng(1).standard_normal(1800)))
    common = {"initial_pressure_bar_g": 7.0, "parameters": rt.twin_parameters, "compressors": rt.compressors, "controller_config": cfg, "timestep_seconds": 1.0}
    a = simulate_baseline(demand, leak_mass_flow_kg_s=0.015, **common)
    b = simulate_baseline([d + 0.015 for d in demand], leak_mass_flow_kg_s=0.0, **common)
    for k in ("pressure_bar_g", "compressor_flow_kg_s", "power_kw", "cumulative_energy_kwh"):
        assert np.allclose(a[k], b[k], rtol=0, atol=1e-9), k
    assert (a.safety_violation == b.safety_violation).all()


def test_runs_are_reproducible_and_the_seed_moves_only_the_noise(lab):
    a, b, c = run(lab, scenario="leak_onset"), run(lab, scenario="leak_onset"), run(lab, scenario="leak_onset", seed=4)
    assert a == b
    assert a["inputs"]["base_demand_kg_s"] != c["inputs"]["base_demand_kg_s"]
    assert a["inputs"]["leak_kg_s"] == c["inputs"]["leak_kg_s"] and a["inputs"]["added_demand_kg_s"] == c["inputs"]["added_demand_kg_s"]
    quiet = run(lab, scenario="none", noise_percent=0)
    assert np.allclose(arr(quiet, "inputs", "base_demand_kg_s"), 0.09)


def test_it_is_labelled_as_a_generic_simulation_and_not_scored(lab):
    r = run(lab, scenario="demand_step")
    assert r["evidence_class"] == "SIMULATED" and r["source"] == "SIMULATED · generic plant" and r["detector"] == "not_applicable"
    assert r["units"]["flow"] == "kg/s" and r["units"]["pressure"] == "bar g" and "demand_kg_s" not in r
    assert len(r["time_seconds"]) == 360 == len(r["response"]["safety_violation"]) and r["time_seconds"][0] == 10
    assert 6.5 < r["summary"]["pressure_min"] and r["summary"]["pressure_max"] < 7.5


def test_a_change_outside_the_run_is_refused():
    with pytest.raises(ValidationError):
        TwinRun(hours=1, onset_minutes=60)
