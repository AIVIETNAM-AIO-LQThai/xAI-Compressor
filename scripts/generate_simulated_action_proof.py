from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ml.optimization.robust_mpc import select_robust_first_action
from ml.optimization.scheduler import OptimizationConfig
from ml.reasoning.inverse_physics import infer_total_outflow
from ml.reasoning.scenarios import generate_physical_scenarios
from ml.reasoning.simulation_validation import (
    assess_outflow_recovery,
    generate_synthetic_pressure_trace,
)
from ml.reasoning.uncertainty import infer_state_uncertainty
from ml.twin.compressor import CompressorSpec
from ml.twin.physics import TwinParameters

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _compressors(payload: dict[str, Any]) -> list[CompressorSpec]:
    return [
        CompressorSpec(
            id=str(item["id"]),
            kind=str(item["kind"]),
            max_mass_flow_kg_s=float(item["max_mass_flow_kg_s"]),
            rated_power_kw=float(item["rated_power_kw"]),
            idle_power_kw=float(item["idle_power_kw"]),
            min_load_fraction=float(item["min_load_fraction"]),
            min_on_seconds=float(item["min_on_seconds"]),
            min_off_seconds=float(item["min_off_seconds"]),
        )
        for item in payload["compressors"]
    ]


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    replay = _load_yaml(ROOT / "configs" / "simulated_action_proof.yaml")
    twin = _load_yaml(ROOT / "configs" / "twin.yaml")
    compressor_cfg = _load_yaml(ROOT / "configs" / "compressors.yaml")
    optimizer_cfg = _load_yaml(ROOT / "configs" / "optimization.yaml")
    project = _load_yaml(ROOT / "configs" / "project.yaml")

    safety = project["safety"]
    if str(safety["deployment_mode"]) != "advisory":
        raise RuntimeError("Simulation proof expects advisory mode.")
    if bool(safety["override_equipment_ctrl"]):
        raise RuntimeError("Simulation proof cannot enable equipment override.")

    air = twin["air"]
    parameters = TwinParameters(
        volume_m3=float(twin["receiver"]["volume_m3"]),
        temperature_k=float(air["temperature_k"]),
        gas_constant_j_per_kg_k=float(air["gas_constant_j_per_kg_k"]),
        ambient_pressure_pa=float(air["ambient_pressure_pa"]),
    )

    simulation = replay["simulation"]
    inflow = tuple(float(value) for value in simulation["inflow_profile_kg_s"])
    truth_outflow = float(simulation["truth_total_outflow_kg_s"])
    interval_seconds = float(simulation["interval_seconds"])
    initial_pressure = float(simulation["initial_pressure_bar_g"])

    pressure_trace = generate_synthetic_pressure_trace(
        initial_pressure_bar_g=initial_pressure,
        inflow_profile_kg_s=inflow,
        total_outflow_kg_s=truth_outflow,
        interval_seconds=interval_seconds,
        parameters=parameters,
    )

    inference = infer_total_outflow(
        pressure_trace,
        inflow,
        interval_seconds=interval_seconds,
        parameters=parameters,
    )

    recovery = assess_outflow_recovery(
        inference,
        truth_total_outflow_kg_s=truth_outflow,
        tolerance_kg_s=float(simulation["recovery_tolerance_kg_s"]),
    )
    if not recovery.passed:
        raise RuntimeError("Synthetic inverse-physics recovery check failed.")

    state = infer_state_uncertainty(
        inference,
        outflow_abs_error_kg_s=float(simulation["outflow_abs_error_kg_s"]),
        independent_demand_kg_s=None,
    )
    if state.leak_identifiable:
        raise RuntimeError("Leak must remain unidentifiable without demand data.")

    scenarios = generate_physical_scenarios(state)

    optimization = optimizer_cfg["optimization"]
    solver = optimization["solver"]
    config = OptimizationConfig(
        interval_seconds=float(optimization["interval_seconds"]),
        reserve_mass_flow_kg_s=float(optimization["reserve_mass_flow_kg_s"]),
        startup_penalty_kwh=float(optimization["startup_penalty_kwh"]),
        overpressure_penalty_kwh_per_bar_hour=float(
            optimization["overpressure_penalty_kwh_per_bar_hour"]
        ),
        terminal_pressure_min_bar_g=float(
            optimization["terminal_pressure_min_bar_g"]
        ),
        time_limit_seconds=float(solver["time_limit_seconds"]),
        mip_rel_gap=float(solver["mip_rel_gap"]),
    )

    pressure_cfg = compressor_cfg["pressure"]
    action = select_robust_first_action(
        scenarios,
        horizon_intervals=int(simulation["robust_horizon_intervals"]),
        initial_pressure_bar_g=initial_pressure,
        parameters=parameters,
        compressors=_compressors(compressor_cfg),
        target_bar_g=float(pressure_cfg["target_bar_g"]),
        safety_min_bar_g=float(pressure_cfg["safety_min_bar_g"]),
        safety_max_bar_g=float(pressure_cfg["safety_max_bar_g"]),
        config=config,
    )
    if not action.selected.robust_safe:
        raise RuntimeError("Simulation action failed the one-step safety gate.")

    payload: dict[str, Any] = {
        "schema_version": "aeroxai.simulated_action_proof.v1",
        "scope": "SIMULATION_ONLY",
        "evidence_class": "SIMULATED",
        "causal_claim": False,
        "connected_to_real_incident": False,
        "deployment": {
            "mode": "advisory",
            "override_equipment_ctrl": False,
        },
        "synthetic_truth": {
            "total_outflow_kg_s": truth_outflow,
            "inflow_profile_kg_s": list(inflow),
            "pressure_trace_bar_g": list(pressure_trace),
            "interval_seconds": interval_seconds,
        },
        "inverse_physics": asdict(inference),
        "recovery_check": asdict(recovery),
        "physical_state_uncertainty": asdict(state),
        "scenario_set": asdict(scenarios),
        "robust_action": asdict(action),
        "claim_boundaries": [
            "This proof is generated from synthetic receiver dynamics, not MetroPT telemetry.",
            "The synthetic physical state is not an estimate of any real MetroPT incident.",
            "Leakage remains unidentifiable because independent process demand is absent.",
            "Energy, pressure, reserve, and safety values are simulated under the current twin.",
            "The result is advisory-only and does not authorize equipment control.",
        ],
    }
    payload["proof_id"] = f"sim_{_digest(payload)[:16]}"

    output_path = ROOT / replay["output"]["proof_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "scope": payload["scope"],
                "connected_to_real_incident": False,
                "recovery_passed": recovery.passed,
                "truth_total_outflow_kg_s": recovery.truth_total_outflow_kg_s,
                "inferred_total_outflow_kg_s": recovery.inferred_total_outflow_kg_s,
                "absolute_recovery_error_kg_s": recovery.absolute_error_kg_s,
                "leak_identifiable": state.leak_identifiable,
                "scenario_count": len(scenarios.scenarios),
                "robust_safe": action.selected.robust_safe,
                "valid_for_seconds": action.valid_for_seconds,
                "commands": dict(action.selected.commands),
                "proof_id": payload["proof_id"],
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
