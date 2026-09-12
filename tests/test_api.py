import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_detection_evidence_is_real():
    response = client.get(
        "/evidence/detection"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["evidence_class"]
        == "REAL"
    )

    assert (
        payload["dataset"]
        == "MetroPT-3"
    )

    assert (
        payload["detector"]["name"]
        == "pca_reconstruction"
    )

    assert (
        payload["metrics"][
            "timely_incident_recall"
        ]
        == pytest.approx(1.0)
    )

    assert (
        len(payload["incidents"])
        == 4
    )

    assert (
        payload["causal_claim"]
        is False
    )


def test_evidence_summary_separates_classes():
    response = client.get(
        "/evidence/summary"
    )

    assert response.status_code == 200

    payload = response.json()

    assert {
        "REAL",
        "SIMULATED",
        "LITERATURE",
    }.issubset(
        payload[
            "evidence_classes"
        ]
    )

    assert (
        payload["safety"][
            "deployment_mode"
        ]
        == "advisory"
    )

    assert (
        payload["safety"][
            "override_equipment_ctrl"
        ]
        is False
    )

    assert (
        payload["safety"][
            "requires_reoptimization"
        ]
        is True
    )

    assert (
        payload["safety"][
            "open_loop_schedule_approved"
        ]
        is False
    )


def test_balanced_twin_simulation():
    response = client.post(
        "/twin/simulate",
        json={
            "initial_pressure_bar_g":
                7.0,
            "mass_flow_in_kg_s":
                0.095,
            "demand_mass_flow_kg_s":
                0.09,
            "leak_mass_flow_kg_s":
                0.005,
            "duration_seconds":
                120,
            "timestep_seconds":
                1,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["evidence_class"]
        == "SIMULATED"
    )

    assert (
        payload[
            "final_pressure_bar_g"
        ]
        == pytest.approx(7.0)
    )


def test_twin_rejects_nondivisible_duration():
    response = client.post(
        "/twin/simulate",
        json={
            "duration_seconds": 61,
            "timestep_seconds": 10,
        },
    )

    assert response.status_code == 422


def test_optimizer_recommendation_is_advisory():
    demand = (
        [0.070] * 15
        + [0.110] * 15
        + [0.145] * 15
        + [0.090] * 15
    )

    response = client.post(
        "/optimization/recommend",
        json={
            "initial_pressure_bar_g":
                7.0,
            "demand_profile_kg_s":
                demand,
            "leak_mass_flow_kg_s":
                0.005,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["evidence_class"]
        == "SIMULATED"
    )

    assert (
        payload["solver_status"]
        == 0
    )

    assert (
        payload["horizon_intervals"]
        == 60
    )

    assert (
        payload["safety"][
            "deployment_mode"
        ]
        == "advisory"
    )

    assert (
        payload["safety"][
            "override_equipment_ctrl"
        ]
        is False
    )

    assert (
        payload["safety"][
            "valid_for_seconds"
        ]
        == pytest.approx(60.0)
    )

    assert (
        payload["safety"][
            "requires_reoptimization"
        ]
        is True
    )

    assert (
        payload["safety"][
            "open_loop_schedule_approved"
        ]
        is False
    )

    assert (
        len(
            payload[
                "current_actions"
            ]
        )
        == 3
    )

    assert (
        payload[
            "explanation"
        ]["causal_claim"]
        is False
    )
