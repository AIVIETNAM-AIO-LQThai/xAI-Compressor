from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

from ml.data.cobra import semantic_representation_sha256
from ml.detection.common import fit_scaler, transform_frame
from ml.temporal.detector import prepare_temporal_batch
from ml.temporal.energy_aware_training import (
    EnergyAwareTrainingConfig,
    train_energy_aware_tcn,
)
from ml.temporal.tcn import TemporalForecastConfig

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs/cobra_router_v2_adaptation.yaml"
FEATURE_PATH = ROOT / "docs/research/cobra_router_v2_raw_feature_contract.json"
INGESTION_PATH = ROOT / "docs/research/cobra_router_v2_train_ingestion.json"
TRAIN_PATH = ROOT / "data/processed/cobra_router_v2/train_5min.csv"

SKLEARN_ARTIFACT = (
    ROOT / "artifacts/research/cobra_router_v2_train_models.pkl"
)
TCN_ARTIFACT = (
    ROOT / "artifacts/research/cobra_router_v2_tcn.pt"
)
RESULT_PATH = (
    ROOT / "docs/research/cobra_router_v2_train_fit.json"
)

REQUIRED_TRAIN_RESULT_COMMIT = (
    "928da0e678f376cd2ced24dc0bf75483d017dd3e"
)

EXPECTED_FEATURE_SHA = (
    "ba54ffa70cda5690ceaef4d0a58d6d14129dd36a0e03abbc95c20344e3e4a9aa"
)

EXPECTED_TRAIN_SEMANTIC_SHA = (
    "2537fe7a126208f3916d6770d36bcee4640989592232774f0288b512d9d094f8"
)

IMPLEMENTATION_PATHS = [
    "scripts/fit_cobra_train_models.py",
    "tests/test_cobra_train_fit.py",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def expected_sequence_count(
    segment_ids: pd.Series,
    *,
    history_bins: int,
) -> int:
    return int(
        sum(
            max(int(count) - history_bins, 0)
            for count in segment_ids.value_counts().tolist()
        )
    )


def validate_frozen_config(config: dict[str, Any]) -> None:
    pca = config["pca"]
    temporal = config["temporal_model"]
    sequence = temporal["sequence"]
    architecture = temporal["architecture"]
    training = temporal["training"]

    assert pca["family"] == "robust_pca_reconstruction"
    assert pca["scaler"] == "robust"
    assert float(pca["variance_retained"]) == 0.95
    assert float(pca["ewma"]["alpha"]) == 0.20

    assert temporal["family"] == "causal_tcn_next_step_prediction"
    assert temporal["scaler"] == "standard"

    assert int(sequence["bin_minutes"]) == 5
    assert int(sequence["history_bins"]) == 12
    assert int(sequence["prediction_horizon_bins"]) == 1
    assert bool(sequence["exact_contiguity_required"])
    assert not bool(sequence["windows_cross_day_boundary"])

    assert int(architecture["residual_blocks"]) == 3
    assert [int(v) for v in architecture["channels"]] == [32, 32, 32]
    assert int(architecture["kernel_size"]) == 3
    assert [int(v) for v in architecture["dilations"]] == [1, 2, 4]
    assert float(architecture["dropout"]) == 0.10

    assert training["optimizer"] == "AdamW"
    assert float(training["learning_rate"]) == 0.001
    assert float(training["weight_decay"]) == 0.0001
    assert int(training["batch_size"]) == 256
    assert int(training["max_epochs"]) == 50
    assert int(training["early_stopping_patience"]) == 5
    assert int(training["seed"]) == 20260915
    assert float(
        training["validation_fraction_of_train_sequences"]
    ) == 0.20
    assert training["validation_split"] == "chronological_tail"
    assert training["calibration_used_for_training"] is False
    assert training["evaluation_used_for_training"] is False
    assert training["hyperparameter_search_allowed"] is False


def _verify_boundary() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_TRAIN_RESULT_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            "Frozen TRAIN representation result is not an ancestor."
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *IMPLEMENTATION_PATHS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip():
        raise RuntimeError(
            "TRAIN fit implementation must be clean and committed."
        )

    for path in (
        RESULT_PATH,
        SKLEARN_ARTIFACT,
        TCN_ARTIFACT,
    ):
        if path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing artifact: {path}"
            )


def load_train_representation(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        parse_dates=["timestamp"],
        float_precision="round_trip",
    )


def main() -> None:
    _verify_boundary()

    config = _yaml(CONFIG_PATH)
    validate_frozen_config(config)

    feature_contract = _json(FEATURE_PATH)
    ingestion = _json(INGESTION_PATH)

    features = [
        str(value)
        for value in feature_contract["final_raw_features"]
    ]

    if len(features) != 219:
        raise RuntimeError("Expected 219 frozen features.")

    if (
        feature_contract["final_raw_feature_sha256"]
        != EXPECTED_FEATURE_SHA
    ):
        raise RuntimeError("Frozen feature SHA changed.")

    if (
        ingestion["output"]["semantic_sha256"]
        != EXPECTED_TRAIN_SEMANTIC_SHA
    ):
        raise RuntimeError("Frozen TRAIN semantic SHA changed.")

    if ingestion["firewall"]["calibration_sensor_values_used"]:
        raise RuntimeError("CALIBRATION firewall already violated.")

    if ingestion["firewall"]["evaluation_sensor_values_used"]:
        raise RuntimeError("EVALUATION firewall already violated.")

    train = load_train_representation(
        TRAIN_PATH
    )

    expected_columns = [
        "day",
        "timestamp",
        "segment_id",
        *features,
    ]

    if list(train.columns) != expected_columns:
        raise RuntimeError("TRAIN representation columns changed.")

    if len(train) != 679:
        raise RuntimeError(
            f"Expected 679 TRAIN rows, got {len(train)}."
        )

    values = train[features].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError("TRAIN representation is not finite.")

    observed_semantic_sha = semantic_representation_sha256(
        train,
        feature_names=features,
    )

    if observed_semantic_sha != EXPECTED_TRAIN_SEMANTIC_SHA:
        raise RuntimeError(
            "Local TRAIN representation differs from frozen result."
        )

    # ---------------------------------------------------------
    # PCA — TRAIN only
    # ---------------------------------------------------------
    pca_scaler = fit_scaler(
        train,
        features,
        method="robust",
    )

    pca_scaled = transform_frame(
        train,
        features,
        pca_scaler,
    )

    pca = PCA(
        n_components=float(config["pca"]["variance_retained"]),
        svd_solver="full",
    )

    latent = pca.fit_transform(pca_scaled)
    reconstructed = pca.inverse_transform(latent)

    reconstruction_error = np.mean(
        (pca_scaled - reconstructed) ** 2,
        axis=1,
    )

    if not np.isfinite(reconstruction_error).all():
        raise RuntimeError(
            "PCA TRAIN reconstruction error is non-finite."
        )

    # ---------------------------------------------------------
    # TCN — TRAIN only
    # ---------------------------------------------------------
    temporal = config["temporal_model"]
    sequence = temporal["sequence"]
    architecture = temporal["architecture"]
    training = temporal["training"]

    indexed = (
        train.sort_values("timestamp")
        .set_index("timestamp")
    )

    tcn_scaler = fit_scaler(
        indexed,
        features,
        method="standard",
    )

    batch = prepare_temporal_batch(
        indexed,
        features=features,
        scaler=tcn_scaler,
        sequence_length=int(sequence["history_bins"]),
        bin_minutes=int(sequence["bin_minutes"]),
    )

    expected_sequences = expected_sequence_count(
        train["segment_id"],
        history_bins=int(sequence["history_bins"]),
    )

    if len(batch.inputs) != expected_sequences:
        raise RuntimeError(
            "Temporal sequence count does not match frozen segments: "
            f"{len(batch.inputs)} != {expected_sequences}"
        )

    channels = [
        int(value)
        for value in architecture["channels"]
    ]

    model_config = TemporalForecastConfig(
        input_dim=len(features),
        hidden_dim=channels[0],
        kernel_size=int(architecture["kernel_size"]),
        dilations=tuple(
            int(value)
            for value in architecture["dilations"]
        ),
        dropout=float(architecture["dropout"]),
    )

    training_config = EnergyAwareTrainingConfig(
        seed=int(training["seed"]),
        max_epochs=int(training["max_epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        validation_fraction=float(
            training["validation_fraction_of_train_sequences"]
        ),
        patience=int(training["early_stopping_patience"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    fit_result = train_energy_aware_tcn(
        batch,
        model_config=model_config,
        training_config=training_config,
    )

    SKLEARN_ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SKLEARN_ARTIFACT.open("wb") as handle:
        pickle.dump(
            {
                "features": features,
                "pca_scaler": pca_scaler,
                "pca": pca,
                "tcn_scaler": tcn_scaler,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    torch.save(
        {
            "features": features,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in fit_result.model.state_dict().items()
            },
            "model_config": {
                "input_dim": model_config.input_dim,
                "hidden_dim": model_config.hidden_dim,
                "kernel_size": model_config.kernel_size,
                "dilations": list(model_config.dilations),
                "dropout": model_config.dropout,
            },
            "history_bins": int(sequence["history_bins"]),
            "bin_minutes": int(sequence["bin_minutes"]),
        },
        TCN_ARTIFACT,
    )

    result = {
        "schema_version": "aeroxai.cobra_router_v2_train_fit.v1",
        "study": "cobra_router_v2_adaptation",
        "status": "TRAIN_MODELS_FITTED",
        "implementation_commit": _git_head(),
        "train_representation_commit": REQUIRED_TRAIN_RESULT_COMMIT,
        "data_use": {
            "train_rows": len(train),
            "train_segments": int(train["segment_id"].nunique()),
            "calibration_sensor_values_used": False,
            "evaluation_sensor_values_used": False,
        },
        "features": {
            "count": len(features),
            "sha256": EXPECTED_FEATURE_SHA,
        },
        "pca": {
            "scaler": "robust",
            "variance_retained_target": 0.95,
            "components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "train_reconstruction_mse_mean": float(
                reconstruction_error.mean()
            ),
            "train_reconstruction_mse_median": float(
                np.median(reconstruction_error)
            ),
            "threshold_selected": False,
        },
        "tcn": {
            "scaler": "standard",
            "eligible_sequences": len(batch.inputs),
            "fit_samples": fit_result.fit_samples,
            "validation_samples": fit_result.validation_samples,
            "device": fit_result.device,
            "best_epoch": fit_result.best_epoch,
            "epochs_ran": fit_result.epochs_ran,
            "best_validation_mse": float(
                min(fit_result.validation_losses)
            ),
            "final_train_mse": float(
                fit_result.train_losses[-1]
            ),
            "threshold_selected": False,
        },
        "artifacts": {
            "sklearn_path": str(
                SKLEARN_ARTIFACT.relative_to(ROOT)
            ),
            "sklearn_sha256": _sha256(SKLEARN_ARTIFACT),
            "tcn_path": str(
                TCN_ARTIFACT.relative_to(ROOT)
            ),
            "tcn_sha256": _sha256(TCN_ARTIFACT),
        },
        "firewall": {
            "calibration_opened": False,
            "evaluation_opened": False,
            "router_scored": False,
            "teacher_thresholds_selected": False,
            "energy_metrics_computed": False,
        },
    }

    RESULT_PATH.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": result["status"],
        "pca_components": result["pca"]["components"],
        "pca_explained_variance": (
            result["pca"]["explained_variance_ratio_sum"]
        ),
        "tcn_sequences": result["tcn"]["eligible_sequences"],
        "tcn_fit_samples": result["tcn"]["fit_samples"],
        "tcn_validation_samples": (
            result["tcn"]["validation_samples"]
        ),
        "tcn_device": result["tcn"]["device"],
        "tcn_best_epoch": result["tcn"]["best_epoch"],
        "tcn_best_validation_mse": (
            result["tcn"]["best_validation_mse"]
        ),
        "calibration_opened": False,
        "evaluation_opened": False,
        "result": str(RESULT_PATH.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
