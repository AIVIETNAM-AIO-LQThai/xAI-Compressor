import pandas as pd

from ml.data.schema import normalize_raw_frame


def test_normalize_raw_frame():
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
            ],
            "TP2": [1.0, 1.1],
            "TP3": [8.0, 8.1],
            "H1": [8.0, 8.1],
            "DV_pressure": [0.0, 0.1],
            "Reservoirs": [8.0, 8.1],
            "Oil_temperature": [40.0, 41.0],
            "Motor_current": [7.0, 7.1],
            "COMP": [0, 1],
            "DV_eletric": [1, 0],
            "Towers": [0, 1],
            "MPG": [0, 1],
            "LPS": [0, 0],
            "Pressure_switch": [0, 1],
            "Oil_level": [0, 0],
            "Caudal_impulses": [0, 1],
        }
    )

    result = normalize_raw_frame(raw)

    assert len(result) == 2

    assert "dv_electric" in result.columns
    assert "dv_eletric" not in result.columns
    assert "Unnamed: 0" not in result.columns

    assert result["timestamp"].is_monotonic_increasing