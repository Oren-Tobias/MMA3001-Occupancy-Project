"""Automated tests for MMA3001 occupancy forecasting."""

import numpy as np
import pandas as pd
import pytest

from src import data_audit
from src import decision_tree_experiments as dt


def test_missing_dataset(monkeypatch, tmp_path):
    """Missing dataset must raise FileNotFoundError."""

    monkeypatch.setattr(
        data_audit,
        "DATA_PATH",
        tmp_path / "missing.csv"
    )

    with pytest.raises(FileNotFoundError):
        data_audit.load_data()


def test_valid_data_loading(monkeypatch, tmp_path):
    """Valid CSV should load and convert data types."""

    csv_path = tmp_path / "sample.csv"

    sample = pd.DataFrame({
        "id": [1, 2],
        "deviceid": [10, 10],
        "floorspaceid": [100, 100],
        "occupancystatus": [
            "CurrentlyOccupied",
            "NotOccupied"
        ],
        "headcount": ["3", "0"],
        "collecteddate": [
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:05:00Z"
        ],
        "occupancystatuschangedate": [
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:05:00Z"
        ],
        "previousoccupancystatus": [
            "NotOccupied",
            "CurrentlyOccupied"
        ]
    })

    sample.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        data_audit,
        "DATA_PATH",
        csv_path
    )

    loaded = data_audit.load_data()

    assert len(loaded) == 2
    assert loaded["headcount"].tolist() == [3, 0]


def test_missing_columns(monkeypatch, tmp_path):
    """Incomplete CSV must be rejected."""

    csv_path = tmp_path / "incomplete.csv"

    pd.DataFrame({
        "id": [1],
        "headcount": [3]
    }).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        data_audit,
        "DATA_PATH",
        csv_path
    )

    with pytest.raises(ValueError):
        data_audit.load_data()


def test_time_series_reconstruction():
    """Sensor readings older than 10 minutes remain missing."""

    observations = pd.DataFrame({
        "collecteddate": pd.to_datetime([
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:30:00Z"
        ]),
        "headcount": [3, 8]
    })

    result = dt.create_time_series(observations)

    values = result.set_index("timestamp")["occupancy"]

    assert values.loc[
        pd.Timestamp("2024-01-01T10:05:00Z")
    ] == 3

    assert pd.isna(
        values.loc[
            pd.Timestamp("2024-01-01T10:15:00Z")
        ]
    )

    assert values.loc[
        pd.Timestamp("2024-01-01T10:30:00Z")
    ] == 8


def test_feature_generation():
    """Lagged features and future targets must align."""

    timestamps = pd.date_range(
        "2024-01-01",
        periods=20,
        freq="5min",
        tz="UTC"
    )

    series = pd.DataFrame({
        "timestamp": timestamps,
        "occupancy": np.arange(20)
    })

    data, features = dt.create_features(
        series,
        horizon_minutes=10
    )

    first = data.iloc[0]

    assert first["current_occupancy"] == 6
    assert first["occupancy_5min_ago"] == 5
    assert first["occupancy_15min_ago"] == 3
    assert first["occupancy_30min_ago"] == 0
    assert first["target"] == 8
    assert len(features) == 6


def test_invalid_horizon():
    """Horizon must be divisible by five minutes."""

    series = pd.DataFrame({
        "timestamp": pd.date_range(
            "2024-01-01",
            periods=20,
            freq="5min",
            tz="UTC"
        ),
        "occupancy": np.arange(20)
    })

    with pytest.raises(ValueError):
        dt.create_features(series, 7)


def test_chronological_split():
    """Check 70/15/15 chronological boundaries."""

    series = pd.DataFrame({
        "timestamp": pd.date_range(
            "2024-01-01",
            periods=100,
            freq="5min",
            tz="UTC"
        )
    })

    train_end, validation_end = (
        dt.get_split_boundaries(series)
    )

    assert train_end == series["timestamp"].iloc[70]
    assert validation_end == series["timestamp"].iloc[85]
    assert train_end < validation_end
