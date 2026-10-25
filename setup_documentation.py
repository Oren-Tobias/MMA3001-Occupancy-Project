
"""
MMA3001 - GitHub Documentation and Testing Setup

This script:
1. Installs documentation and testing packages.
2. Creates automated tests for existing project functions.
3. Runs pytest.
4. Generates an HTML test report.
5. Generates HTML source-code documentation.

Run from the project root:
    python setup_documentation.py
"""

from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "src"
TESTS = ROOT / "tests"
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"

TESTS.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Make the source folder importable.
(SRC / "__init__.py").touch(exist_ok=True)

# ------------------------------------------------------------
# 1. CREATE REQUIREMENTS FILE
# ------------------------------------------------------------

requirements = """
numpy
pandas
scikit-learn
matplotlib
pytest
pytest-html
pdoc
"""

requirements_file = ROOT / "requirements.txt"

if not requirements_file.exists():
    requirements_file.write_text(
        textwrap.dedent(requirements).strip() + "\n",
        encoding="utf-8"
    )

# ------------------------------------------------------------
# 2. CREATE AUTOMATED TESTS
# ------------------------------------------------------------

test_code = '''
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
'''

test_file = TESTS / "test_project.py"

if not test_file.exists():
    test_file.write_text(
        textwrap.dedent(test_code).lstrip(),
        encoding="utf-8"
    )

print("Project test files prepared.")

# ------------------------------------------------------------
# 3. INSTALL DEPENDENCIES
# ------------------------------------------------------------

print("\nInstalling required packages...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file)
    ],
    cwd=ROOT,
    check=True
)

# ------------------------------------------------------------
# 4. GENERATE HTML DOCUMENTATION
# ------------------------------------------------------------

print("\nGenerating HTML documentation...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pdoc",
        "src",
        "-o",
        str(DOCS)
    ],
    cwd=ROOT,
    check=True
)

print(f"Documentation saved to: {DOCS}")

# ------------------------------------------------------------
# 5. RUN AUTOMATED TESTS AND GENERATE REPORT
# ------------------------------------------------------------

print("\nRunning automated tests...")

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-v",
        f"--html={RESULTS / 'test_report.html'}",
        "--self-contained-html"
    ],
    cwd=ROOT,
    check=False
)

# ------------------------------------------------------------
# 6. FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 60)

if result.returncode == 0:
    print("ALL AUTOMATED TESTS PASSED")
else:
    print("SOME TESTS FAILED - REVIEW TERMINAL OUTPUT")

print("=" * 60)

print("\nGenerated files:")
print(f"Tests: {test_file}")
print(f"HTML documentation: {DOCS}")
print(f"HTML test report: {RESULTS / 'test_report.html'}")

if result.returncode != 0:
    sys.exit(result.returncode)
