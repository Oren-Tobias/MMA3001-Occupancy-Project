
"""
MMA3001 - NUMERICAL METHODS AND MACHINE LEARNING
MONASH SMART INFRASTRUCTURE OCCUPANCY PROJECT

STAGE 1: INITIAL DATA AUDIT

Purpose:
    Investigate the structure, quality and temporal characteristics
    of the occupancy dataset before developing forecasting models.

Dataset:
    Approximately 9 million occupancy sensor records.

Outputs:
    results/tables/initial_data_audit.txt
    results/tables/floorspace_summary.csv
    results/tables/missing_values.csv
"""

from pathlib import Path
from contextlib import redirect_stdout
import time

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "occupancy.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS = 250_000

EXPECTED_COLUMNS = [
    "id",
    "deviceid",
    "floorspaceid",
    "occupancystatus",
    "headcount",
    "collecteddate",
    "occupancystatuschangedate",
    "previousoccupancystatus",
]


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

def load_data():
    """Load and validate a sample of the occupancy dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find occupancy dataset: {DATA_PATH}"
        )

    start = time.perf_counter()

    df = pd.read_csv(
        DATA_PATH,
        nrows=SAMPLE_ROWS,
        low_memory=False
    )

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    for column in [
        "collecteddate",
        "occupancystatuschangedate"
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True
        )

    df["headcount"] = pd.to_numeric(
        df["headcount"],
        errors="coerce"
    )

    print(f"Loaded {len(df):,} rows.")
    print(f"Loading time: {time.perf_counter() - start:.2f} seconds")

    return df


# ============================================================
# 3. DATA QUALITY AUDIT
# ============================================================

def audit_data(df):
    """Analyse dataset structure, missing values and headcounts."""

    print("\n" + "=" * 65)
    print("1. DATASET STRUCTURE")
    print("=" * 65)

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Unique devices: {df['deviceid'].nunique():,}")
    print(f"Unique floor spaces: {df['floorspaceid'].nunique():,}")

    print("\nData types:")
    print(df.dtypes)

    print("\n" + "=" * 65)
    print("2. DATE RANGE")
    print("=" * 65)

    first = df["collecteddate"].min()
    last = df["collecteddate"].max()

    print(f"First measurement: {first}")
    print(f"Last measurement: {last}")
    print(f"Observed duration: {last - first}")

    print("\n" + "=" * 65)
    print("3. MISSING VALUES")
    print("=" * 65)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": df.isna().mean() * 100
    })

    print(missing.round(3))

    missing.to_csv(
        RESULTS_DIR / "missing_values.csv"
    )

    print("\n" + "=" * 65)
    print("4. DUPLICATE RECORDS")
    print("=" * 65)

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates:,}")
    print(f"Duplicate percentage: {100 * duplicates / len(df):.3f}%")

    print("\n" + "=" * 65)
    print("5. OCCUPANCY STATUS")
    print("=" * 65)

    print(
        df["occupancystatus"]
        .value_counts(dropna=False)
    )

    print("\n" + "=" * 65)
    print("6. HEADCOUNT")
    print("=" * 65)

    print(df["headcount"].describe())

    print("\nHeadcount frequencies:")
    print(
        df["headcount"]
        .value_counts(dropna=False)
        .sort_index()
        .head(30)
    )

    print("\n" + "=" * 65)
    print("7. OCCUPANCY STATUS VS HEADCOUNT")
    print("=" * 65)

    print(
        pd.crosstab(
            df["occupancystatus"],
            df["headcount"]
        )
    )

    occupied_zero = (
        (df["occupancystatus"] == "CurrentlyOccupied")
        & (df["headcount"] == 0)
    ).sum()

    unoccupied_positive = (
        (df["occupancystatus"] == "NotOccupied")
        & (df["headcount"] > 0)
    ).sum()

    print(f"\nCurrentlyOccupied with zero headcount: {occupied_zero:,}")
    print(f"NotOccupied with positive headcount: {unoccupied_positive:,}")


# ============================================================
# 4. ANALYSE SENSOR TIMING
# ============================================================

def analyse_timing(df):
    """Investigate time between consecutive sensor records."""

    print("\n" + "=" * 65)
    print("8. SENSOR TIMING ANALYSIS")
    print("=" * 65)

    timing_df = df.sort_values(
        ["deviceid", "floorspaceid", "collecteddate"]
    ).copy()

    timing_df["interval_seconds"] = (
        timing_df
        .groupby(["deviceid", "floorspaceid"])["collecteddate"]
        .diff()
        .dt.total_seconds()
    )

    intervals = timing_df["interval_seconds"]

    print("\nTime between observations (seconds):")

    print(
        intervals.describe(
            percentiles=[
                0.01, 0.05, 0.25,
                0.50, 0.75, 0.95, 0.99
            ]
        )
    )

    print("\nMost common intervals:")
    print(intervals.value_counts().head(20))

    print("\nIntervals longer than one hour:")
    print((intervals > 3600).sum())


# ============================================================
# 5. ANALYSE FLOOR SPACES
# ============================================================

def analyse_floor_spaces(df):
    """Summarise occupancy records by floor space."""

    print("\n" + "=" * 65)
    print("9. FLOOR-SPACE COVERAGE")
    print("=" * 65)

    summary = (
        df.groupby("floorspaceid")
        .agg(
            observations=("id", "size"),
            unique_devices=("deviceid", "nunique"),
            first_measurement=("collecteddate", "min"),
            last_measurement=("collecteddate", "max"),
            mean_headcount=("headcount", "mean"),
            maximum_headcount=("headcount", "max")
        )
        .sort_values("observations", ascending=False)
    )

    print(summary.head(20).to_string())

    summary.to_csv(
        RESULTS_DIR / "floorspace_summary.csv"
    )


# ============================================================
# 6. ANALYSE OCCUPANCY TRANSITIONS
# ============================================================

def analyse_transitions(df):
    """Investigate changes between occupancy states."""

    print("\n" + "=" * 65)
    print("10. OCCUPANCY STATE TRANSITIONS")
    print("=" * 65)

    transitions = pd.crosstab(
        df["previousoccupancystatus"],
        df["occupancystatus"]
    )

    print(transitions)

    differences = (
        df["collecteddate"]
        - df["occupancystatuschangedate"]
    ).dt.total_seconds()

    print("\nCollection time minus status-change time:")
    print(differences.describe())

    print("\nThis analysis helps determine whether the dataset")
    print("contains regularly sampled data or event-based records.")


# ============================================================
# 7. EXECUTE AUDIT
# ============================================================

def main():
    """Run the initial occupancy dataset audit."""

    print("=" * 65)
    print("MMA3001 - INITIAL OCCUPANCY DATA AUDIT")
    print("=" * 65)

    print(f"Dataset: {DATA_PATH}")
    print(f"Sample size: {SAMPLE_ROWS:,}")

    df = load_data()

    audit_data(df)
    analyse_timing(df)
    analyse_floor_spaces(df)
    analyse_transitions(df)

    print("\n" + "=" * 65)
    print("DATA AUDIT COMPLETE")
    print("=" * 65)

    print(
        "\nThese results describe the initial sample, "
        "not necessarily the complete dataset."
    )


if __name__ == "__main__":

    output_file = RESULTS_DIR / "initial_data_audit.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            main()

    print("Data audit completed successfully!")
    print(f"Results saved to: {output_file}")
