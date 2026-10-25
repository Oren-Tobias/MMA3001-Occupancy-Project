
"""
MMA3001 - NUMERICAL METHODS AND MACHINE LEARNING
MONASH SMART INFRASTRUCTURE OCCUPANCY PROJECT

STAGE 1: INITIAL DATA AUDIT

Purpose
-------
Investigate the structure, quality and temporal characteristics
of occupancy sensor data before developing forecasting models.

Dataset
-------
Approximately 9 million occupancy sensor records.
This initial audit examines only the first 250,000 rows.

Outputs
-------
results/tables/initial_data_audit.txt
results/tables/floorspace_summary.csv
results/tables/missing_values.csv

Limitations
-----------
The first 250,000 records are not necessarily representative
of the complete dataset. Results should be interpreted as
a preliminary investigation.
"""

from pathlib import Path
from contextlib import redirect_stdout
import time

import pandas as pd



# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Resolve paths relative to the project directory so the
# script can run on different computers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "occupancy.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

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
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_data():
    """
    Load and validate the initial occupancy dataset sample.

    Reads the first 250,000 records from the occupancy CSV
    and checks that all required columns are present.

    Collection timestamps and occupancy-status change
    timestamps are converted to UTC datetime values.
    Headcounts are converted to numeric values.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing sampled occupancy records.

    Raises
    ------
    FileNotFoundError
        If the occupancy CSV does not exist.
    ValueError
        If required columns are missing.
    """

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

    # Confirm that the dataset has the expected structure.
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    # Convert timestamps to a consistent UTC representation.
    # Invalid timestamps become NaT and are counted in the
    # missing-value analysis.
    for column in [
        "collecteddate",
        "occupancystatuschangedate"
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True
        )

    # Invalid or non-numeric headcounts become NaN.
    df["headcount"] = pd.to_numeric(
        df["headcount"],
        errors="coerce"
    )

    elapsed = time.perf_counter() - start

    print(f"Loaded {len(df):,} rows.")
    print(f"Loading time: {elapsed:.2f} seconds")

    return df


# ============================================================
# 3. DATA QUALITY AUDIT
# ============================================================

def audit_data(df):
    """
    Examine the structure and quality of occupancy records.

    Reports dataset dimensions, date coverage, missing
    values, duplicate records, occupancy status frequencies
    and headcount statistics.

    Also checks for inconsistencies between occupancy
    status and recorded headcount.

    Parameters
    ----------
    df : pandas.DataFrame
        Sampled occupancy records with validated columns.

    Returns
    -------
    None
        Results are printed and missing-value statistics
        are exported as a CSV file.
    """

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

    # Calculate the number and percentage of missing
    # values for every dataset column.
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": df.isna().mean() * 100
    })

    print(missing.round(3))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    missing.to_csv(
        RESULTS_DIR / "missing_values.csv"
    )

    print("\n" + "=" * 65)
    print("4. DUPLICATE RECORDS")
    print("=" * 65)

    # Identify records where all column values match.
    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates:,}")

    if len(df) > 0:
        duplicate_percentage = 100 * duplicates / len(df)
    else:
        duplicate_percentage = 0.0

    print(
        f"Duplicate percentage: "
        f"{duplicate_percentage:.3f}%"
    )

    # Repeated timestamps may indicate multiple sensor
    # observations or duplicate measurements.
    # They are not automatically considered invalid.
    repeated_timestamps = df.duplicated(
        subset=[
            "deviceid",
            "floorspaceid",
            "collecteddate"
        ]
    ).sum()

    print(
        "Repeated device/floor-space timestamps: "
        f"{repeated_timestamps:,}"
    )

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

    # Identify physically questionable headcounts.
    negative_headcounts = (df["headcount"] < 0).sum()

    print(
        f"\nNegative headcounts: "
        f"{negative_headcounts:,}"
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

    # Compare categorical occupancy status against
    # the numerical headcount measurement.
    occupied_zero = (
        (df["occupancystatus"] == "CurrentlyOccupied")
        & (df["headcount"] == 0)
    ).sum()

    unoccupied_positive = (
        (df["occupancystatus"] == "NotOccupied")
        & (df["headcount"] > 0)
    ).sum()

    print(
        "\nCurrentlyOccupied with zero headcount: "
        f"{occupied_zero:,}"
    )

    print(
        "NotOccupied with positive headcount: "
        f"{unoccupied_positive:,}"
    )


# ============================================================
# 4. SENSOR TIMING ANALYSIS
# ============================================================

def analyse_timing(df):
    """
    Investigate the timing of consecutive sensor records.

    Records are sorted by device, floor space and timestamp.
    Time differences are calculated within each device and
    floor-space group.

    This helps determine whether the sensor data are
    regularly sampled or contain irregular intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Occupancy records containing device identifiers,
        floor-space identifiers and collection timestamps.

    Returns
    -------
    None
        Timing statistics are printed to the audit report.
    """

    print("\n" + "=" * 65)
    print("8. SENSOR TIMING ANALYSIS")
    print("=" * 65)

    timing_df = df.sort_values(
        ["deviceid", "floorspaceid", "collecteddate"]
    ).copy()

    # Calculate time differences separately for each
    # device and floor-space combination.
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
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99
            ]
        )
    )

    print("\nMost common intervals:")
    print(intervals.value_counts().head(20))

    # Large gaps may indicate missing observations,
    # sensor inactivity or event-driven recording.
    long_gaps = (intervals > 3600).sum()

    print("\nIntervals longer than one hour:")
    print(f"{long_gaps:,}")

    if intervals.notna().any():
        print(
            "Longest recorded interval (seconds): "
            f"{intervals.max():,.0f}"
        )


# ============================================================
# 5. FLOOR-SPACE ANALYSIS
# ============================================================

def analyse_floor_spaces(df):
    """
    Summarise sensor coverage and occupancy by floor space.

    Calculates the number of observations, unique devices,
    first and last timestamps, mean headcount and maximum
    headcount for each floor space.

    Parameters
    ----------
    df : pandas.DataFrame
        Occupancy records containing floor-space IDs,
        device IDs, timestamps and headcounts.

    Returns
    -------
    None
        A summary is printed and exported to
        results/tables/floorspace_summary.csv.
    """

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        RESULTS_DIR / "floorspace_summary.csv"
    )


# ============================================================
# 6. OCCUPANCY TRANSITION ANALYSIS
# ============================================================

def analyse_transitions(df):
    """
    Investigate occupancy status transitions.

    Compares the previous occupancy status with the current
    occupancy status and measures the elapsed time between
    a status change and the collection timestamp.

    Parameters
    ----------
    df : pandas.DataFrame
        Occupancy records containing current and previous
        status fields and associated timestamps.

    Returns
    -------
    None
        Transition statistics are printed to the report.
    """

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

    print(
        "\nThis analysis helps investigate whether "
        "the dataset contains regularly sampled "
        "measurements or event-based records."
    )


# ============================================================
# 7. EXECUTE INITIAL DATA AUDIT
# ============================================================

def main():
    """
    Execute the initial occupancy data audit.

    Loads a sample of the occupancy dataset and performs
    data quality, sensor timing, floor-space coverage and
    occupancy transition analyses.

    Returns
    -------
    None
        Audit results are written to the configured output
        files when the script is executed directly.
    """

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


# ============================================================
# 8. RUN SCRIPT AND SAVE REPORT
# ============================================================

if __name__ == "__main__":

    # Create the results directory only when required.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "initial_data_audit.txt"

    # Save printed analysis results to a text file.
    with open(output_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            main()

    print("Data audit completed successfully!")
    print(f"Results saved to: {output_file}")


