
"""
MMA3001 - DECISION TREE SENSITIVITY ANALYSIS

Engineering application:
    Short-term room occupancy forecasting.

Methods:
    - Decision tree regression (MMA3001 Week 5)
    - Persistence forecasting baseline
    - Chronological training and validation
    - Mean absolute error (MAE)
    - Computational runtime analysis

Experiments:
    1. Decision tree depth: 1 to 20
    2. Minimum samples per leaf: 13 settings
    3. Forecast horizon: 5 to 120 minutes
    4. Training runtime versus tree depth

IMPORTANT:
    This is a preliminary experiment using 250,000 raw rows.

    Occupancy values are reconstructed using the most recent
    measurement, subject to a maximum age of 10 minutes.

    This reconstruction assumption must be verified against
    the sensor data before results are used in the final report.

    The final project must also use an independent test set.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "occupancy.csv"

FIGURE_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Number of raw CSV rows used for this preliminary experiment
SAMPLE_ROWS = 250_000

# Regular time-series interval
INTERVAL_MINUTES = 5

# Maximum permitted age of a sensor measurement
MAX_SENSOR_AGE_MINUTES = 10

# Expanded sensitivity-analysis values
DEPTH_VALUES = list(range(1, 21))

LEAF_VALUES = [
    1, 2, 5, 10, 15, 20, 30,
    40, 50, 75, 100, 150, 200
]

HORIZON_VALUES = list(range(5, 121, 5))

# Fixed settings for one-variable-at-a-time comparisons
FIXED_DEPTH = 6
FIXED_LEAF = 20
FIXED_HORIZON = 30

# Chronological split proportions
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# 15% reserved for future final testing


# ============================================================
# 3. LOAD OCCUPANCY DATA
# ============================================================

def load_occupancy_data():
    """
    Load the first SAMPLE_ROWS records and select the
    floor space with the greatest number of observations.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted occupancy measurements.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find dataset: {DATA_PATH}"
        )

    print("\nLoading occupancy dataset...")

    df = pd.read_csv(
        DATA_PATH,
        nrows=SAMPLE_ROWS,
        usecols=[
            "floorspaceid",
            "collecteddate",
            "headcount"
        ]
    )

    df["collecteddate"] = pd.to_datetime(
        df["collecteddate"],
        utc=True,
        errors="coerce"
    )

    df["headcount"] = pd.to_numeric(
        df["headcount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "floorspaceid",
            "collecteddate",
            "headcount"
        ]
    )

    df = df[df["headcount"] >= 0].copy()

    if df.empty:
        raise ValueError("No valid occupancy records found.")

    selected_space = (
        df["floorspaceid"]
        .value_counts()
        .index[0]
    )

    df = df[
        df["floorspaceid"] == selected_space
    ].copy()

    df = df.sort_values(
        "collecteddate"
    ).reset_index(drop=True)

    print(f"Selected floor space: {selected_space}")
    print(f"Number of observations: {len(df):,}")

    return df


# ============================================================
# 4. CREATE REGULAR TIME SERIES
# ============================================================

def create_time_series(df):
    """
    Create a regularly spaced occupancy time series.

    Uses the most recent observed headcount, but only
    if the observation is sufficiently recent.

    Long gaps remain missing rather than being assigned
    an artificial occupancy of zero.

    NOTE:
        This is a provisional reconstruction method.
        It does not prove that sensor states persist
        between observations.
    """

    print("\nCreating regular time series...")

    observations = (
        df[["collecteddate", "headcount"]]
        .drop_duplicates(
            subset=["collecteddate"],
            keep="last"
        )
        .sort_values("collecteddate")
        .reset_index(drop=True)
    )

    start = (
        observations["collecteddate"]
        .min()
        .ceil(f"{INTERVAL_MINUTES}min")
    )

    end = (
        observations["collecteddate"]
        .max()
        .floor(f"{INTERVAL_MINUTES}min")
    )

    grid = pd.DataFrame({
        "timestamp": pd.date_range(
            start=start,
            end=end,
            freq=f"{INTERVAL_MINUTES}min"
        )
    })

    series = pd.merge_asof(
        grid,
        observations,
        left_on="timestamp",
        right_on="collecteddate",
        direction="backward",
        tolerance=pd.Timedelta(
            minutes=MAX_SENSOR_AGE_MINUTES
        )
    )

    series = series.rename(
        columns={"headcount": "occupancy"}
    )

    total_points = len(series)

    valid_points = (
        series["occupancy"]
        .notna()
        .sum()
    )

    coverage = (
        100 * valid_points / total_points
        if total_points > 0
        else 0
    )

    print(f"Total time points: {total_points:,}")
    print(f"Valid time points: {valid_points:,}")
    print(f"Coverage: {coverage:.2f}%")

    return series


# ============================================================
# 5. CREATE FORECASTING FEATURES
# ============================================================

def create_features(series, horizon_minutes):
    """
    Create historical occupancy features and a future target.

    Inputs:
        Occupancy at time t
        Occupancy 5 minutes ago
        Occupancy 15 minutes ago
        Occupancy 30 minutes ago
        Hour of day
        Day of week

    Output:
        Headcount at time t + forecast horizon

    All features use information available at or before
    the prediction time.
    """

    if horizon_minutes % INTERVAL_MINUTES != 0:
        raise ValueError(
            "Forecast horizon must be divisible by "
            "the sampling interval."
        )

    data = series.copy()

    steps = (
        horizon_minutes // INTERVAL_MINUTES
    )

    data["current_occupancy"] = data["occupancy"]

    data["occupancy_5min_ago"] = (
        data["occupancy"].shift(1)
    )

    data["occupancy_15min_ago"] = (
        data["occupancy"].shift(3)
    )

    data["occupancy_30min_ago"] = (
        data["occupancy"].shift(6)
    )

    local_time = (
        data["timestamp"]
        .dt.tz_convert("Australia/Melbourne")
    )

    data["hour"] = local_time.dt.hour

    data["day_of_week"] = local_time.dt.dayofweek

    # Future headcount is the prediction target
    data["target"] = (
        data["occupancy"].shift(-steps)
    )

    # Store the target timestamp for leakage-safe splitting
    data["target_timestamp"] = (
        data["timestamp"].shift(-steps)
    )

    features = [
        "current_occupancy",
        "occupancy_5min_ago",
        "occupancy_15min_ago",
        "occupancy_30min_ago",
        "hour",
        "day_of_week"
    ]

    data = data.dropna(
        subset=features + [
            "target",
            "target_timestamp"
        ]
    ).copy()

    return data, features


# ============================================================
# 6. DEFINE CHRONOLOGICAL SPLIT BOUNDARIES
# ============================================================

def get_split_boundaries(series):
    """
    Define common time boundaries for all experiments.

    70% training
    15% validation
    15% held-out test period

    The test period is not used for model selection.
    """

    timestamps = series["timestamp"]

    train_index = int(
        len(timestamps) * TRAIN_FRACTION
    )

    validation_index = int(
        len(timestamps) * (
            TRAIN_FRACTION + VALIDATION_FRACTION
        )
    )

    train_boundary = timestamps.iloc[train_index]

    validation_boundary = timestamps.iloc[
        validation_index
    ]

    print("\nChronological split boundaries:")
    print(f"Training ends: {train_boundary}")
    print(f"Validation ends: {validation_boundary}")

    return train_boundary, validation_boundary


# ============================================================
# 7. EVALUATE DECISION TREE
# ============================================================

def evaluate_model(
    data,
    features,
    depth,
    leaf,
    train_boundary,
    validation_boundary
):
    """
    Train a decision tree and evaluate on unseen future data.

    Training samples:
        Target time before training boundary.

    Validation samples:
        Prediction time at or after training boundary,
        with target time before validation boundary.

    Returns
    -------
    mae : float
        Mean absolute error in people.

    runtime : float
        Model fitting time in seconds.

    baseline_mae : float
        Persistence forecast MAE in people.
    """

    train = data[
        data["target_timestamp"] < train_boundary
    ].copy()

    validation = data[
        (data["timestamp"] >= train_boundary)
        & (
            data["target_timestamp"]
            < validation_boundary
        )
    ].copy()

    if len(train) < 50 or len(validation) < 20:
        raise ValueError(
            "Insufficient valid observations in the "
            "training or validation periods."
        )

    X_train = train[features]
    y_train = train["target"]

    X_validation = validation[features]
    y_validation = validation["target"]

    model = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=42
    )

    start_time = perf_counter()

    model.fit(X_train, y_train)

    training_time = (
        perf_counter() - start_time
    )

    predictions = model.predict(
        X_validation
    )

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    # Persistence baseline:
    # Future occupancy = current occupancy
    baseline_predictions = (
        validation["current_occupancy"]
    )

    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions
    )

    return mae, training_time, baseline_mae


# ============================================================
# 8. RUN EXPANDED SENSITIVITY EXPERIMENTS
# ============================================================

def run_experiments(series):
    """
    Run three sensitivity experiments:

    A. Tree depth: 20 values
    B. Minimum leaf size: 13 values
    C. Forecast horizon: 24 values

    Fixed parameters are used so that each graph
    isolates the effect of one independent variable.
    """

    results = []

    train_boundary, validation_boundary = (
        get_split_boundaries(series)
    )

    # --------------------------------------------------------
    # EXPERIMENT A: TREE DEPTH
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT A: DECISION TREE DEPTH")
    print("=" * 60)

    data_30, features = create_features(
        series,
        FIXED_HORIZON
    )

    for depth in DEPTH_VALUES:

        mae, runtime, baseline_mae = evaluate_model(
            data=data_30,
            features=features,
            depth=depth,
            leaf=FIXED_LEAF,
            train_boundary=train_boundary,
            validation_boundary=validation_boundary
        )

        results.append({
            "experiment": "tree_depth",
            "horizon_minutes": FIXED_HORIZON,
            "max_depth": depth,
            "min_samples_leaf": FIXED_LEAF,
            "mae_people": mae,
            "baseline_mae_people": baseline_mae,
            "training_time_seconds": runtime
        })

        print(
            f"Depth {depth:2d} | "
            f"MAE: {mae:.4f} | "
            f"Baseline: {baseline_mae:.4f} | "
            f"Time: {runtime:.5f}s"
        )

    # --------------------------------------------------------
    # EXPERIMENT B: MINIMUM SAMPLES PER LEAF
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT B: MINIMUM LEAF SIZE")
    print("=" * 60)

    for leaf in LEAF_VALUES:

        mae, runtime, baseline_mae = evaluate_model(
            data=data_30,
            features=features,
            depth=FIXED_DEPTH,
            leaf=leaf,
            train_boundary=train_boundary,
            validation_boundary=validation_boundary
        )

        results.append({
            "experiment": "leaf_size",
            "horizon_minutes": FIXED_HORIZON,
            "max_depth": FIXED_DEPTH,
            "min_samples_leaf": leaf,
            "mae_people": mae,
            "baseline_mae_people": baseline_mae,
            "training_time_seconds": runtime
        })

        print(
            f"Leaf {leaf:3d} | "
            f"MAE: {mae:.4f} | "
            f"Baseline: {baseline_mae:.4f} | "
            f"Time: {runtime:.5f}s"
        )

    # --------------------------------------------------------
    # EXPERIMENT C: FORECAST HORIZON
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT C: FORECAST HORIZON")
    print("=" * 60)

    for horizon in HORIZON_VALUES:

        data, horizon_features = create_features(
            series,
            horizon
        )

        mae, runtime, baseline_mae = evaluate_model(
            data=data,
            features=horizon_features,
            depth=FIXED_DEPTH,
            leaf=FIXED_LEAF,
            train_boundary=train_boundary,
            validation_boundary=validation_boundary
        )

        results.append({
            "experiment": "forecast_horizon",
            "horizon_minutes": horizon,
            "max_depth": FIXED_DEPTH,
            "min_samples_leaf": FIXED_LEAF,
            "mae_people": mae,
            "baseline_mae_people": baseline_mae,
            "training_time_seconds": runtime
        })

        print(
            f"Horizon {horizon:3d} min | "
            f"MAE: {mae:.4f} | "
            f"Baseline: {baseline_mae:.4f} | "
            f"Time: {runtime:.5f}s"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        TABLE_DIR / "decision_tree_experiments.csv",
        index=False
    )

    return results_df


# ============================================================
# 9. GENERATE GRAPHS
# ============================================================

def save_graph(
    x,
    y,
    xlabel,
    ylabel,
    title,
    filename,
    baseline=None
):
    """
    Generate and save a high-resolution line graph.
    """

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        x,
        y,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Decision tree"
    )

    if baseline is not None:
        plt.plot(
            x,
            baseline,
            linestyle="--",
            linewidth=1.5,
            label="Persistence baseline"
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(alpha=0.3)

    if baseline is not None:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300
    )

    plt.close()


def generate_graphs(results):
    """
    Generate four graphs for the MMA3001 report.
    """

    # --------------------------------------------------------
    # GRAPH 1: TREE DEPTH VS ERROR
    # --------------------------------------------------------

    depth_data = results[
        results["experiment"] == "tree_depth"
    ].sort_values("max_depth")

    save_graph(
        x=depth_data["max_depth"],
        y=depth_data["mae_people"],
        xlabel="Maximum decision tree depth",
        ylabel="Mean absolute error (people)",
        title="Decision Tree Depth vs Forecast Error",
        filename="tree_depth_vs_error.png",
        baseline=depth_data["baseline_mae_people"]
    )

    # --------------------------------------------------------
    # GRAPH 2: LEAF SIZE VS ERROR
    # --------------------------------------------------------

    leaf_data = results[
        results["experiment"] == "leaf_size"
    ].sort_values("min_samples_leaf")

    save_graph(
        x=leaf_data["min_samples_leaf"],
        y=leaf_data["mae_people"],
        xlabel="Minimum samples per leaf",
        ylabel="Mean absolute error (people)",
        title="Minimum Leaf Size vs Forecast Error",
        filename="leaf_size_vs_error.png",
        baseline=leaf_data["baseline_mae_people"]
    )

    # --------------------------------------------------------
    # GRAPH 3: FORECAST HORIZON VS ERROR
    # --------------------------------------------------------

    horizon_data = results[
        results["experiment"] == "forecast_horizon"
    ].sort_values("horizon_minutes")

    save_graph(
        x=horizon_data["horizon_minutes"],
        y=horizon_data["mae_people"],
        xlabel="Forecast horizon (minutes)",
        ylabel="Mean absolute error (people)",
        title="Forecast Horizon vs Prediction Error",
        filename="horizon_vs_error.png",
        baseline=horizon_data["baseline_mae_people"]
    )

    # --------------------------------------------------------
    # GRAPH 4: TREE DEPTH VS TRAINING RUNTIME
    # --------------------------------------------------------

    save_graph(
        x=depth_data["max_depth"],
        y=depth_data["training_time_seconds"],
        xlabel="Maximum decision tree depth",
        ylabel="Training time (seconds)",
        title="Decision Tree Depth vs Training Runtime",
        filename="tree_depth_vs_runtime.png"
    )

    print("\nAll four graphs saved successfully.")


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():
    """
    Run the complete preliminary decision-tree analysis.
    """

    print("=" * 65)
    print("MMA3001 - DECISION TREE SENSITIVITY ANALYSIS")
    print("=" * 65)

    df = load_occupancy_data()

    series = create_time_series(df)

    results = run_experiments(series)

    generate_graphs(results)

    print("\n" + "=" * 65)
    print("EXPERIMENTS COMPLETED")
    print("=" * 65)

    print(f"\nTotal experiments: {len(results)}")

    print("\nResults table:")
    print(
        results.round(4).to_string(index=False)
    )

    print("\nGraphs saved to:")
    print(FIGURE_DIR)

    print("\nCSV results saved to:")
    print(
        TABLE_DIR / "decision_tree_experiments.csv"
    )

    print(
        "\nNOTE: These are preliminary validation results, "
        "not final test-set results."
    )


if __name__ == "__main__":
    main()
