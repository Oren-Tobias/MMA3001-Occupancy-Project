
"""
MMA3001 - DECISION TREE SENSITIVITY ANALYSIS

Methods:
    Decision tree regression (Week 5)
    Chronological training and validation
    Mean absolute error (MAE)
    Runtime measurement

Experiments:
    1. Maximum tree depth
    2. Minimum samples per leaf
    3. Forecast horizon
    4. Training time versus tree depth

NOTE:
    This is a preliminary experiment using 250,000 raw rows.
    Final results require full-dataset preprocessing and validation
    of the occupancy sensor's timestamp/state semantics.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "occupancy.csv"

OUTPUT_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS = 250_000

INTERVAL_MINUTES = 5
MAX_SENSOR_AGE_MINUTES = 10

DEPTH_VALUES = [2, 4, 6, 8, 12]
LEAF_VALUES = [5, 20, 50]
HORIZON_VALUES = [15, 30, 60]


# ============================================================
# LOAD DATA
# ============================================================

def load_occupancy_data():
    """Load initial occupancy sample and select its busiest space."""

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
        subset=["floorspaceid", "collecteddate", "headcount"]
    )

    df = df[df["headcount"] >= 0].copy()

    space_counts = df["floorspaceid"].value_counts()

    selected_space = space_counts.index[0]

    df = df[df["floorspaceid"] == selected_space].copy()

    df = df.sort_values("collecteddate")

    print("Selected floor space:", selected_space)
    print("Raw observations:", len(df))

    return df


# ============================================================
# CREATE REGULAR TIME SERIES
# ============================================================

def create_time_series(df):
    """
    Construct a 5-minute grid using the latest observed headcount.

    A grid value is retained only if its source observation
    is at most MAX_SENSOR_AGE_MINUTES old.

    This assumes collected headcount is a valid point measurement.
    It does not establish that values persist between updates.
    """

    observations = (
        df[["collecteddate", "headcount"]]
        .drop_duplicates(
            subset=["collecteddate"],
            keep="last"
        )
        .sort_values("collecteddate")
        .reset_index(drop=True)
    )

    start = observations["collecteddate"].min().ceil(
        f"{INTERVAL_MINUTES}min"
    )

    end = observations["collecteddate"].max().floor(
        f"{INTERVAL_MINUTES}min"
    )

    grid = pd.DataFrame({
        "timestamp": pd.date_range(
            start=start,
            end=end,
            freq=f"{INTERVAL_MINUTES}min",
            tz="UTC"
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

    print("Regular grid points:", len(series))
    print(
        "Valid grid points:",
        series["occupancy"].notna().sum()
    )

    return series


# ============================================================
# CREATE FORECASTING FEATURES
# ============================================================

def create_features(series, horizon_minutes):
    """
    Build features using only present/past observations.

    Predict occupancy at time t + forecast horizon.

    Missing or unreliable grid values remain missing.
    """

    data = series.copy()

    steps = horizon_minutes // INTERVAL_MINUTES

    data["current_occupancy"] = data["occupancy"]
    data["occupancy_5min_ago"] = data["occupancy"].shift(1)
    data["occupancy_15min_ago"] = data["occupancy"].shift(3)
    data["occupancy_30min_ago"] = data["occupancy"].shift(6)

    local_time = data["timestamp"].dt.tz_convert(
        "Australia/Melbourne"
    )

    data["hour"] = local_time.dt.hour
    data["day_of_week"] = local_time.dt.dayofweek

    data["target"] = data["occupancy"].shift(-steps)

    features = [
        "current_occupancy",
        "occupancy_5min_ago",
        "occupancy_15min_ago",
        "occupancy_30min_ago",
        "hour",
        "day_of_week"
    ]

    data = data.dropna(
        subset=features + ["target"]
    ).copy()

    return data, features


# ============================================================
# CHRONOLOGICAL TRAINING AND VALIDATION
# ============================================================

def evaluate_model(data, features, depth, leaf):
    """
    Train on earlier data and validate on later data.

    Returns:
        MAE in people
        Training runtime in seconds
    """

    split = int(len(data) * 0.70)

    if split < 50 or len(data) - split < 20:
        raise ValueError(
            "Not enough valid observations for model evaluation."
        )

    train = data.iloc[:split]
    validation = data.iloc[split:]

    X_train = train[features]
    y_train = train["target"]

    X_val = validation[features]
    y_val = validation["target"]

    model = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=42
    )

    start = perf_counter()

    model.fit(X_train, y_train)

    training_time = perf_counter() - start

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)

    return mae, training_time


# ============================================================
# RUN SENSITIVITY EXPERIMENTS
# ============================================================

def run_experiments(series):

    results = []

    for horizon in HORIZON_VALUES:

        data, features = create_features(
            series,
            horizon
        )

        print(
            f"\nHorizon: {horizon} minutes"
            f" | Valid examples: {len(data)}"
        )

        for depth in DEPTH_VALUES:

            for leaf in LEAF_VALUES:

                mae, runtime = evaluate_model(
                    data=data,
                    features=features,
                    depth=depth,
                    leaf=leaf
                )

                results.append({
                    "horizon_minutes": horizon,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "mae_people": mae,
                    "training_time_seconds": runtime
                })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        TABLE_DIR / "decision_tree_experiments.csv",
        index=False
    )

    return results_df


# ============================================================
# CREATE GRAPHS
# ============================================================

def save_graph(x, y, xlabel, ylabel, title, filename):

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()


def generate_graphs(results):

    # Graph 1: Tree depth versus MAE
    depth_data = results[
        (results["horizon_minutes"] == 30)
        & (results["min_samples_leaf"] == 20)
    ].sort_values("max_depth")

    save_graph(
        depth_data["max_depth"],
        depth_data["mae_people"],
        "Maximum tree depth",
        "MAE (people)",
        "Decision Tree Depth vs Forecast Error",
        "tree_depth_vs_error.png"
    )

    # Graph 2: Minimum leaf samples versus MAE
    leaf_data = results[
        (results["horizon_minutes"] == 30)
        & (results["max_depth"] == 6)
    ].sort_values("min_samples_leaf")

    save_graph(
        leaf_data["min_samples_leaf"],
        leaf_data["mae_people"],
        "Minimum samples per leaf",
        "MAE (people)",
        "Leaf Size vs Forecast Error",
        "leaf_size_vs_error.png"
    )

    # Graph 3: Horizon versus MAE
    horizon_data = results[
        (results["max_depth"] == 6)
        & (results["min_samples_leaf"] == 20)
    ].sort_values("horizon_minutes")

    save_graph(
        horizon_data["horizon_minutes"],
        horizon_data["mae_people"],
        "Forecast horizon (minutes)",
        "MAE (people)",
        "Forecast Horizon vs Prediction Error",
        "horizon_vs_error.png"
    )

    # Graph 4: Depth versus training time
    save_graph(
        depth_data["max_depth"],
        depth_data["training_time_seconds"],
        "Maximum tree depth",
        "Training time (seconds)",
        "Tree Depth vs Computational Runtime",
        "tree_depth_vs_runtime.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("MMA3001 DECISION TREE EXPERIMENTS")

    df = load_occupancy_data()

    series = create_time_series(df)

    results = run_experiments(series)

    generate_graphs(results)

    print("\nExperiments completed.")
    print("\nResults table:")
    print(results.round(4).to_string(index=False))

    print("\nGraphs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
