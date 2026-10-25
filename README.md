
# Short-Term Room Occupancy Forecasting

## MMA3001 - Numerical Methods and Machine Learning

### Project Overview

This project investigates short-term room occupancy forecasting using the Monash Smart Infrastructure Occupancy Dataset.

The aim is to predict future room headcounts from historical occupancy measurements and temporal features. These predictions could potentially support occupancy-aware building management and HVAC scheduling.

Decision-tree regression is used as the main forecasting method and compared with a persistence baseline, which assumes that future occupancy will remain equal to the current headcount.

## Dataset

The original occupancy dataset contains approximately 9 million time-stamped sensor records.

The preliminary investigation uses the first 250,000 records. The forecasting experiments focus on one selected floor space.

The dataset includes:
- Device and floor-space identifiers
- Occupancy status
- Recorded headcount
- Collection timestamps
- Occupancy-status change timestamps

The original dataset is not distributed with this repository.

To reproduce the experiments, place an authorised copy of the dataset at:

`data/raw/occupancy.csv`

## Computational Method

The forecasting workflow involves:

1. Importing and inspecting the occupancy dataset.
2. Cleaning timestamps and headcount measurements.
3. Reconstructing occupancy measurements at five-minute intervals.
4. Generating historical headcount and temporal features.
5. Splitting observations chronologically into training, validation and reserved test periods.
6. Training decision-tree regression models.
7. Comparing predictions against a persistence baseline.
8. Investigating sensitivity to tree depth, minimum leaf size and forecast horizon.

The model is evaluated using mean absolute error (MAE), measured in people.

## Software Requirements

The project uses Python and the following libraries:

- pandas
- NumPy
- scikit-learn
- Matplotlib
- pytest
- pytest-html
- pdoc

Install the required libraries by running:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Open a terminal in the project root directory.

Run the initial data audit:

```bash
python src/data_audit.py
```

Run the forecasting experiments:

```bash
python src/decision_tree_experiments.py
```

## Results

Generated outputs are stored in the `results/` directory.

### Figures

- `tree_depth_vs_error.png` - Decision-tree depth against validation MAE.
- `leaf_size_vs_error.png` - Minimum leaf size against validation MAE.
- `horizon_vs_error.png` - Forecast horizon against validation MAE.
- `tree_depth_vs_runtime.png` - Decision-tree depth against training runtime.

### Tables

- `initial_data_audit.txt`
- `floorspace_summary.csv`
- `missing_values.csv`
- `decision_tree_experiments.csv`

## Automated Testing

Run the project tests using:

```bash
python -m pytest tests/ -v
```

Generate an HTML test report:

```bash
python -m pytest tests/ -v --html=results/test_report.html --self-contained-html
```

The test report records the results of the automated checks.

## HTML Documentation

Generate the source-code documentation using:

```bash
python -m pdoc src -o docs
```

The generated documentation is stored in the `docs/` directory.

## Project Limitations

The current experiments are based on an initial dataset sample and a single selected floor space.

The occupancy sensors record observations at irregular intervals, requiring assumptions when reconstructing a regular time series.

Reported validation performance does not establish generalisation to all floor spaces or unseen periods. Further testing is required before the model can be considered suitable for operational building control.

## AI Use

ChatGPT was used to assist with Python development, debugging, documentation, testing guidance and report preparation. AI-generated suggestions require independent verification. The accompanying report contains a more detailed AI-use reflection.

## Licence

See the `LICENSE` file for the applicable source-code licence once added. The occupancy dataset remains subject to its owner's usage conditions.

