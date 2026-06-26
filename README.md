# ONeSAMP_ML

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Released under the GNU GPLv3; see `LICENSE` for details. Developed by Boucher Lab.

ONeSAMP_ML estimates effective population size (Ne) from genomic data in GENEPOP format. The program calculates five summary statistics from an input population file, generates reference population statistics with the ONeSAMP simulator when needed, and uses supervised machine learning models to infer Ne from the summary statistics.

The current implementation supports four regression models:

- Random Forest
- XGBoost
- Lasso regression
- Ridge regression

Hyperparameters were tuned once for each regression model using 5-fold cross-validation during model development. The selected hyperparameters are fixed in the released training workflow.

It is strongly recommended that users read the accompanying manuscript before applying ONeSAMP_ML to empirical data.

## Overview

ONeSAMP_ML has two user-facing execution modes:

| Mode | Purpose |
| --- | --- |
| `train` | Train selected machine learning models using fixed hyperparameters. If the requested `allPopStats` file does not exist, the program first generates it. |
| `infer` | Load trained model files and estimate Ne for an input GENEPOP file. |

The default mode is `train`.

## Requirements

- macOS or Linux
- Python 3.8 or later
- Executable ONeSAMP simulator at `build/OneSamp`

Python packages used by the user-facing workflow:

```bash
pip install pandas numpy numba scikit-learn xgboost joblib matplotlib statsmodels
```

## Installation

Clone the repository:

```bash
git clone https://github.com/NiharaDeSilva/ONeSAMP_ML.git
cd ONeSAMP_ML
```

Make the ONeSAMP simulator executable:

```bash
chmod +x build/OneSamp
```

Create output directories if needed:

```bash
mkdir -p output temp
```

## Configuration

Before running the program, update paths in `config.py` for your system.

Important settings:

```python
BASE_PATH = "/path/to/ONeSAMP_ML"
OUTPUT = os.path.join(BASE_PATH, "Final")
OUTPUT_PATH = os.path.join(OUTPUT, "output_ml_100")
TEMP_DIR = os.path.join(BASE_PATH, "temp")
PLOT_DIR = os.path.join(OUTPUT, "plots_ml_100")
POPULATION_GENERATOR = os.path.join(BASE_PATH, "build", "OneSamp")
```

`BASE_PATH` should point to the local ONeSAMP_ML repository directory. The output and plot directories are created or used by the training and inference workflows.

## Quick Start

Train all models. If the default `allPopStats` file does not already exist, ONeSAMP_ML generates it first:

```bash
python main.py --mode train --o data_100/genePop100x1000_1 --s 1000
```

Train all models using an existing `allPopStats` file:

```bash
python main.py --mode train --o data_100/genePop100x1000_1 --allpopstats /path/to/allPopStats_genePop100x1000_1
```

Run inference with all available trained models:

```bash
python main.py --mode infer --o data_100/genePop100x1000_1 --allpopstats /path/to/allPopStats_genePop100x1000_1
```

Train or infer with selected models only:

```bash
python main.py --mode train --models rf,xb --o data_100/genePop100x1000_1 --allpopstats /path/to/allPopStats_genePop100x1000_1
python main.py --mode infer --models ls,rd --o data_100/genePop100x1000_1 --allpopstats /path/to/allPopStats_genePop100x1000_1
```

## Model Selection

Use `--models` with comma-separated model codes in `train` or `infer` mode.

| Code | Model |
| --- | --- |
| `rf` | Random Forest |
| `xb` | XGBoost |
| `ls` | Lasso regression |
| `rd` | Ridge regression |
| `all` | All four models |

Examples:

```bash
python main.py --mode train --models rf
python main.py --mode train --models rf,xb
python main.py --mode infer --models ls,rd
```

If `--models` is omitted, the default is `all`.

## Command Line Parameters

General syntax:

```bash
python main.py [options]
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--mode` | string | `train` | Execution mode. Choices: `train`, `infer`. |
| `--o` | string | `data_100/genePop100x1000_1` | Input GENEPOP file. |
| `--allpopstats` | string | auto-generated path under `OUTPUT_PATH` | Path to an existing or output `allPopStats` file. In `train` mode, the file is generated if missing. In `infer` mode, it must already exist. |
| `--models` | string | `all` | Comma-separated model codes for `train` or `infer`: `rf`, `xb`, `ls`, `rd`, or `all`. |
| `--s` | integer | `20000` | Number of ONeSAMP trials used when generating a missing `allPopStats` file during training. |
| `--m` | float | `0.05` | Minimum allele frequency. |
| `--r` | float | `0.000000012` | Mutation rate. |
| `--lNe` | integer | `4` | Lower bound of the Ne range. |
| `--uNe` | integer | `200` | Upper bound of the Ne range. |
| `--lT` | float | `0.000048` | Lower bound of the theta range. |
| `--uT` | float | `0.0048` | Upper bound of the theta range. |
| `--lD` | float | `2` | Lower bound of the duration range. |
| `--uD` | float | `8` | Upper bound of the duration range. |
| `--i` | float | `0.2` | Maximum allowed missing data for individuals. |
| `--l` | float | `0.2` | Maximum allowed missing data for loci. |
| `--n` | boolean | `False` | Whether to filter monomorphic loci. |
| `--t` | integer | accepted, currently unused | Repeat count argument retained for compatibility. |

Note: `--n` is currently parsed as a Python boolean argument. If using it from the command line, pass an explicit value such as `--n True`.

## Recommended Workflow

### 1. Prepare the input file

Prepare a population dataset in GENEPOP format.

### 2. Configure paths

Edit `config.py` so `BASE_PATH`, `OUTPUT_PATH`, `TEMP_DIR`, `PLOT_DIR`, and `POPULATION_GENERATOR` match your local or cluster environment.

### 3. Train model files

Train all models and generate reference statistics if needed:

```bash
python main.py \
  --mode train \
  --models all \
  --o data_100/genePop100x1000_1 \
  --s 20000
```

Train all models from an existing `allPopStats` file:

```bash
python main.py \
  --mode train \
  --models all \
  --o data_100/genePop100x1000_1 \
  --allpopstats /path/to/allPopStats_genePop100x1000_1
```

Train selected models:

```bash
python main.py \
  --mode train \
  --models rf,xb \
  --o data_100/genePop100x1000_1 \
  --allpopstats /path/to/allPopStats_genePop100x1000_1
```

### 4. Run inference

Infer Ne using all available trained models:

```bash
python main.py \
  --mode infer \
  --models all \
  --o data_100/genePop100x1000_1 \
  --allpopstats /path/to/allPopStats_genePop100x1000_1
```

Infer Ne using selected trained models:

```bash
python main.py \
  --mode infer \
  --models rf,rd \
  --o data_100/genePop100x1000_1 \
  --allpopstats /path/to/allPopStats_genePop100x1000_1
```

## Outputs

Outputs are written under the paths configured in `config.py`.

Common outputs include:

- `inputPopStats_<input-file-name>`: summary statistics calculated from the input GENEPOP file.
- `allPopStats_genePop<sampleSize>x<numLoci>_1`: reference population statistics generated by ONeSAMP_ML.
- `<OUTPUT_PATH>/<sampleSize>x<numLoci>/`: trained model directory.
- `scaler.joblib`: scaler used for Lasso and Ridge models.
- `RandomForest/rf_model.joblib`: trained Random Forest model.
- `XGBoost/xgb_model.joblib`: trained XGBoost model.
- `Lasso/lasso_model.joblib`: trained Lasso model.
- `Ridge/ridge_model.joblib`: trained Ridge model.
- `training_summary.txt`: combined training report.
- `training_metrics.json`: training metrics in JSON format.
- `cv_results.json`: cross-validation performance summary for trained models.
- Calibration plots under `PLOT_DIR`.

Inference prints prediction summaries to standard output, including median prediction and 95% confidence interval for each selected model.

## Repository Structure

```text
main.py                 Command line entry point
config.py               Path and runtime configuration
statistics.py           GENEPOP parsing and summary statistic calculations
models/train.py         Model training and evaluation
models/predict.py       Prediction, evaluation, and bootstrap uncertainty
models/model_utils.py   Model loading and inference helpers
models/calibration.py   Calibration plot generation
build/OneSamp           ONeSAMP simulation executable
scripts/                Helper and legacy shell/R scripts
```

## Notes

- Training can be computationally intensive, especially when generating a new `allPopStats` file with a large `--s` value.
- `infer` requires previously trained model files in the configured output directory.
- `--allpopstats` should point to the same reference statistics file used for model training and inference.
- Lasso and Ridge require the saved `scaler.joblib` generated during training.

## Citation

Please cite the accompanying ONeSAMP_ML manuscript when using this software. Citation details will be added here when available.

## Help

For questions, issues, or bug reports, please contact suhashidesilva@ufl.edu or open a GitHub Issue.

## License

ONeSAMP_ML is released under the GNU General Public License v3.0. See `LICENSE` for details.
