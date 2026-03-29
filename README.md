# Comparison of Text Classifications TF-IDF and fastText

A lightweight benchmark for text classification with two baseline approaches:

- TF-IDF + Logistic Regression
- fastText

The project runs experiments on two standard datasets:

- IMDb
- AG News

It includes training, validation-based hyperparameter selection, final evaluation across multiple runs, result export, and figure generation.

## Features

- unified pipeline for two text classification models
- automatic hyperparameter search
- repeated final evaluation with fixed seeds
- export of predictions, confusion matrices, and misclassified examples
- generation of summary plots

## Repository Structure

```text
.
├── try_3.py      # main experiment script
├── plot.py       # plotting script
├── results/      # generated CSV files
└── figures/      # generated plots
```

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- scikit-learn
- datasets
- fasttext

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn datasets fasttext
```

## Run Experiments

```bash
python try_3.py
```

This will:

- load and preprocess the datasets
- split training data into train/validation
- search for the best hyperparameters
- run final evaluation
- save outputs to `results/`

## Generate Figures

```bash
python plot.py
```

Generated plots will be saved to `figures/`.

## Output Files

The `results/` directory contains:

- `validation_results.csv` — validation results for all hyperparameter combinations
- `selected_hyperparameters.csv` — best settings for each model
- `final_runs.csv` — metrics for each final run
- `summary_results.csv` — aggregated final results
- `final_predictions.csv` — predictions for test examples
- `confusion_matrices.csv` — confusion matrix counts
- `misclassified_examples.csv` — incorrectly classified examples

The `figures/` directory contains:

- `final_accuracy_bar.png`
- `final_runtime_bar.png`
- `agnews_fasttext_confusion_matrix.png`

## Reproducibility

The pipeline uses fixed seeds for validation splitting and final runs. All outputs are saved automatically, making it easier to reproduce and inspect results.
