from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    summary_df = pd.read_csv(RESULTS_DIR / "summary_results.csv")
    confusion_df = pd.read_csv(RESULTS_DIR / "confusion_matrices.csv")
    return summary_df, confusion_df


def plot_accuracy_bar(summary_df: pd.DataFrame):
    order = [
        ("imdb", "TF-IDF + Logistic Regression"),
        ("imdb", "fastText"),
        ("ag_news", "TF-IDF + Logistic Regression"),
        ("ag_news", "fastText"),
    ]

    labels = []
    values = []
    errors = []

    for dataset, model in order:
        row = summary_df[
            (summary_df["dataset"] == dataset) & (summary_df["model"] == model)
        ].iloc[0]
        dataset_label = "IMDb" if dataset == "imdb" else "AG News"
        short_model = "TF-IDF + LR" if model == "TF-IDF + Logistic Regression" else "fastText"

        labels.append(f"{dataset_label}\n{short_model}")
        values.append(row["accuracy_mean"])
        errors.append(row["accuracy_std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, yerr=errors, capsize=5)

    ax.set_ylabel("Accuracy")
    # ax.set_title("Final Test Accuracy")
    ax.set_ylim(0, 1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_accuracy_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_bar(summary_df: pd.DataFrame):
    order = [
        ("imdb", "TF-IDF + Logistic Regression"),
        ("imdb", "fastText"),
        ("ag_news", "TF-IDF + Logistic Regression"),
        ("ag_news", "fastText"),
    ]

    labels = []
    values = []
    errors = []

    for dataset, model in order:
        row = summary_df[
            (summary_df["dataset"] == dataset) & (summary_df["model"] == model)
        ].iloc[0]
        dataset_label = "IMDb" if dataset == "imdb" else "AG News"
        short_model = "TF-IDF + LR" if model == "TF-IDF + Logistic Regression" else "fastText"

        labels.append(f"{dataset_label}\n{short_model}")
        values.append(row["runtime_mean_sec"])
        errors.append(row["runtime_std_sec"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, yerr=errors, capsize=5)

    ax.set_ylabel("Total Runtime (s)")
    # ax.set_title("Mean End-to-End Runtime")
    ax.set_ylim(0, max(values) + max(errors) + 5)

    for bar, value, error in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + max(values) * 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_runtime_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_aggregated_confusion_matrix(
    confusion_df: pd.DataFrame,
    dataset: str,
    model: str,
):
    subset = confusion_df[
        (confusion_df["dataset"] == dataset) & (confusion_df["model"] == model)
    ].copy()

    grouped = (
        subset.groupby(["true_label", "pred_label"], as_index=False)["count"]
        .sum()
        .sort_values(["true_label", "pred_label"])
    )

    labels = sorted(set(grouped["true_label"]).union(set(grouped["pred_label"])))
    size = len(labels)
    matrix = np.zeros((size, size), dtype=int)

    for _, row in grouped.iterrows():
        i = labels.index(int(row["true_label"]))
        j = labels.index(int(row["pred_label"]))
        matrix[i, j] = int(row["count"])

    return matrix, labels


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def plot_agnews_fasttext_confusion_matrix(confusion_df: pd.DataFrame):
    matrix, labels = build_aggregated_confusion_matrix(
        confusion_df,
        dataset="ag_news",
        model="fastText",
    )
    norm_matrix = normalize_rows(matrix)

    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm_matrix, aspect="equal")

    # ax.set_title("AG News fastText Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            pct = norm_matrix[i, j] * 100
            count = matrix[i, j]
            text = f"{pct:.1f}%\n({count})"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized proportion")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "agnews_fasttext_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    summary_df, confusion_df = load_data()

    plot_accuracy_bar(summary_df)
    plot_runtime_bar(summary_df)
    plot_agnews_fasttext_confusion_matrix(confusion_df)

    print("Saved figures to:", FIGURES_DIR.resolve())
    print("- final_accuracy_bar.png")
    print("- final_runtime_bar.png")
    print("- agnews_fasttext_confusion_matrix.png")


if __name__ == "__main__":
    main()