import ast
import re
import time
import tempfile
from itertools import product
from pathlib import Path

import fasttext
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

VAL_SPLIT_SEED = 42
RUN_SEEDS = [1, 2, 3, 4, 5]

TFIDF_GRID = list(product([(1, 1), (1, 2)], [20_000, 50_000], [0.5, 1.0, 2.0]))
FASTTEXT_GRID = list(product([0.1, 0.5, 1.0], [5, 10, 25], [1, 2]))
MODEL_NAMES = {"tfidf": "TF-IDF + Logistic Regression", "fasttext": "fastText"}


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").lower()
    return re.sub(r"\s+", " ", text).strip()


def load_data(dataset_name: str):
    dataset = load_dataset(dataset_name)
    train_df = pd.DataFrame(dataset["train"])[["text", "label"]]
    test_df = pd.DataFrame(dataset["test"])[["text", "label"]]

    for df in (train_df, test_df):
        df["text"] = df["text"].map(clean_text)
        df["label"] = df["label"].astype(int)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    test_df["example_id"] = test_df.index

    train_sub, valid_sub = train_test_split(
        train_df,
        test_size=0.10,
        random_state=VAL_SPLIT_SEED,
        stratify=train_df["label"],
    )

    return (
        train_sub.reset_index(drop=True),
        valid_sub.reset_index(drop=True),
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def write_fasttext_file(df: pd.DataFrame) -> Path:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
        for text, label in zip(df["text"], df["label"]):
            f.write(f"__label__{label} {text}\n")
    return Path(f.name)


def predict_fasttext(model, texts):
    preds = []
    for text in texts:
        labels, _ = model.predict(text)
        preds.append(int(labels[0].replace("__label__", "")))
    return preds


def train_and_eval_tfidf(
    train_df: pd.DataFrame,
    eval_texts,
    eval_labels,
    ngram_range,
    max_features,
    c,
    return_predictions: bool = False,
    random_seed: int = VAL_SPLIT_SEED,
):
    start = time.perf_counter()

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=False,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_df["text"])
    x_eval = vectorizer.transform(eval_texts)

    model = LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=2000,
        random_state=random_seed,
    )
    model.fit(x_train, train_df["label"])
    preds = model.predict(x_eval)
    preds = [int(p) for p in preds]

    runtime = time.perf_counter() - start
    metrics = get_metrics(eval_labels, preds)
    metrics["total_runtime_sec"] = runtime

    if return_predictions:
        return metrics, preds
    return metrics


def train_and_eval_fasttext(
    train_df: pd.DataFrame,
    eval_texts,
    eval_labels,
    lr,
    epoch,
    word_ngrams,
    return_predictions: bool = False,
):
    start = time.perf_counter()

    train_file = write_fasttext_file(train_df)
    try:
        model = fasttext.train_supervised(
            input=str(train_file),
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            dim=100,
            loss="softmax",
            thread=1,
            verbose=0,
        )
        preds = predict_fasttext(model, eval_texts)
    finally:
        train_file.unlink(missing_ok=True)

    runtime = time.perf_counter() - start
    metrics = get_metrics(eval_labels, preds)
    metrics["total_runtime_sec"] = runtime

    if return_predictions:
        return metrics, preds
    return metrics

def build_confusion_matrix_rows(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = predictions_df.groupby(["dataset", "model", "run", "seed"], sort=False)
    for (dataset, model, run, seed), group in grouped:
        labels = sorted(set(group["true_label"]).union(set(group["pred_label"])))
        cm = confusion_matrix(group["true_label"], group["pred_label"], labels=labels)

        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "run": run,
                        "seed": seed,
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "count": int(cm[i, j]),
                    }
                )

    return pd.DataFrame(rows)

def run_validation_search(dataset_name: str, model_name: str, train_sub: pd.DataFrame, valid_sub: pd.DataFrame):
    rows = []

    if model_name == "tfidf":
        for ngram_range, max_features, c in TFIDF_GRID:
            print(
                f"[{dataset_name}] TF-IDF + Logistic Regression | "
                f"ngram_range={ngram_range}, max_features={max_features}, C={c}"
            )
            metrics = train_and_eval_tfidf(
                train_sub,
                valid_sub["text"],
                valid_sub["label"],
                ngram_range,
                max_features,
                c,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": MODEL_NAMES[model_name],
                    "split": "validation",
                    "ngram_range": str(ngram_range),
                    "max_features": max_features,
                    "C": c,
                    "lr": None,
                    "epoch": None,
                    "wordNgrams": None,
                    **metrics,
                }
            )
    else:
        for lr, epoch, word_ngrams in FASTTEXT_GRID:
            print(
                f"[{dataset_name}] fastText | lr={lr}, epoch={epoch}, wordNgrams={word_ngrams}"
            )
            metrics = train_and_eval_fasttext(
                train_sub,
                valid_sub["text"],
                valid_sub["label"],
                lr,
                epoch,
                word_ngrams,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": MODEL_NAMES[model_name],
                    "split": "validation",
                    "ngram_range": None,
                    "max_features": None,
                    "C": None,
                    "lr": lr,
                    "epoch": epoch,
                    "wordNgrams": word_ngrams,
                    **metrics,
                }
            )

    search_df = pd.DataFrame(rows)
    best_row = (
        search_df.sort_values(
            by=["macro_f1", "accuracy", "total_runtime_sec"],
            ascending=[False, False, True],
        )
        .iloc[0]
        .to_dict()
    )
    return search_df, best_row


def final_runs(
    dataset_name: str,
    model_name: str,
    full_train: pd.DataFrame,
    test_df: pd.DataFrame,
    best_row,
):
    metric_rows = []
    prediction_rows = []
    misclassified_rows = []

    for run_id, seed in enumerate(RUN_SEEDS, start=1):
        shuffled_train = full_train.sample(frac=1, random_state=seed).reset_index(drop=True)

        if model_name == "tfidf":
            metrics, preds = train_and_eval_tfidf(
                shuffled_train,
                test_df["text"],
                test_df["label"],
                ast.literal_eval(best_row["ngram_range"]),
                int(best_row["max_features"]),
                float(best_row["C"]),
                return_predictions=True,
                random_seed=seed,
            )
        else:
            metrics, preds = train_and_eval_fasttext(
                shuffled_train,
                test_df["text"],
                test_df["label"],
                float(best_row["lr"]),
                int(best_row["epoch"]),
                int(best_row["wordNgrams"]),
                return_predictions=True,
            )

        metric_rows.append(
            {
                "dataset": dataset_name,
                "model": MODEL_NAMES[model_name],
                "run": run_id,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "total_runtime_sec": metrics["total_runtime_sec"],
            }
        )

        run_predictions = pd.DataFrame(
            {
                "dataset": dataset_name,
                "model": MODEL_NAMES[model_name],
                "run": run_id,
                "seed": seed,
                "example_id": test_df["example_id"],
                "true_label": test_df["label"],
                "pred_label": preds,
            }
        )
        run_predictions["is_correct"] = run_predictions["true_label"] == run_predictions["pred_label"]
        prediction_rows.append(run_predictions)

        run_misclassified = run_predictions.loc[~run_predictions["is_correct"]].merge(
            test_df[["example_id", "text"]],
            on="example_id",
            how="left",
        )
        misclassified_rows.append(run_misclassified)

    final_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    misclassified_df = pd.concat(misclassified_rows, ignore_index=True)

    return final_df, predictions_df, misclassified_df


def summarize_results(final_df: pd.DataFrame) -> pd.DataFrame:
    return (
        final_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            runtime_mean_sec=("total_runtime_sec", "mean"),
            runtime_std_sec=("total_runtime_sec", "std"),
        )
        .sort_values(["dataset", "accuracy_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )


def format_selected_params(row: dict) -> str:
    if row["model"] == MODEL_NAMES["tfidf"]:
        grams = "unigrams" if row["ngram_range"] == "(1, 1)" else "unigrams + bigrams"
        return f"{grams}, {int(row['max_features'])} features, C={float(row['C'])}"
    return f"lr={float(row['lr'])}, epochs={int(row['epoch'])}, wordNgrams={int(row['wordNgrams'])}"


def main():
    validation_results = []
    selected_rows = []
    final_results = []
    all_predictions = []
    all_misclassified = []

    for dataset_name in ["imdb", "ag_news"]:
        print(f"\n=== {dataset_name.upper()} ===")
        train_sub, valid_sub, full_train, test_df = load_data(dataset_name)

        for model_name in ["tfidf", "fasttext"]:
            search_df, best_row = run_validation_search(dataset_name, model_name, train_sub, valid_sub)
            validation_results.append(search_df)

            selected_rows.append(
                {
                    "dataset": dataset_name,
                    "model": MODEL_NAMES[model_name],
                    "selected_hyperparameters": format_selected_params(best_row),
                }
            )
            print(f"Best {MODEL_NAMES[model_name]} config: {format_selected_params(best_row)}")

            final_df_one, predictions_df_one, misclassified_df_one = final_runs(
                dataset_name,
                model_name,
                full_train,
                test_df,
                best_row,
            )

            final_results.append(final_df_one)
            all_predictions.append(predictions_df_one)
            all_misclassified.append(misclassified_df_one)

    validation_df = pd.concat(validation_results, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    final_df = pd.concat(final_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    misclassified_df = pd.concat(all_misclassified, ignore_index=True)

    summary_df = summarize_results(final_df)
    confusion_df = build_confusion_matrix_rows(predictions_df)

    validation_df.to_csv(RESULTS_DIR / "validation_results.csv", index=False)
    selected_df.to_csv(RESULTS_DIR / "selected_hyperparameters.csv", index=False)
    final_df.to_csv(RESULTS_DIR / "final_runs.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "summary_results.csv", index=False)

    predictions_df.to_csv(RESULTS_DIR / "final_predictions.csv", index=False)
    confusion_df.to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False)
    misclassified_df.to_csv(RESULTS_DIR / "misclassified_examples.csv", index=False)

    print("\n=== Selected hyperparameters ===")
    print(selected_df.to_string(index=False))
    print("\n=== Final results over 5 runs ===")
    print(summary_df.round(4).to_string(index=False))
    print("\nSaved additional files:")
    print("- final_predictions.csv")
    print("- confusion_matrices.csv")
    print("- misclassified_examples.csv")


if __name__ == "__main__":
    main()
