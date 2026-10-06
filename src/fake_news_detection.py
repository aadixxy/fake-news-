"""
Fake News Detection
====================
Task: Classify news articles as FAKE or REAL.

Dataset:
    Kaggle - Fake and Real News Dataset
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Model:
    TF-IDF + Linear Support Vector Classifier (LinearSVC)

The large Kaggle CSV files are intentionally NOT stored in this GitHub
repository. The program downloads them automatically with KaggleHub.
"""

from pathlib import Path
import re
import json

import joblib
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
RANDOM_STATE = 42
TEST_SIZE = 0.20

for folder in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Text preprocessing
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Perform simple NLP preprocessing."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"<.*?>", " ", text)            # HTML
    text = re.sub(r"[^a-z\s]", " ", text)         # numbers/punctuation
    text = re.sub(r"\s+", " ", text).strip()      # extra spaces
    return text


# ---------------------------------------------------------------------
# 2. Download Kaggle dataset
# ---------------------------------------------------------------------

def download_dataset() -> tuple[Path, Path]:
    """
    Download the public Kaggle dataset and return paths to Fake.csv
    and True.csv.
    """
    print("\n[1/7] Downloading Kaggle dataset...")
    dataset_dir = kagglehub.dataset_download(
        KAGGLE_DATASET,
        output_dir=str(DATA_DIR / "kaggle_dataset"),
    )

    dataset_dir = Path(dataset_dir)

    fake_files = [p for p in dataset_dir.rglob("*.csv")
                  if p.name.lower() == "fake.csv"]
    true_files = [p for p in dataset_dir.rglob("*.csv")
                  if p.name.lower() == "true.csv"]

    if not fake_files or not true_files:
        raise FileNotFoundError(
            "Fake.csv and/or True.csv were not found after downloading "
            "the Kaggle dataset."
        )

    fake_path = fake_files[0]
    true_path = true_files[0]

    print(f"Fake.csv: {fake_path}")
    print(f"True.csv: {true_path}")

    return fake_path, true_path


# ---------------------------------------------------------------------
# 3. Load and prepare the dataset
# ---------------------------------------------------------------------

def prepare_dataset(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Load both CSV files, label them, clean text and remove duplicates."""
    print("\n[2/7] Loading and preparing data...")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(true_path)

    # Keep the fields relevant to this project.
    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"

    df = pd.concat([fake_df, real_df], ignore_index=True)

    required_columns = {"title", "text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Use title + article body as the classifier input.
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df["clean_text"] = df["content"].map(clean_text)

    before = len(df)

    # Remove empty text and duplicate articles.
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    print(f"Original records : {before:,}")
    print(f"Final records    : {len(df):,}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # These files are generated locally and are ignored by Git.
    df.to_csv(RESULTS_DIR / "processed_dataset.csv", index=False)

    return df


# ---------------------------------------------------------------------
# 4. Exploratory Data Analysis
# ---------------------------------------------------------------------

def create_eda(df: pd.DataFrame) -> None:
    """Create simple charts required for the project."""
    print("\n[3/7] Creating EDA charts...")

    # Class distribution
    class_counts = df["label"].value_counts().reindex(["FAKE", "REAL"])

    plt.figure(figsize=(7, 5))
    class_counts.plot(kind="bar")
    plt.title("Fake vs Real News Distribution")
    plt.xlabel("News Class")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=160)
    plt.close()

    # Article length
    word_count = df["content"].str.split().str.len()

    length_df = pd.DataFrame({
        "label": df["label"],
        "word_count": word_count,
    })

    average_length = (
        length_df.groupby("label")["word_count"]
        .mean()
        .reindex(["FAKE", "REAL"])
    )

    plt.figure(figsize=(7, 5))
    average_length.plot(kind="bar")
    plt.title("Average Article Length")
    plt.xlabel("News Class")
    plt.ylabel("Average Number of Words")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "average_article_length.png", dpi=160)
    plt.close()

    print("Saved EDA charts in results/.")


# ---------------------------------------------------------------------
# 5. Train TF-IDF + LinearSVC
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
    """Create the complete text-classification pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50_000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LinearSVC(random_state=RANDOM_STATE),
        ),
    ])


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train, evaluate and save the model plus evaluation artifacts."""
    print("\n[4/7] Splitting data...")

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records : {len(X_test):,}")

    print("\n[5/7] Training model...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Training complete.")

    print("\n[6/7] Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        labels=["FAKE", "REAL"],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=["FAKE", "REAL"],
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["FAKE", "REAL"],
    )

    metrics = {
        "accuracy": float(accuracy),
        "fake_precision": float(report["FAKE"]["precision"]),
        "fake_recall": float(report["FAKE"]["recall"]),
        "fake_f1": float(report["FAKE"]["f1-score"]),
        "real_precision": float(report["REAL"]["precision"]),
        "real_recall": float(report["REAL"]["recall"]),
        "real_f1": float(report["REAL"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
    }

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save metrics.
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
    )

    (RESULTS_DIR / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    # Save confusion matrix image.
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["FAKE", "REAL"],
    ).plot(ax=ax)
    ax.set_title("Confusion Matrix - TF-IDF + LinearSVC")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Save test predictions.
    pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": y_pred,
    }).to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False,
    )

    # Save trained pipeline.
    model_path = MODELS_DIR / "fake_news_tfidf_linearsvc.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model: {model_path}")

    return model, metrics


# ---------------------------------------------------------------------
# 6. Example + custom predictions
# ---------------------------------------------------------------------

def predict_news(model: Pipeline, news_text: str) -> str:
    """Return FAKE or REAL for new text."""
    cleaned = clean_text(news_text)

    if not cleaned:
        raise ValueError("The news text is empty.")

    return str(model.predict([cleaned])[0])


def interactive_prediction(model: Pipeline) -> None:
    """Allow the user to enter news manually."""
    print("\n" + "=" * 60)
    print("CUSTOM NEWS PREDICTION")
    print("=" * 60)
    print("Paste a headline/article and press Enter.")
    print("Press Enter on an empty line to exit.")

    while True:
        text = input("\nNews: ").strip()

        if not text:
            break

        prediction = predict_news(model, text)
        print("Prediction:", prediction)


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("TASK 2 - FAKE NEWS DETECTION")
    print("=" * 60)

    fake_path, true_path = download_dataset()
    df = prepare_dataset(fake_path, true_path)
    create_eda(df)
    model, metrics = train_and_evaluate(df)

    print("\nExample prediction 1:")
    example_1 = (
        "Government officials announced a new public policy "
        "after a formal review."
    )
    print(example_1)
    print("Prediction:", predict_news(model, example_1))

    print("\nExample prediction 2:")
    example_2 = (
        "Scientists discovered a magical device that makes people "
        "live forever overnight."
    )
    print(example_2)
    print("Prediction:", predict_news(model, example_2))

    # Save a small project summary.
    summary = {
        "dataset": KAGGLE_DATASET,
        "model": "TF-IDF + LinearSVC",
        **metrics,
    }

    (RESULTS_DIR / "project_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nProject completed successfully.")
    interactive_prediction(model)


if __name__ == "__main__":
    main()
