from pathlib import Path
import re
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_tfidf_linearsvc.joblib"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run run_project.bat first."
    )

model = joblib.load(MODEL_PATH)

print("=" * 50)
print("FAKE NEWS DETECTOR")
print("=" * 50)
print("Enter a headline/article.")
print("Press Enter on an empty line to exit.")

while True:
    news = input("\nNews: ").strip()

    if not news:
        break

    prediction = model.predict([clean_text(news)])[0]
    print("Prediction:", prediction)
