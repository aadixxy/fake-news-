# Fake News Detection using TF-IDF and LinearSVC

A beginner-friendly NLP machine learning project that classifies news
articles as **FAKE** or **REAL** using TF-IDF features and a linear model.

## Dataset

This project uses the Kaggle **Fake and Real News Dataset**:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Kaggle describes the dataset as an ISOT fake-news binary text-classification
dataset with two files:

- `Fake.csv` — 23,502 fake news articles
- `True.csv` — 21,417 true news articles

The CSVs contain title, article text, subject, and date.

### Why the CSV files are not in this GitHub repository

`Fake.csv` alone is larger than GitHub's 25 MiB browser upload limit.
The dataset is also external data and should not be committed into the code
repository.

Instead, this project uses **KaggleHub** to download the dataset
automatically when the program runs.

## Project structure

```text
Fake-News-Detection/
│
├── fake_news_detection.py
├── predict.py
├── run_project.bat
├── run_prediction.bat
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   └── fake_news_detection.py
│
├── data/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
└── models/
    └── .gitkeep
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- KaggleHub
- NLP
- TF-IDF
- LinearSVC

## Installation

Open PowerShell in the project folder:

```powershell
python -m pip install -r requirements.txt
```

## Run the complete project

```powershell
python fake_news_detection.py
```

or simply double-click:

```text
run_project.bat
```

The program will:

1. Download the Kaggle dataset.
2. Find `Fake.csv` and `True.csv`.
3. Add FAKE and REAL labels.
4. Combine title and article body.
5. Clean the text.
6. Remove empty and duplicate articles.
7. Split the data into 80% training and 20% testing.
8. Convert text to TF-IDF features.
9. Train LinearSVC.
10. Calculate accuracy, precision, recall and F1-score.
11. Generate a confusion matrix.
12. Generate basic EDA charts.
13. Save the trained model.
14. Allow custom news predictions.

## Model pipeline

```text
Kaggle Dataset
      |
      v
Fake.csv + True.csv
      |
      v
FAKE / REAL Labels
      |
      v
Title + Article Text
      |
      v
Text Cleaning
      |
      v
80/20 Stratified Split
      |
      v
TF-IDF (Unigrams + Bigrams)
      |
      v
LinearSVC
      |
      v
Evaluation
      |
      +--> Accuracy
      +--> Precision
      +--> Recall
      +--> F1-score
      +--> Confusion Matrix
```

## Generated files

After the first run:

```text
results/
├── processed_dataset.csv
├── metrics.csv
├── metrics.json
├── classification_report.txt
├── confusion_matrix.png
├── class_distribution.png
├── average_article_length.png
├── test_predictions.csv
└── project_summary.json

models/
└── fake_news_tfidf_linearsvc.joblib
```

These files are generated locally and are intentionally ignored by Git by
default.

## Custom prediction

After training:

```powershell
python predict.py
```

or double-click:

```text
run_prediction.bat
```

## GitHub upload

Upload the source code and project files, but **do not upload**:

```text
Fake.csv
True.csv
data/kaggle_dataset/
```

The `.gitignore` already prevents the large data files and generated model/
results from being committed.

Recommended repository contents:

```text
fake_news_detection.py
predict.py
run_project.bat
run_prediction.bat
requirements.txt
.gitignore
README.md
src/
```

## Dataset attribution and license

The Kaggle dataset page lists the dataset license as
**CC BY-NC-SA 4.0**. Review the dataset's current license terms before
redistributing the raw data or using it commercially.

## Limitations

This is a machine-learning classifier, not an independent fact-checking
system. A FAKE/REAL prediction means the article resembles patterns learned
from the training data; it does not prove that a real-world claim is true or
false.

## Learning outcomes

This project demonstrates:

- Text preprocessing
- TF-IDF vectorization
- Train/test splitting
- Linear classification
- Classification metrics
- Confusion matrix
- Exploratory data analysis
- Model persistence
- Reproducible GitHub project organization
