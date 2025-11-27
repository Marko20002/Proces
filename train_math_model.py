
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path




CSV_FILES = [
    "database/Aud1do6kalk.csv",
    "database/Aud1do6kalk2.csv",
    "database/Aud6do12kalk.csv",
    "database/Aud6do12kalk2.csv",
    "database/diskretna_pravila_samo.csv",
    "database/dm_1.csv",
    "database/dm_2.csv",
    "database/verojatnost_ispiti_zadaci_resenija.csv",
    "database/verojatnost_ispiti_zadaci_resenija_part2.csv",
    "database/verojatnost_zadaci_i_formuli.csv",
    "database/verojatnost_zadaci_i_formuli_part2.csv",
]


def load_dataset(csv_files):
    dfs = []
    for path in csv_files:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] CSV file not found: {p}")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
    data = pd.concat(dfs,ignore_index=True)
    data["text"]= data["question"].fillna("") + "/" + data["solution"].fillna("")
    data["label"] = data["label"].astype(str)
    data = data[data["text"].str.len() > 10] # ako tekst ima pomalko od 10 karaktera
    # primer lim samo stoi da se izbrise
    return data

def train_model(load_data):
    X=load_data["text"]
    y=load_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc}")
    out_path = "math_task_classifier.joblib"
    joblib.dump(model, out_path)

if __name__ == "__main__":
    train_model(load_dataset(CSV_FILES))