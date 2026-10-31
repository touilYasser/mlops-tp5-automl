import time
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


TRACKING_URI = "http://127.0.0.1:5000"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv("data/transactions.csv")
    x = df.drop(columns="fraude")
    y = df["fraude"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    started = time.time()
    model = RandomForestClassifier(
        n_estimators=500, max_depth=16, random_state=42
    ).fit(x_train, y_train)
    duration = round(time.time() - started, 1)
    f1 = round(f1_score(y_test, model.predict(x_test)), 4)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model_manuel.pkl")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("tp5-automl-deploiement")
    with mlflow.start_run(run_name="manuel-rf-500-16"):
        mlflow.log_params(
            {"outil": "scikit-learn", "n_estimators": 500, "max_depth": 16}
        )
        mlflow.log_metrics({"f1": f1, "duree_entrainement_s": duration})
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_test.head(3),
            registered_model_name="fraude-manuel",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    print("modele manuel : F1 =", f1, "| entrainement", duration, "s")
    print("enregistre : fraude-manuel, version", info.registered_model_version)


if __name__ == "__main__":
    main()
