import time
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from flaml import AutoML
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


BUDGET = 90
TRACKING_URI = "http://127.0.0.1:5000"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv("data/transactions.csv")
    x = df.drop(columns="fraude")
    y = df["fraude"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    automl = AutoML()
    started = time.time()
    automl.fit(
        x_train,
        y_train,
        task="classification",
        metric="f1",
        time_budget=BUDGET,
        seed=42,
        verbose=0,
        log_file_name="reports/flaml_tp5.log",
    )
    duration = round(time.time() - started, 1)
    model = automl.model.estimator
    f1_flaml = round(f1_score(y_test, automl.predict(x_test)), 4)
    f1_model = round(f1_score(y_test, model.predict(x_test)), 4)
    if f1_flaml != f1_model:
        raise RuntimeError(
            f"Le modele exporte diverge de FLAML: {f1_flaml} != {f1_model}"
        )

    joblib.dump(model, "models/model_automl.pkl")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("tp5-automl-deploiement")
    with mlflow.start_run(run_name=f"automl-{BUDGET}s"):
        mlflow.log_params(
            {
                "outil": "flaml",
                "budget_s": BUDGET,
                "famille": automl.best_estimator,
                **{f"cfg_{key}": value for key, value in automl.best_config.items()},
            }
        )
        mlflow.log_metrics({"f1": f1_model, "duree_recherche_s": duration})
        mlflow.log_artifact("reports/flaml_tp5.log")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_test.head(3),
            registered_model_name="fraude-automl",
            serialization_format="cloudpickle",
        )
    print("famille retenue :", automl.best_estimator)
    print("F1 :", f1_model, "| recherche :", duration, "s")
    print("enregistre : fraude-automl, version", info.registered_model_version)


if __name__ == "__main__":
    main()
