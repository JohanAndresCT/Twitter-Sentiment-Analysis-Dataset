"""Evalúa el modelo en test y genera métricas y figuras.

Ejecutar después de ``python src/train.py``:
    python src/evaluate.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from preprocess import limpiar

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main():
    model_path = ARTIFACTS / "best_model.keras"
    preprocessing_path = ARTIFACTS / "preprocessing.pkl"
    if not model_path.exists() or not preprocessing_path.exists():
        raise FileNotFoundError(
            "Faltan artefactos. Ejecuta primero: python src/train.py"
        )

    test = pd.read_csv(ROOT / "data" / "test.csv")
    with open(preprocessing_path, "rb") as file:
        preprocessing = pickle.load(file)

    tfidf = preprocessing["tfidf"]
    label_encoder = preprocessing["label_encoder"]
    x_test = tfidf.transform(test["text"].apply(limpiar)).toarray()
    y_test = label_encoder.transform(test["airline_sentiment"])
    model = keras.models.load_model(model_path)

    probabilities = model.predict(x_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    labels = list(label_encoder.classes_)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "n_test": int(len(y_test)),
    }
    print(json.dumps(metrics, indent=2))

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = classification_report(
        y_test, y_pred, target_names=labels, digits=4
    )
    (ARTIFACTS / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=labels, yticklabels=labels,
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - conjunto de test")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=150)
    plt.close()

    history_path = ARTIFACTS / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        plt.figure(figsize=(8, 4))
        plt.plot(history["loss"], label="train")
        plt.plot(history["val_loss"], label="validation")
        plt.xlabel("Época")
        plt.ylabel("Pérdida")
        plt.title("Curvas de pérdida")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "loss_curves.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
