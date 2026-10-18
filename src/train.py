"""Entrena el MLP y guarda todo lo necesario para evaluar y predecir.

Ejecutar desde cualquier directorio:
    python src/train.py
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from model import construir_mlp
from preprocess import cargar_datos

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    (
        x_train, x_val, _, y_train, y_val, _, class_weight, label_encoder, tfidf,
    ) = cargar_datos(data_dir=ROOT / "data")

    model = construir_mlp(input_dim=x_train.shape[1])
    ARTIFACTS.mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(ARTIFACTS / "training_log.csv"),
        keras.callbacks.ModelCheckpoint(
            ARTIFACTS / "best_model.keras", monitor="val_loss",
            save_best_only=True
        ),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(ARTIFACTS / "model.keras")
    with open(ARTIFACTS / "preprocessing.pkl", "wb") as file:
        pickle.dump({"tfidf": tfidf, "label_encoder": label_encoder}, file)

    # Se conserva una copia explícita para graficar incluso si el entrenamiento
    # se detuvo antes de completar todas las épocas.
    pd.DataFrame(history.history).to_csv(ARTIFACTS / "history.csv", index=False)
    print(f"Modelo y artefactos guardados en {ARTIFACTS}")


if __name__ == "__main__":
    main()
