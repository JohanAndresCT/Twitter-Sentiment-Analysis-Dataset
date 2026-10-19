"""Compara configuraciones del MLP y guarda una tabla de resultados.

La selección de configuraciones se hace con el conjunto de validación. El
conjunto de test se reserva para la evaluación final del modelo seleccionado.

Ejecutar desde la raíz del proyecto:
    python src/experiments.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

from model import construir_mlp
from preprocess import cargar_datos

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

CONFIGURACIONES = [
    {
        "nombre": "base",
        "hidden_units": (128, 64),
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 64,
    },
    {
        "nombre": "mayor_regularizacion",
        "hidden_units": (128, 64),
        "dropout": 0.5,
        "learning_rate": 1e-3,
        "batch_size": 64,
    },
    {
        "nombre": "red_mas_profunda",
        "hidden_units": (256, 128, 64),
        "dropout": 0.3,
        "learning_rate": 5e-4,
        "batch_size": 64,
    },
]


def fijar_semilla():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def ejecutar_experimento(configuracion, datos):
    x_train, x_val, y_train, y_val, class_weight = datos
    fijar_semilla()
    model = construir_mlp(
        input_dim=x_train.shape[1],
        hidden_units=configuracion["hidden_units"],
        dropout=configuracion["dropout"],
        learning_rate=configuracion["learning_rate"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=configuracion["batch_size"],
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    y_pred = model.predict(x_val, verbose=0).argmax(axis=1)
    return {
        "experimento": configuracion["nombre"],
        "capas_neuronas": "-".join(map(str, configuracion["hidden_units"])),
        "dropout": configuracion["dropout"],
        "learning_rate": configuracion["learning_rate"],
        "batch_size": configuracion["batch_size"],
        "epocas_ejecutadas": len(history.history["loss"]),
        "val_accuracy": accuracy_score(y_val, y_pred),
        "val_f1_macro": f1_score(y_val, y_pred, average="macro"),
        "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),
        "mejor_val_loss": min(history.history["val_loss"]),
    }


def main():
    fijar_semilla()
    x_train, x_val, _, y_train, y_val, _, class_weight, _, _ = cargar_datos(
        data_dir=ROOT / "data"
    )
    datos = (x_train, x_val, y_train, y_val, class_weight)
    resultados = [
        ejecutar_experimento(configuracion, datos)
        for configuracion in CONFIGURACIONES
    ]
    tabla = pd.DataFrame(resultados).sort_values(
        "val_f1_macro", ascending=False
    )
    ARTIFACTS.mkdir(exist_ok=True)
    tabla.to_csv(ARTIFACTS / "experiments.csv", index=False)
    print(tabla.to_string(index=False))
    print(f"\nResultados guardados en {ARTIFACTS / 'experiments.csv'}")


if __name__ == "__main__":
    main()
