"""Construcción del MLP usado para clasificar sentimientos."""

from tensorflow import keras
from tensorflow.keras import layers


def construir_mlp(input_dim, num_classes=3, hidden_units=(128, 64),
                  dropout=0.5, learning_rate=1e-4):
    """Crea y compila un MLP reproducible para vectores TF-IDF."""
    model = keras.Sequential(name="twitter_sentiment_mlp")
    model.add(keras.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model