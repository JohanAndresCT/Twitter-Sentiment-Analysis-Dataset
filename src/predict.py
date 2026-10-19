"""Demo ejecutable para predecir el sentimiento de un tweet.

Uso:
    python src/predict.py "I love this airline!"
"""

import argparse
import pickle
from pathlib import Path

from tensorflow import keras

from preprocess import limpiar

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main():
    parser = argparse.ArgumentParser(description="Clasificador de sentimiento")
    parser.add_argument("text", nargs="+", help="Texto del tweet")
    args = parser.parse_args()

    model_path = ARTIFACTS / "best_model.keras"
    preprocessing_path = ARTIFACTS / "preprocessing.pkl"
    if not model_path.exists() or not preprocessing_path.exists():
        raise FileNotFoundError(
            "Faltan artefactos. Ejecuta primero: python src/train.py"
        )

    with open(preprocessing_path, "rb") as file:
        preprocessing = pickle.load(file)
    text = " ".join(args.text)
    vector = preprocessing["tfidf"].transform([limpiar(text)]).toarray()
    probabilities = keras.models.load_model(model_path).predict(vector, verbose=0)[0]
    index = probabilities.argmax()
    sentiment = preprocessing["label_encoder"].inverse_transform([index])[0]
    print(f"Sentimiento: {sentiment}")
    print(f"Confianza: {probabilities[index]:.2%}")


if __name__ == "__main__":
    main()
