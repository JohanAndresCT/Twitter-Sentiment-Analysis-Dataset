"""
preprocess.py
Limpia el texto, lo convierte a vectores TF-IDF y prepara etiquetas
y pesos de clase.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def limpiar(texto):
    """Minúsculas, sin menciones, sin links y sin signos/números."""
    texto = texto.lower()
    texto = re.sub(r"@\w+", "", texto)       # quita @menciones
    texto = re.sub(r"http\S+", "", texto)    # quita links
    texto = re.sub(r"[^a-z\s#]", " ", texto) # deja solo letras y #
    return re.sub(r"\s+", " ", texto).strip()


def cargar_datos(max_features=10000, data_dir=None):
    """Devuelve los datos vectorizados, pesos de clase, LabelEncoder y TF-IDF."""
    data_dir = (
        Path(data_dir)
        if data_dir
        else Path(__file__).resolve().parent.parent / "data"
    )
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    for d in (train, val, test):
        d["clean"] = d["text"].apply(limpiar)

    # TF-IDF: se ajusta SOLO con train para no filtrar información de test
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=max_features, dtype=np.float32
    )
    X_train = tfidf.fit_transform(train["clean"]).toarray()
    X_val = tfidf.transform(val["clean"]).toarray()
    X_test = tfidf.transform(test["clean"]).toarray()

    # Etiquetas de texto a números (negative=0, neutral=1, positive=2)
    le = LabelEncoder()
    y_train = le.fit_transform(train["airline_sentiment"])
    y_val = le.transform(val["airline_sentiment"])
    y_test = le.transform(test["airline_sentiment"])

    # Pesos de clase para compensar el desbalance (muchos más negativos)
    pesos = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(pesos))

    return (
        X_train, X_val, X_test, y_train, y_val, y_test,
        class_weight, le, tfidf,
    )
