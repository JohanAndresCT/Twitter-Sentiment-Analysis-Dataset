"""
make_split.py
Descarga el dataset Twitter Airline Sentiment y genera los splits
train / val / test que usará todo el equipo.
Ejecutar desde la raíz del proyecto: python src/make_split.py
"""
import os
import shutil
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

SEED = 42  # semilla fija: todos obtienen exactamente los mismos splits

path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
os.makedirs("data", exist_ok=True)
shutil.copy(os.path.join(path, "Tweets.csv"), "data/Tweets.csv")

df = pd.read_csv("data/Tweets.csv")[["text", "airline_sentiment"]]
df = df.dropna().drop_duplicates()

# 4. Split estratificado: 70% train, 15% val, 15% test
# stratify mantiene la misma proporción de clases en los tres conjuntos

train, temp = train_test_split(
    df, test_size=0.30, stratify=df["airline_sentiment"], random_state=SEED
)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["airline_sentiment"], random_state=SEED
)

# 5. Guardar los tres conjuntos
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("train / val / test:", len(train), len(val), len(test))