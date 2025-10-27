import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

df = pd.read_csv("../../../data/preprocessed/dataset.csv")
df['tokens'] = df['review_clean'].astype(str).apply(lambda x: word_tokenize(x.lower()))

w2v_model = Word2Vec(
    sentences=df['tokens'],
    vector_size=25,      # tamaño del vector
    window=5,            # tamaño del contexto
    min_count=2,         # ignora palabras muy raras
    workers=4
)

# Palabras más similares
print(w2v_model.wv.most_similar('worse', topn=4))


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np

# Vectorización TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(df['review_clean'])
y = df['Rating']

# Entrenamiento con TF-IDF
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)
print("=== Resultados con TF-IDF ===")
print(classification_report(y_test, lr_tfidf.predict(X_test_tfidf)))


def vectorizar_texto(tokens, model, vector_size=100):
    palabras_validas = [w for w in tokens if w in model.wv]
    if not palabras_validas:
        return np.zeros(vector_size)
    return np.mean(model.wv[palabras_validas], axis=0)

# Aplicar a todo el dataset
X_embed = np.vstack(df['tokens'].apply(lambda t: vectorizar_texto(t, w2v_model)))
X_train_emb, X_test_emb, y_train, y_test = train_test_split(X_embed, y, test_size=0.2, random_state=42)

# Entrenar modelo sobre embeddings
lr_embed = LogisticRegression(max_iter=1000)
lr_embed.fit(X_train_emb, y_train)
print("=== Resultados con Word2Vec embeddings ===")
print(classification_report(y_test, lr_embed.predict(X_test_emb)))
