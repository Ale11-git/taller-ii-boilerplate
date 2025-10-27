import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import joblib


df = pd.read_csv("../../../data/preprocessed/dataset.csv")
print("Tamaño del dataset:", df.shape)
print(df.head())
print("Distribución de clases:\n", df["Rating"].value_counts())


# Dividimos el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["review_clean"], df["Rating"],
    test_size=0.2,
    random_state=42,
    stratify=df["Rating"] # asegura proporción por clase
)

# Vectorizamos como en vectorizacion.py
bow_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1,2))
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# Para tener una linea base de comparación de nuestros modelos
majority_class = Counter(y_train).most_common(1)[0][0]
y_pred_baseline = [majority_class] * len(y_test)

print("=== Baseline (clase mayoritaria) ===")
print(classification_report(y_test, y_pred_baseline))


def entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test, nombre="Modelo"):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"\n=== {nombre} ===")
    print(classification_report(y_test, y_pred, digits=3))
    # Creamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=modelo.classes_,
                yticklabels=modelo.classes_)
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()
    # Devuelvo F1 macro para comparar
    return f1_score(y_test, y_pred, average="macro")


# Naive Bayes con BoW
nb_bow = MultinomialNB()
f1_nb_bow = entrenar_y_evaluar(nb_bow, X_train_bow, y_train, X_test_bow, y_test, "Naïve Bayes (BoW)")

# Regresión Logística con BoW
lr_bow = LogisticRegression(class_weight="balanced", C=0.01, penalty='l2', solver='saga') #TODO cómo cambia la performance ajustando hiperparámetros?
f1_lr_bow = entrenar_y_evaluar(lr_bow, X_train_bow, y_train, X_test_bow, y_test, "Regresión Logística BoW)")

# SVM Lineal con BoW
svm_bow = LinearSVC(class_weight="balanced", C=10, loss='squared_hinge', penalty='l1') #TODO cómo cambia la performance ajustando hiperparámetros?
f1_svm_bow = entrenar_y_evaluar(svm_bow, X_train_bow, y_train, X_test_bow, y_test, "SVM Lineal (BoW)")

# Naïve Bayes con TF-IDF
nb_tfidf = MultinomialNB()
f1_nb_tfidf = entrenar_y_evaluar(nb_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test, "Naïve Bayes (TF-IDF)")

# Regresión Logística con TF-IDF
lr_tfidf = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial", C=0.01, penalty='l2', solver='lbfgs') #TODO cómo cambia la performance ajustando hiperparámetros?
f1_lr_tfidf = entrenar_y_evaluar(lr_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test, "Regresión logística (TF-IDF)")

# SVM Lineal
svm_tfidf = LinearSVC(class_weight="balanced") #TODO cómo cambia la performance ajustando hiperparámetros?
f1_svm_tfidf = entrenar_y_evaluar(svm_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test, "SVM Lineal (TF-IDF)")

resultados = pd.DataFrame({
    "Modelo": [
        "Naïve Bayes (BoW)", "Regresión Logistica (BoW)", "SVM (BoW)",
        "Naïve Bayes (TF-IDF)", "Regresión Logistica (TF-IDF)", "SVM (TF-IDF)"
    ],
    "Macro-F1": [
        f1_nb_bow, f1_lr_bow, f1_svm_bow,
        f1_nb_tfidf, f1_lr_tfidf, f1_svm_tfidf
    ]
})

print("\n=== Comparación de modelos ===")
print(resultados.sort_values(by="Macro-F1", ascending=False))

joblib.dump(bow_vectorizer, "../../../models/vectorizador_bow.pkl")
joblib.dump(tfidf_vectorizer, "../../../models/vectorizador_tfidf.pkl")
joblib.dump(nb_bow, "../../../models/modelo_nb_bow.pkl")
