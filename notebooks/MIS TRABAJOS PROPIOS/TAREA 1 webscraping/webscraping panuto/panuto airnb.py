# -*- coding: utf-8 -*-
# ============================================================
# Pipeline completo: Preprocesamiento + Entrenamiento + Reportes
# ============================================================
PATH_IN  = r"C:\Users\MSI KATANA\taller-ii-boilerplate\data\preprocessed\reviews_tokens.csv"
PATH_OUT = r"C:\Users\MSI KATANA\taller-ii-boilerplate\data\preprocessed\dataset.csv"
SAVE_MODELS = True

# -----------------------------------------
# 2) ENTRENAMIENTO / EVALUACIÓN DE MODELOS
# -----------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd
df=pd.read_csv("C:\\Users\\ROG BY NG\\OneDrive\\Escritorio\\UNIVERSIDAD UNSTA\\2DO AÑO\\TALLER\\2do cuatrimestre\\taller-ii-boilerplate\\notebooks\\MIS TRABAJOS PROPIOS\\TAREA 1 webscraping\\reviews_tokens_clean.csv")
X = df["review_clean"]
y = df["Rating_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nDistribución de clases (train):")
print(pd.Series(y_train).value_counts().sort_index())

# Vectorizadores con filtro de rarezas
bow_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow  = bow_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

# ---------- Balanceo por oversampling SOLO en TRAIN ----------
# pip install imbalanced-learn
try:
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_bow,   y_train = ros.fit_resample(X_train_bow,   y_train)
    X_train_tfidf, y_train = ros.fit_resample(X_train_tfidf, y_train)
    print("Oversampling aplicado. Nuevas proporciones:", Counter(y_train))
except Exception as e:
    print("Aviso: no se aplicó oversampling (instalá 'imbalanced-learn' si lo querés).", e)

# ---------- Helper de entrenamiento ----------
classes_all = np.sort(df["rating_num"].unique())

def entrenar_y_evaluar(modelo, Xtr, ytr, Xte, yte, nombre="Modelo"):
    modelo.fit(Xtr, ytr)
    y_pred = modelo.predict(Xte)
    print(f"\n=== {nombre} ===")
    print(classification_report(yte, y_pred, digits=3, zero_division=0))
    print("Balanced Acc:", balanced_accuracy_score(yte, y_pred))

    cm = confusion_matrix(yte, y_pred, labels=classes_all)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes_all, yticklabels=classes_all)
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicho"); plt.ylabel("Real")
    plt.tight_layout(); plt.show()

    return f1_score(yte, y_pred, average="macro")

# Baseline (clase mayoritaria)
majority_class = Counter(y_train).most_common(1)[0][0]
y_pred_baseline = [majority_class] * len(y_test)
print("\n=== Baseline (clase mayoritaria) ===")
print(classification_report(y_test, y_pred_baseline, digits=3, zero_division=0))
print("Balanced Acc:", balanced_accuracy_score(y_test, y_pred_baseline))

# ---------- Modelos ----------
resultados = []

# Naive Bayes (BoW)
f1_nb_bow = entrenar_y_evaluar(MultinomialNB(), X_train_bow, y_train, X_test_bow, y_test, "Naïve Bayes (BoW)")
resultados.append(("Naïve Bayes (BoW)", f1_nb_bow))

# Regresión Logística (BoW)
lr_bow = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
f1_lr_bow = entrenar_y_evaluar(lr_bow, X_train_bow, y_train, X_test_bow, y_test, "Regresión Logística (BoW)")
resultados.append(("Regresión Logística (BoW)", f1_lr_bow))

# SVM lineal (BoW)
svm_bow = LinearSVC(class_weight="balanced")
f1_svm_bow = entrenar_y_evaluar(svm_bow, X_train_bow, y_train, X_test_bow, y_test, "SVM Lineal (BoW)")
resultados.append(("SVM (BoW)", f1_svm_bow))

# Naive Bayes (TF-IDF)
f1_nb_tfidf = entrenar_y_evaluar(MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, y_test, "Naïve Bayes (TF-IDF)")
resultados.append(("Naïve Bayes (TF-IDF)", f1_nb_tfidf))

# Regresión Logística (TF-IDF)
lr_tfidf = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
f1_lr_tfidf = entrenar_y_evaluar(lr_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test, "Regresión Logística (TF-IDF)")
resultados.append(("Regresión Logística (TF-IDF)", f1_lr_tfidf))

# SVM lineal (TF-IDF)
svm_tfidf = LinearSVC(class_weight="balanced")
f1_svm_tfidf = entrenar_y_evaluar(svm_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test, "SVM Lineal (TF-IDF)")
resultados.append(("SVM (TF-IDF)", f1_svm_tfidf))

# Comparación
df_res = pd.DataFrame(resultados, columns=["Modelo", "Macro-F1"]).sort_values("Macro-F1", ascending=False)
print("\n=== Comparación de modelos (ordenado por Macro-F1) ===")
print(df_res)

# -----------------------------------------
# 3) (OPCIONAL) GUARDAR VECTORIZADORES/MODELOS
# -----------------------------------------
if SAVE_MODELS:
    import joblib
    joblib.dump(bow_vectorizer,   "bow_vectorizer.joblib")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")
    joblib.dump(lr_tfidf,         "modelo_lr_tfidf.joblib")
    joblib.dump(svm_tfidf,        "modelo_svm_tfidf.joblib")
    joblib.dump(lr_bow,           "modelo_lr_bow.joblib")
    joblib.dump(svm_bow,          "modelo_svm_bow.joblib")
    print("\nVectorizadores y modelos guardados.")