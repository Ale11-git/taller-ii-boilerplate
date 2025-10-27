import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df = pd.read_csv("../../../data/preprocessed/dataset.csv")

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

# Instancio el modelo a optimizar
lr = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial", random_state=42,)

# Definimos los hiperparámetros a optimizar (depende de cada algoritmo)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],           # fuerza de regularización
    'solver': ['lbfgs', 'saga'],       # métodos de optimización
    'penalty': ['l2']                  # tipo de regularización
}

def optimizar_modelo(modelo, hiperparametros, X_train, y_train):
    # Definimos la busqueda en grilla o grid search
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=hiperparametros,
        cv=5,                     # k=5 para validación cruzada
        scoring='f1_macro',       # métrica de evaluación (la que queremos maximizar u optimizar)
        verbose=4
    )
    # Optimizamos el modelo
    grid_search.fit(X_train, y_train)
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    print("Mejor F1 macro promedio:", grid_search.best_score_)


optimizar_modelo(lr, param_grid_lr, X_train_bow, y_train)
optimizar_modelo(LinearSVC(class_weight="balanced"), {
    'penalty': ['l1','l2'],
    'loss': ['hinge', 'squared_hinge'],
    'C': [0.01, 0.1, 1, 10]
}, X_train_bow, y_train)
