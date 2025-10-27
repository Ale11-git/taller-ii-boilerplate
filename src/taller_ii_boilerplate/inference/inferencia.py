import joblib
from taller_ii_boilerplate.preprocessing.procesamiento import preprocess_text
import numpy as np

vectorizer = joblib.load("../../../models/vectorizador_bow.pkl")
model = joblib.load("../../../models/modelo_nb_bow.pkl")

def predecir_sentimiento(texto):
    limpio = preprocess_text(texto)
    vector = vectorizer.transform([limpio])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    return {"texto": texto, "prediccion": pred, "probabilidades": dict(zip(model.classes_, np.round(proba, 3)))}

ejemplo = "Great experience!"
resultado = predecir_sentimiento(ejemplo)
print(resultado)
