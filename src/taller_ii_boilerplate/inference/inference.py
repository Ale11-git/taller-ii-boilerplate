import joblib
from taller_ii_boilerplate.preprocessing.procesamiento import preprocess_text
import numpy as np

def predict_sentiment(text, vectorizer, model):
    cleaned_text = preprocess_text(text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    cleaned_probabilities = {str(k): v for k, v in dict(zip(model.classes_, np.round(probabilities, 3))).items()}
    return {"text": text, "prediction": prediction, "probabilities": cleaned_probabilities}

if __name__ == "__main__":
    vectorizer = joblib.load("../../../models/vectorizador_bow.pkl")
    model = joblib.load("../../../models/modelo_nb_bow.pkl")
    example = "Quick & Friendly Customer Service"
    result = predict_sentiment(example, vectorizer, model)
    print(result)
