import joblib
import gradio as gr
from taller_ii_boilerplate.inference.inference import predict_sentiment

vectorizer = joblib.load("../../models/vectorizador_bow.pkl")
model = joblib.load("../../models/modelo_nb_bow.pkl")

def interface(text):
    result = predict_sentiment(text, vectorizer, model)
    return result.get("prediction"), result.get("probabilities")


demo = gr.Interface(
    fn=interface,
    inputs=gr.Textbox(label="", lines=3, placeholder="Escribe una texto aquí..."),
    outputs=[
        gr.Label(label="Sentimiento predicho"),
        gr.JSON(label="Probabilidades del modelo")
    ],
    title="Análisis de sentimientos",
    description="Escribe un texto y el modelo clasificará su sentimiento como positivo, negativo o neutro."
)

demo.launch()
