import streamlit as st
from transformers import pipeline
import pandas as pd

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# --- Interfaz ---
st.set_page_config(page_title="Zero-Shot Classifier", layout="centered")
st.title("🌐 Clasificador de Tópicos Zero-Shot")
st.write("Clasifica texto en cualquier categoría sin necesidad de entrenar el modelo.")

# Entrada de texto
text = st.text_area(
    "✍️ Escribe un texto para clasificar:",
    "La inteligencia artificial está cambiando la medicina y la educación."
)

# Entrada de etiquetas
labels_input = st.text_input(
    "📌 Ingresa categorías separadas por comas:",
    "tecnología, política, deporte, salud, educación, economía"
)

# Botón
if st.button("🔍 Clasificar"):
    if text.strip() and labels_input.strip():
        candidate_labels = [lbl.strip() for lbl in labels_input.split(",")]
        result = classifier(text, candidate_labels)

        # Mostrar resultados
        st.subheader("Resultados")
        df = pd.DataFrame({
            "Categoría": result["labels"],
            "Probabilidad": result["scores"]
        })
        st.dataframe(df)

        # Gráfico de barras
        st.bar_chart(df.set_index("Categoría"))
    else:
        st.warning("Por favor ingresa texto y categorías.")
