"""
App - Clasificador de Imágenes con CNN (CIFAR-10)
Examen - Computación en la Nube | UTH
Autor: Astrid  
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Clasificador de Imágenes - CIFAR10",
    page_icon="🔍",
    layout="centered"
)

CLASES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo',
          'perro', 'rana', 'caballo', 'barco', 'camión']

NOMBRE_AUTOR = "Astrid"  

UMBRAL_CONFIANZA = 0.50 

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_clasificador.h5")
    return modelo


def predecir(imagen: Image.Image, modelo):
    """Recibe una imagen PIL, la prepara y devuelve (clase, confianza, todas las probabilidades)."""
    img = imagen.convert("RGB").resize((32, 32))
    arreglo = np.array(img).astype("float32") / 255.0
    arreglo = np.expand_dims(arreglo, axis=0)

    predicciones = modelo.predict(arreglo, verbose=0)[0]
    indice_clase = int(np.argmax(predicciones))
    confianza = float(predicciones[indice_clase])

    return CLASES[indice_clase], confianza, predicciones


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
st.title("🔍 Clasificador de Imágenes con IA")
st.caption(f"Proyecto de Computación en la Nube — hecho por **{NOMBRE_AUTOR}**")

st.markdown(
    """
    Subí una foto o tomá una con tu cámara y el modelo va a intentar identificar
    qué objeto aparece. Está entrenado con el dataset **CIFAR-10**, así que reconoce
    10 categorías: avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.
    """
)

with st.expander("📋 Ver las 10 clases que el modelo reconoce"):
    columnas = st.columns(5)
    for i, nombre_clase in enumerate(CLASES):
        columnas[i % 5].markdown(f"- {nombre_clase.capitalize()}")
    st.caption(
        "⚠️ El modelo SOLO reconoce estas 10 categorías. Cualquier otra imagen "
        "(personas, objetos, paisajes, etc.) va a forzarse dentro de una de ellas."
    )

modelo = cargar_modelo()

tab_subir, tab_camara = st.tabs(["📁 Subir imagen", "📷 Tomar foto"])

imagen_entrada = None

with tab_subir:
    archivo = st.file_uploader("Elegí una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        imagen_entrada = Image.open(archivo)

with tab_camara:
    foto = st.camera_input("Tomá una foto")
    if foto is not None:
        imagen_entrada = Image.open(foto)

if imagen_entrada is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(imagen_entrada, caption="Imagen de entrada")

    with st.spinner("Analizando imagen..."):
        clase, confianza, todas_probs = predecir(imagen_entrada, modelo)

    with col2:
        st.subheader("Resultado")

        if confianza < UMBRAL_CONFIANZA:
            st.warning(
                "⚠️ No se pudo identificar con certeza. Es probable que la imagen "
                "no corresponda a ninguna de las 10 categorías conocidas por el modelo "
                f"(mejor coincidencia: **{clase}**, {confianza*100:.1f}% de confianza)."
            )
        else:
            st.metric(label="Predicción", value=clase.capitalize())
            st.metric(label="Confianza", value=f"{confianza*100:.1f}%")
            st.progress(min(confianza, 1.0))

    with st.expander("Ver todas las probabilidades"):
        for nombre_clase, prob in sorted(
            zip(CLASES, todas_probs), key=lambda x: x[1], reverse=True
        ):
            st.write(f"{nombre_clase.capitalize()}: {prob*100:.2f}%")
else:
    st.info("Subí una imagen o tomá una foto para empezar.")

st.markdown("---")
st.caption("Modelo CNN entrenado en Google Colab con el dataset CIFAR-10 · TensorFlow/Keras + Streamlit")
