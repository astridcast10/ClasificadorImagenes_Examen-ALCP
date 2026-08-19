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

UMBRAL_CONFIANZA = 0.75


@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_clasificador.h5")
    return modelo


def predecir(imagen: Image.Image, modelo):
    if imagen.mode in ("RGBA", "LA", "P"):
        imagen = imagen.convert("RGBA")
        fondo = Image.new("RGB", imagen.size, (255, 255, 255))
        fondo.paste(imagen, mask=imagen.split()[-1])
        img = fondo.resize((32, 32))
    else:
        img = imagen.convert("RGB").resize((32, 32))

    arreglo = np.array(img).astype("float32") / 255.0
    arreglo = np.expand_dims(arreglo, axis=0)

    predicciones = modelo.predict(arreglo, verbose=0)[0]
    indice_clase = int(np.argmax(predicciones))
    confianza = float(predicciones[indice_clase])

    return CLASES[indice_clase], confianza, predicciones


st.title("🔍 Clasificador de Imágenes con IA")
st.caption(f"Proyecto de Computación en la Nube — hecho por **{NOMBRE_AUTOR}**")

st.markdown(
    """
    Subí una foto o tomá una con tu cámara y el modelo va a intentar identificar
    qué objeto aparece. Está entrenado con el dataset **CIFAR-10**, así que reconoce
    10 categorías: avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.
    """
)

st.markdown(
    """
    <div style="background-color:#EAF2FF; border:1px solid #B6D4FE; border-radius:10px; padding:18px 20px; margin-bottom:20px;">
        <h4 style="margin-top:0; color:#0B3D91;">📋 Categorías que el modelo reconoce</h4>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin:12px 0;">
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Avión</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Auto</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Pájaro</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Gato</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Ciervo</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Perro</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Rana</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Caballo</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Barco</span>
            <span style="background:#0B3D91; color:white; padding:6px 14px; border-radius:20px; font-size:14px;">Camión</span>
        </div>
        <p style="margin-bottom:0; color:#8A6D00; background:#FFF3CD; border-left:4px solid #FFC107; padding:10px 14px; border-radius:6px; font-size:14px;">
            ⚠️ El modelo <b>SOLO</b> reconoce estas 10 categorías. Cualquier otra imagen (personas, objetos, paisajes, etc.) va a forzarse dentro de una de ellas.
        </p>
    </div>
    """,
    unsafe_allow_html=True
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
        if confianza >= UMBRAL_CONFIANZA:
            st.metric(label="Predicción", value=clase.capitalize())
            st.metric(label="Confianza", value=f"{confianza:.2f}")
            st.progress(min(confianza, 1.0))
        else:
            st.metric(label="Mejor coincidencia", value=clase.capitalize())
            st.metric(label="Confianza", value=f"{confianza:.2f}")

    if confianza < UMBRAL_CONFIANZA:
        st.markdown("---")
        st.error(
            f"""
### 🚫 Clase no reconocida

La confianza del modelo ({confianza:.2f}) está por debajo del umbral mínimo ({UMBRAL_CONFIANZA:.2f}).

**Esto significa que la imagen probablemente NO corresponde a ninguna de las 10 categorías** que el modelo aprendió a reconocer (avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco, camión).

En vez de forzar una respuesta poco confiable, la app te avisa para que no la tomes como válida.
            """
        )

    with st.expander("Ver todas las probabilidades"):
        for nombre_clase, prob in sorted(
            zip(CLASES, todas_probs), key=lambda x: x[1], reverse=True
        ):
            st.write(f"{nombre_clase.capitalize()}: {prob:.2f}")
else:
    st.info("Subí una imagen o tomá una foto para empezar.")

st.markdown("---")
st.caption("Modelo CNN entrenado en Google Colab con el dataset CIFAR-10 · TensorFlow/Keras + Streamlit")
