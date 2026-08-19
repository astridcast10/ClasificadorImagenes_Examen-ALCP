# Documentación - Clasificador de Imágenes con CNN

**Examen — Computación en la Nube (UTH)**
**Docente:** Ing. Asalia Zavala
**Autor:** Astrid *(completar apellido)*

## ¿Qué hace la app?

Es una aplicación web que usa un modelo de Machine Learning (una red neuronal
convolucional, CNN) para identificar qué objeto aparece en una foto. El usuario
puede subir una imagen desde su computadora o tomar una foto directo con la
cámara, y la app muestra la predicción junto con el porcentaje de confianza.

El modelo reconoce 10 categorías, tomadas del dataset **CIFAR-10**:
avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

## ¿Cómo se hizo?

1. **Entrenamiento (Google Colab):** se cargó el dataset CIFAR-10 (60,000
   imágenes de 32x32 píxeles), se normalizaron los datos y se entrenó una CNN
   con varias capas convolucionales, batch normalization y dropout para evitar
   sobreajuste. Se aplicó data augmentation (rotaciones, volteos, zoom) para
   mejorar la generalización. El modelo se guardó en formato `.h5`.

2. **Aplicación (Streamlit):** se construyó una interfaz que carga el modelo
   entrenado, permite subir o capturar una imagen, la redimensiona a 32x32
   píxeles (mismo formato que el entrenamiento), y muestra la clase predicha
   junto con el nivel de confianza y el detalle de probabilidades por clase.

3. **Despliegue:** la app se publicó en Streamlit Community Cloud, conectando
   el repositorio de GitHub que contiene `app.py`, `requirements.txt` y el
   modelo `modelo_clasificador.h5`.

## ¿Cómo usarla?

1. Entrar a la URL pública de la app.
2. Elegir la pestaña "Subir imagen" o "Tomar foto".
3. Cargar o capturar una imagen.
4. Ver la predicción y el porcentaje de confianza en pantalla.

## Archivos entregados

- `entrenamiento_modelo_cifar10.ipynb` — notebook de Colab con todo el proceso
  de entrenamiento.
- `app.py` — código de la aplicación Streamlit.
- `requirements.txt` — dependencias para el despliegue.
- `modelo_clasificador.h5` — modelo entrenado (se genera al correr el notebook).
- URL de la aplicación desplegada.
