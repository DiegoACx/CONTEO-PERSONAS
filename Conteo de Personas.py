import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Configuración de la página de Streamlit
st.set_page_config(page_title="Conteo de Personas", layout="wide")

# Cargar el modelo YOLOv8 preentrenado
model_path = "yolov8s.pt"  
model = YOLO(model_path)

# Función para el procesamiento de imágenes y conteo de personas
def process_frame(frame):
    # Detectar objetos en el frame
    results = model(frame)
    detections = results[0]  # Resultados de la primera imagen

    # Filtrar detecciones de personas (clase 0 en COCO dataset)
    person_detections = [det for det in detections if det[5] == 0]

    # Contar personas
    person_count = len(person_detections)

    # Dibujar cajas alrededor de las personas detectadas
    for det in person_detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, person_count

# Configuración de la cámara 
camera = cv2.VideoCapture(0)  

st.title("Conteo de Personas frente al Establecimiento")
st.text("Esta aplicación utiliza el modelo YOLOv8 para detectar y contar personas.")

# Bucle de la aplicación
if st.button("Iniciar conteo"):
    stframe = st.empty()
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Procesar el frame para detección y conteo
        processed_frame, person_count = process_frame(frame)

        # Mostrar el conteo en la interfaz
        st.subheader(f"Conteo de Personas: {person_count}")

        # Convertir frame de BGR a RGB para mostrar en Streamlit
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        stframe.image(frame_pil, channels="RGB", use_column_width=True)
        
camera.release()
st.write("Conteo de personas finalizado")
