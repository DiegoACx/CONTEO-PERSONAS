import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Conteo de Personas", layout="centered")

model_path = r"C:\Users\santi\OneDrive\Escritorio\CONTEO-PERSONAS-main\CONTEO-PERSONAS-main\yolov8s.pt"
model = YOLO(model_path)

# Variable para almacenar el contador total de personas
total_person_count = 0
# Variable para mantener los centroides de las personas detectadas
previous_centroids = []

def get_centroid(x1, y1, x2, y2):
    # Calcula el centro de la caja delimitadora
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def distance(p1, p2):
    # Calcula la distancia entre dos puntos (p1 y p2 son centroides)
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def process_frame(frame):
    global total_person_count, previous_centroids
    results = model(frame)  # Se obtiene el resultado del modelo para la imagen actual
    detections = results[0].boxes  # Accedemos a las cajas de las detecciones

    current_centroids = []

    # Iteramos sobre cada detección
    for det in detections:
        # Det tiene la forma (x1, y1, x2, y2, conf, class)
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Coordenadas del cuadro delimitador
        conf = det.conf[0]  # Confianza de la detección
        cls = int(det.cls[0])  # Clase de la detección (0 para personas)
        
        # Si la clase es 0 (persona), procesamos la detección
        if cls == 0:
            centroid = get_centroid(x1, y1, x2, y2)
            current_centroids.append(centroid)

            # Verificamos si el centroide es nuevo o si ya se había detectado previamente
            is_new_person = True
            for prev_centroid in previous_centroids:
                if distance(centroid, prev_centroid) < 50:  # Umbral de proximidad de 50 píxeles
                    is_new_person = False
                    break

            if is_new_person:
                total_person_count += 1  # Incrementamos el contador solo si es una persona nueva

    # Actualizamos los centroides previos
    previous_centroids = current_centroids

    # Dibujar rectángulos alrededor de las personas detectadas
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Coordenadas de la caja delimitadora
        # Dibujar el rectángulo alrededor de la persona detectada
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Rectángulo verde

    return frame

camera = cv2.VideoCapture(0)

st.title("Conteo de Personas")
st.text("Esta aplicación utiliza YOLOv8 para detectar y contar personas de manera precisa.")

if st.button("Iniciar conteo"):
    stframe = st.empty()
    count_display = st.empty()
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("No se pudo acceder a la cámara.")
            break

        frame = process_frame(frame)

        # Mostrar el conteo total de personas
        count_display.subheader(f"Conteo de Personas: {total_person_count}")

        # Redimensionamos la imagen
        frame = cv2.resize(frame, (640, 480)) 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        stframe.image(frame_pil, channels="RGB", use_column_width=False)

camera.release()
st.write("Conteo de Personas Finalizado")


# cd "C:\Users\santi\OneDrive\Escritorio\CONTEO-PERSONAS-main\CONTEO-PERSONAS-main"
# streamlit run "Conteo de Personas Streamlit.py"