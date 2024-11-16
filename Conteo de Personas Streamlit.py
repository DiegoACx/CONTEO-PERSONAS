import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Conteo de Personas", layout="centered")

model_path = r"C:\Users\pc\OneDrive\UNIVERSIDAD\CUARTO SEMESTRE\Inteligencia Artificial\Proyecto 2 Corte\yolov8s.pt"
model = YOLO(model_path)

total_person_count = 0

def process_frame(frame):
    global total_person_count  
    results = model(frame)
    detections = results[0]

    person_detections = [det for det in detections if det[5] == 0]

    total_person_count += len(person_detections)

    for det in person_detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

camera = cv2.VideoCapture(0)

st.title("Conteo de Personas")
st.text("Esta aplicación utiliza YOLOv8 para detectar y contar personas de manera acumulativa.")

if st.button("Iniciar conteo"):
    stframe = st.empty()
    count_display = st.empty()
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("No se pudo acceder a la cámara.")
            break

        frame = process_frame(frame)

        count_display.subheader(f"Conteo acumulado de Personas: {total_person_count}")

        frame = cv2.resize(frame, (640, 480)) 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        stframe.image(frame_pil, channels="RGB", use_column_width=False)

camera.release()
st.write("Conteo de Personas Finalizado")