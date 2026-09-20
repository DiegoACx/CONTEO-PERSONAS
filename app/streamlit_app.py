"""Conteo de personas con YOLOv8 y seguimiento ByteTrack (Streamlit).

Uso:
    streamlit run app/streamlit_app.py

Fuente de video: una camara (indice configurable) o un archivo de video local. No se guardan
fotogramas ni videos. La logica de conteo esta en src/counter.py.
"""
import sys
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from counter import TrackingCounter  # noqa: E402

MODEL_NAME = "yolov8s.pt"  # Ultralytics lo descarga solo si no existe
AVISO = ("No se guardan fotogramas ni videos. El conteo por ID puede contar dos veces "
         "a una persona si el seguimiento la pierde.")


@st.cache_resource
def cargar_modelo() -> YOLO:
    return YOLO(MODEL_NAME)


def reiniciar_seguimiento(modelo: YOLO) -> None:
    """Con persist=True el seguimiento conserva su estado entre llamadas; y el modelo esta cacheado.
    Se reinicia al iniciar para que cada conteo empiece de cero."""
    predictor = getattr(modelo, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def abrir_captura(fuente: str, indice: int, ruta: str):
    """Devuelve (captura, None) o (None, mensaje de error)."""
    if fuente == "Cámara":
        captura = cv2.VideoCapture(int(indice))
        if not captura.isOpened():
            captura.release()
            return None, (f"No se pudo abrir la cámara con índice {int(indice)}. Comprueba que esté conectada, "
                          "que ninguna otra aplicación la esté usando y que el sistema permita el acceso, "
                          "o prueba otro índice.")
        return captura, None
    ruta = ruta.strip().strip('"')
    if not ruta:
        return None, "Escribe la ruta de un archivo de video."
    if not Path(ruta).is_file():
        return None, f"No existe el archivo: {ruta}"
    captura = cv2.VideoCapture(ruta)
    if not captura.isOpened():
        captura.release()
        return None, "OpenCV no pudo abrir ese archivo como video (formato o códec no soportado)."
    return captura, None


def extraer_personas(resultado):
    """Cajas (x1, y1, x2, y2) e IDs (None si no hay ID) de las personas del fotograma."""
    cajas = resultado.boxes
    n = len(cajas)
    if n == 0:
        return [], []
    xyxy = cajas.xyxy.cpu().numpy().astype(int).tolist()
    ids = cajas.id.int().cpu().tolist() if cajas.id is not None else [None] * n
    return xyxy, ids


def dibujar(frame, xyxy, ids) -> None:
    for (x1, y1, x2, y2), track_id in zip(xyxy, ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if track_id is not None:
            cv2.putText(frame, f"ID {track_id}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)


def main() -> None:
    st.set_page_config(page_title="Conteo de personas", layout="centered")
    st.title("Conteo de personas")
    st.warning(AVISO)

    if "contador" not in st.session_state:
        st.session_state.contador = TrackingCounter()
        st.session_state.presentes = 0
        st.session_state.unicos = 0

    fuente = st.radio("Fuente de video", ["Cámara", "Archivo de video"], horizontal=True, key="fuente")
    indice, ruta = 0, ""
    if fuente == "Cámara":
        indice = st.number_input("Índice de la cámara", min_value=0, max_value=10, value=0, step=1,
                                 key="indice_camara")
    else:
        ruta = st.text_input("Ruta del archivo de video (local)", key="ruta_video")

    col_iniciar, col_detener = st.columns(2)
    iniciar = col_iniciar.button("Iniciar", key="iniciar", type="primary", use_container_width=True)
    detener = col_detener.button("Detener", key="detener", use_container_width=True)

    col_presentes, col_unicos = st.columns(2)
    ph_presentes, ph_unicos = col_presentes.empty(), col_unicos.empty()
    ph_video = st.empty()

    def mostrar_contadores() -> None:
        ph_presentes.metric("Personas en el fotograma", st.session_state.presentes)
        ph_unicos.metric("IDs de seguimiento únicos", st.session_state.unicos)

    mostrar_contadores()

    # Pulsar "Detener" provoca una nueva ejecucion del script, que interrumpe el bucle de captura
    # (su bloque finally libera la captura). Los contadores quedan en st.session_state.
    if detener:
        st.info("Conteo detenido.")

    if not iniciar:
        return

    captura, error = abrir_captura(fuente, indice, ruta)
    if error:
        st.error(error)
        return

    try:
        with st.spinner("Cargando el modelo..."):
            modelo = cargar_modelo()
    except Exception as exc:  # p. ej., sin internet la primera vez que se descarga el modelo
        captura.release()
        st.error(f"No se pudo cargar el modelo {MODEL_NAME}: {exc}")
        return

    contador: TrackingCounter = st.session_state.contador
    contador.reset()
    st.session_state.presentes = 0
    st.session_state.unicos = 0
    reiniciar_seguimiento(modelo)

    leyo_todo = False
    try:
        while captura.isOpened():
            ok, frame = captura.read()
            if not ok:
                leyo_todo = True
                break
            resultado = modelo.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)[0]
            xyxy, ids = extraer_personas(resultado)
            conteo = contador.update(ids)
            st.session_state.presentes = conteo.present
            st.session_state.unicos = conteo.unique
            dibujar(frame, xyxy, ids)
            ph_video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            mostrar_contadores()
    finally:
        captura.release()

    if leyo_todo:
        if fuente == "Archivo de video":
            st.info("Fin del video.")
        else:
            st.error("Se perdió la señal de la cámara: no se pudo leer un fotograma.")


main()
