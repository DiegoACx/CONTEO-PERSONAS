<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-1.54-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
</p>

<h1 align="center">Conteo de personas con YOLOv8 y ByteTrack</h1>

<p align="center">App de Streamlit que detecta y sigue personas en una cámara o un video, y cuenta los IDs de seguimiento.</p>

<p align="center">🇬🇧 <a href="README.en.md">Read in English</a></p>

## Acerca del proyecto

Detecta personas con YOLOv8s (Ultralytics) y las sigue con ByteTrack, a partir de una cámara web o un archivo de video local. Muestra el video con cada persona enmarcada y su ID, y dos contadores: personas en el fotograma actual e IDs de seguimiento únicos acumulados.

La lógica de conteo (`src/counter.py`) es independiente de Streamlit, de la cámara y de Ultralytics, y tiene 8 tests.

Desarrollado en el curso de *Inteligencia Artificial* (4.º semestre) de la Universidad Autónoma de Bucaramanga (UNAB), en noviembre de 2024, y reescrito en septiembre de 2026.

### Lo que NO hace

- **No guarda fotogramas ni videos.** El código propio no escribe nada en disco.
- **No mide precisión.** No hay datos anotados ni métricas.
- **No es un sistema de aforo validado.** No hay línea ni zona de cruce, y cuenta IDs de seguimiento, no personas únicas.

## Funcionalidades

- **Fuentes:** cámara (índice configurable de 0 a 10, por defecto 0) o archivo de video con una ruta local. El video no se sube ni se copia.
- **Controles:** botones **Iniciar** y **Detener**. Detener funciona porque Streamlit relanza el script, lo que interrumpe el bucle; un bloque `finally` libera la captura.
- **Detección y seguimiento:** `model.track(..., classes=[0], tracker="bytetrack.yaml")`. Solo personas. El seguimiento se reinicia en cada **Iniciar**.
- **Contadores** (en `st.session_state`):
  - *Personas en el fotograma:* detecciones del fotograma actual.
  - *IDs de seguimiento únicos:* IDs distintos acumulados desde el último **Iniciar**.
- **Dibujo:** un rectángulo verde por persona y la etiqueta `ID n` si tiene ID.
- **Errores claros:** ruta vacía o inexistente, archivo que no es video, cámara que no abre, pérdida de señal y modelo que no carga (por ejemplo, sin internet la primera vez).
- **Aviso visible en la app:** "No se guardan fotogramas ni videos. El conteo por ID puede contar dos veces a una persona si el seguimiento la pierde."

## Stack tecnológico

| Paquete | Versión | Para qué se usa |
|---|---|---|
| streamlit | 1.54.0 | Interfaz |
| ultralytics | 8.3.34 | YOLOv8 y ByteTrack |
| opencv-python | 4.10.0.84 | Captura de cámara y video |
| numpy | 2.1.1 | Arreglos |
| pillow | 12.3.0 | Imágenes |
| lapx | 0.10.0 | Requerido por ByteTrack |
| pytest | 9.1.1 | Tests (`requirements-dev.txt`) |

`torch` y `torchvision` los instala Ultralytics y no están fijados; en las pruebas resolvieron a torch 2.14.0 (CPU) y torchvision 0.29.0.

`lapx` está declarado porque, sin él, Ultralytics intenta instalarlo por su cuenta con el `pip` que encuentre, que puede ser el de otro entorno.

**Python probado:** 3.11 en Windows. Las versiones 3.10, 3.12 y 3.13 solo se comprobaron como resolución de dependencias, no en ejecución.

## Estructura del proyecto

```
.
├─ app/streamlit_app.py
├─ src/counter.py
├─ tests/test_counter.py
└─ requirements.txt · requirements-dev.txt · .gitignore
```

`yolov8s.pt`, `.venv/` y `runs/` no se versionan (están en el `.gitignore`).

## Cómo correrlo (Windows, PowerShell)

Requiere Python 3.11 y, para la app, una cámara web o un archivo de video.

```powershell
py -3.11 -m venv .venv
# Si PowerShell bloquea la activación: Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
pytest
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

- La instalación tarda unos minutos por `torch` (4 min 7 s en un venv limpio).
- `pip install --upgrade pip` importa: el `pip` que trae un venv nuevo de Python 3.11.4 (23.1.2) tiene 7 avisos de seguridad conocidos, corregidos desde `pip` 26.2.
- `--server.address 127.0.0.1` limita el acceso a tu equipo. Por defecto Streamlit escucha en todas las interfaces.
- Los tests no necesitan cámara, modelo ni internet.
- **Modelo:** `yolov8s.pt` se descarga solo al pulsar **Iniciar** si no está, desde `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt` (unos 22,6 MB), en la carpeta desde la que lances el comando. Necesita internet la primera vez.

## Qué se probó y qué no

**Automatizado** (venv temporal de Python 3.11.4, versiones actuales de `requirements.txt`):

- 8 tests con detecciones simuladas.
- Arranque headless: `/_stcore/health` responde `ok` y `GET /` responde 200.
- `AppTest` de la app real, sin excepciones.
- `pip check` sin conflictos.
- `pip-audit`: 0 vulnerabilidades en 67 paquetes (PyPI y OSV), a 19/09/2026.

**Prueba manual del autor** (no cubierta por los tests): cámara real y personas, incluido el botón **Detener**, con Streamlit 1.54.0 y Pillow 12.3.0, en Python 3.11 y Windows. Funcionó bien.

**No verificado:**

- Python 3.10, 3.12 y 3.13 en ejecución.
- Linux, macOS y GPU.
- Varios usuarios a la vez.
- Rendimiento medido y precisión del conteo.

## Limitaciones

- **Doble conteo.** Si el seguimiento pierde a alguien (por ejemplo, por oclusión) y le asigna otro ID al reaparecer, se cuenta dos veces. Un test lo documenta.
- **Modelo compartido entre sesiones.** `st.cache_resource` comparte el modelo, y con él el estado del seguimiento. Con varios usuarios a la vez podrían interferir.
- **Sin métricas de precisión.**
- **Aviso de deprecación.** `st.image(use_column_width=True)` sigue funcionando en Streamlit 1.54.0, pero emite un aviso por fotograma y se eliminará en una versión futura.
- **Dependencias transitivas sin fijar** (`torch` y otras): el resultado de `pip-audit` vale para lo resuelto ese día.

## Privacidad

- **No se guarda** ningún fotograma, video ni contador. Todo va en memoria y se pierde al cerrar.
- **Sí queda en disco**, por las dependencias: `yolov8s.pt` en la carpeta de ejecución y el archivo de ajustes de Ultralytics en `%APPDATA%\Ultralytics\settings.json`, que incluye un identificador anónimo (hash).
- **Red:** la app no hace llamadas propias. Ultralytics descarga el modelo la primera vez y, si la telemetría sigue activa, envía eventos anónimos.
- **Telemetría de Ultralytics** (según el código de la versión 8.3.34): viene activa por defecto (`sync: True`) y envía eventos anónimos a Google Analytics (modo, tarea, nombre del modelo, versiones de Python y de Ultralytics, entorno e ID de sesión aleatorio). No envía fotogramas. Para desactivarla:
```
  python -c "from ultralytics import settings; settings.update({'sync': False})"
```
  No se verificó con captura de tráfico que deje de enviar.
- Si la usas con cámara en un espacio con gente, revisa la normativa local de protección de datos.

## Licencia

Este repositorio no tiene un archivo `LICENSE`. Ultralytics se distribuye bajo **AGPL-3.0** (verificado en el paquete instalado), que puede imponer obligaciones al redistribuir o exponer la app en red. Léela antes de publicarla o reutilizarla. La licencia de los pesos `yolov8s.pt` no se encontró en el paquete.

## Historia del proyecto

La primera versión (noviembre de 2024) contaba por proximidad de centroides entre fotogramas y traía una ruta absoluta de otro equipo. En septiembre de 2026 se reescribió con ByteTrack, la lógica de conteo se separó de la interfaz y se redujo `requirements.txt` (era un `pip freeze` de 68 líneas). Los archivos antiguos, incluido el modelo, siguen en el historial de git.

## Autor

- Diego Castro — [@DiegoACx](https://github.com/DiegoACx)

El refactor de 2026 se desarrolló con asistencia de Claude (Anthropic).
