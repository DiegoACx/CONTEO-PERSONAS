<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-1.54-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
</p>

<h1 align="center">People Counting with YOLOv8 and ByteTrack</h1>

<p align="center">Streamlit app that detects and tracks people in a webcam or a video, and counts tracking IDs.</p>

<p align="center">🇪🇸 <a href="README.md">Leer en español</a></p>

## About

It detects people with YOLOv8s (Ultralytics) and tracks them with ByteTrack, from a webcam or a local video file. It shows the video with each person boxed and labeled with an ID, and two counters: people in the current frame and accumulated unique tracking IDs.

The counting logic (`src/counter.py`) is independent of Streamlit, the camera and Ultralytics, and has 8 tests.

Built in the *Artificial Intelligence* course (4th semester) at Universidad Autónoma de Bucaramanga (UNAB) in November 2024, and rewritten in September 2026.

### What it does NOT do

- **It does not save frames or videos.** The project's own code writes nothing to disk.
- **It does not measure accuracy.** There is no annotated data or metrics.
- **It is not a validated occupancy system.** There is no crossing line or zone, and it counts tracking IDs, not unique people.

## Features

- **Sources:** webcam (configurable index from 0 to 10, default 0) or a video file given as a local path. The video is not uploaded or copied.
- **Controls:** **Start** and **Stop** buttons. Stop works because Streamlit reruns the script, which interrupts the loop; a `finally` block releases the capture.
- **Detection and tracking:** `model.track(..., classes=[0], tracker="bytetrack.yaml")`. People only. Tracking is reset on every **Start**.
- **Counters** (in `st.session_state`):
  - *People in frame:* detections in the current frame.
  - *Unique tracking IDs:* distinct IDs accumulated since the last **Start**.
- **Drawing:** a green rectangle per person and an `ID n` label when it has an ID.
- **Clear errors:** empty or missing path, file that is not a video, camera that fails to open, signal loss, and model that fails to load (for example, no internet on first run).
- **Visible in-app notice:** "No frames or videos are saved. ID counting may count a person twice if tracking loses them."

## Tech stack

| Package | Version | Used for |
|---|---|---|
| streamlit | 1.54.0 | UI |
| ultralytics | 8.3.34 | YOLOv8 and ByteTrack |
| opencv-python | 4.10.0.84 | Camera and video capture |
| numpy | 2.1.1 | Arrays |
| pillow | 12.3.0 | Images |
| lapx | 0.10.0 | Required by ByteTrack |
| pytest | 9.1.1 | Tests (`requirements-dev.txt`) |

`torch` and `torchvision` are installed by Ultralytics and are not pinned; in testing they resolved to torch 2.14.0 (CPU) and torchvision 0.29.0.

`lapx` is declared because, without it, Ultralytics tries to install it on its own with whatever `pip` it finds, which may belong to another environment.

**Tested Python:** 3.11 on Windows. Versions 3.10, 3.12 and 3.13 were only checked as dry-run dependency resolution, not in execution.

## Project structure

```
.
├─ app/streamlit_app.py
├─ src/counter.py
├─ tests/test_counter.py
└─ requirements.txt · requirements-dev.txt · .gitignore
```

`yolov8s.pt`, `.venv/` and `runs/` are not versioned (they are in `.gitignore`).

## How to run (Windows, PowerShell)

Requires Python 3.11 and, for the app, a webcam or a video file.

```powershell
py -3.11 -m venv .venv
# If PowerShell blocks activation: Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
pytest
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

- Installation takes a few minutes because of `torch` (4 min 7 s in a clean venv).
- `pip install --upgrade pip` matters: the `pip` bundled with a fresh Python 3.11.4 venv (23.1.2) has 7 known security advisories, fixed since `pip` 26.2.
- `--server.address 127.0.0.1` restricts access to your machine. By default Streamlit listens on all interfaces.
- Tests need no camera, model or internet.
- **Model:** `yolov8s.pt` downloads automatically when you press **Start** if it is missing, from `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt` (about 22.6 MB), into the folder you launch the command from. It needs internet on first run.

## What was tested and what was not

**Automated** (temporary Python 3.11.4 venv, current `requirements.txt` versions):

- 8 tests with simulated detections.
- Headless startup: `/_stcore/health` returns `ok` and `GET /` returns 200.
- `AppTest` of the real app, with no exceptions.
- `pip check` with no conflicts.
- `pip-audit`: 0 vulnerabilities across 67 packages (PyPI and OSV), as of 2026-09-19.

**Manual test by the author** (not covered by the tests): real camera and real people, including the **Stop** button, with Streamlit 1.54.0 and Pillow 12.3.0, on Python 3.11 and Windows. It worked well.

**Not verified:**

- Python 3.10, 3.12 and 3.13 in execution.
- Linux, macOS and GPU.
- Several users at once.
- Measured performance and counting accuracy.

## Limitations

- **Double counting.** If tracking loses someone (for example, due to occlusion) and assigns a new ID when they reappear, they are counted twice. A test documents this.
- **Model shared across sessions.** `st.cache_resource` shares the model, and with it the tracking state. Several simultaneous users could interfere with each other.
- **No accuracy metrics.**
- **Deprecation warning.** `st.image(use_column_width=True)` still works in Streamlit 1.54.0, but it emits a warning per frame and will be removed in a future version.
- **Unpinned transitive dependencies** (`torch` and others): the `pip-audit` result holds for what resolved that day.

## Privacy

- **Nothing is saved:** no frames, videos or counters. Everything stays in memory and is lost on close.
- **What does end up on disk**, through dependencies: `yolov8s.pt` in the working folder and the Ultralytics settings file at `%APPDATA%\Ultralytics\settings.json`, which includes an anonymous identifier (a hash).
- **Network:** the app makes no calls of its own. Ultralytics downloads the model on first run and, if telemetry is still on, sends anonymous events.
- **Ultralytics telemetry** (per the code of version 8.3.34): it is on by default (`sync: True`) and sends anonymous events to Google Analytics (mode, task, model name, Python and Ultralytics versions, environment and a random session ID). It does not send frames. To turn it off:
```
  python -c "from ultralytics import settings; settings.update({'sync': False})"
```
  It was not verified with a traffic capture that it stops sending.
- If you use it with a camera in a space with people, check your local data protection rules.

## License

This repository has no `LICENSE` file. Ultralytics is distributed under **AGPL-3.0** (verified in the installed package), which may impose obligations when redistributing or exposing the app over a network. Read it before publishing or reusing this. The license of the `yolov8s.pt` weights was not found in the package.

## Project history

The first version (November 2024) counted by centroid proximity between frames and contained an absolute path from another machine. In September 2026 it was rewritten with ByteTrack, the counting logic was separated from the UI, and `requirements.txt` was reduced (it was a 68-line `pip freeze`). The old files, including the model, remain in the git history.

## Author

- Diego Castro — [@DiegoACx](https://github.com/DiegoACx)

The 2026 refactor was developed with assistance from Claude (Anthropic).
