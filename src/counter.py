"""Logica de conteo, independiente de Streamlit, de la camara y de Ultralytics.

Recibe, por cada fotograma, un elemento por persona detectada: su ID de seguimiento, o None si
esa deteccion no tiene ID. Cuenta:

* `present`: personas detectadas en ese fotograma (con o sin ID).
* `unique`: IDs de seguimiento distintos acumulados desde el ultimo `reset()`.

Limitacion: si el seguimiento pierde a una persona y le asigna otro ID al reaparecer, se cuenta
dos veces; el conteo por ID no identifica personas, solo IDs.
"""
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class FrameCount:
    present: int
    unique: int


class TrackingCounter:
    def __init__(self) -> None:
        self._seen: set[int] = set()

    def update(self, track_ids: Iterable[Optional[float]]) -> FrameCount:
        """Registra un fotograma. `track_ids` tiene un elemento por persona detectada (None = sin ID)."""
        ids = list(track_ids)
        self._seen.update(int(i) for i in ids if i is not None)
        return FrameCount(present=len(ids), unique=len(self._seen))

    @property
    def unique(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        self._seen.clear()
