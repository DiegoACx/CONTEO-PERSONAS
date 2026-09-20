"""Pruebas de src/counter.py con detecciones simuladas (sin camara, sin modelo, sin Streamlit)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counter import FrameCount, TrackingCounter  # noqa: E402


def test_frame_vacio_da_cero():
    contador = TrackingCounter()
    assert contador.update([]) == FrameCount(present=0, unique=0)
    assert contador.unique == 0


def test_ids_nuevos_suman():
    contador = TrackingCounter()
    assert contador.update([1, 2]) == FrameCount(present=2, unique=2)
    assert contador.update([2, 3]) == FrameCount(present=2, unique=3)


def test_ids_repetidos_no_suman():
    contador = TrackingCounter()
    contador.update([1, 2])
    resultado = contador.update([1, 2])
    assert resultado == FrameCount(present=2, unique=2)
    assert contador.update([2, 1, 2]).unique == 2  # tambien dentro del mismo fotograma


def test_un_frame_vacio_no_borra_lo_acumulado():
    contador = TrackingCounter()
    contador.update([1, 2, 3])
    assert contador.update([]) == FrameCount(present=0, unique=3)


def test_detecciones_sin_id_cuentan_como_presentes_pero_no_como_unicas():
    contador = TrackingCounter()
    assert contador.update([None, None]) == FrameCount(present=2, unique=0)
    assert contador.update([5, None]) == FrameCount(present=2, unique=1)


def test_acepta_ids_numpy_y_flotantes_como_los_que_devuelve_el_tensor():
    contador = TrackingCounter()
    contador.update(np.array([1.0, 2.0]))
    assert contador.update([np.int64(2), 3.0]).unique == 3


def test_reset_reinicia_los_ids_unicos():
    contador = TrackingCounter()
    contador.update([1, 2])
    contador.reset()
    assert contador.unique == 0
    assert contador.update([1]) == FrameCount(present=1, unique=1)


def test_limitacion_documentada_un_id_nuevo_para_la_misma_persona_se_cuenta_dos_veces():
    """Si el seguimiento pierde a alguien y le da otro ID al reaparecer, el conteo por ID suma 2."""
    contador = TrackingCounter()
    contador.update([1])   # la persona aparece con el ID 1
    contador.update([])    # el seguimiento la pierde
    assert contador.update([2]).unique == 2  # reaparece como ID 2: se cuenta otra vez
