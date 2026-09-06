"""
Tests para los nombres legibles de agentes (src/ai/labels.py).

El punto de estos tests es que el nombre sea EL MISMO en duelo.py y en la GUI:
si la tabla de resultados y la ventana llaman distinto al mismo modelo, no se
puede cruzar lo que juegas con lo que mides.

Ejecutar con:
    pytest tests/ai/test_labels.py -v
"""
import pytest

from src.ai.labels import etiqueta_rl, nombre_reglas, nombre_rl


@pytest.mark.parametrize("ruta, esperado", [
    ("models/hybrid/rl_2m/modelo.zip",    "hybrid/rl_2m"),
    ("models/self-play/rl_2m/modelo.zip", "self-play/rl_2m"),
    ("models/rl/modelo.zip",              "rl"),
    ("models/rl_2m.zip",                  "rl_2m"),      # sin subcarpeta
    ("otro/sitio/pepe.zip",               "otro/sitio/pepe"),  # fuera de models/
    ("modelo.zip",                        "modelo"),     # no se queda vacío
])
def test_etiqueta_rl(ruta, esperado):
    assert etiqueta_rl(ruta) == esperado


def test_etiqueta_rl_acepta_path():
    from pathlib import Path
    assert etiqueta_rl(Path("models/hybrid/rl_2m/modelo.zip")) == "hybrid/rl_2m"


def test_etiqueta_rl_normaliza_el_separador_nativo():
    # La ruta llega del sistema de ficheros (con \ en Windows), pero el nombre
    # mostrado usa / en cualquier plataforma: si no, la misma tabla de duelos
    # tendría claves distintas según dónde se generase.
    from pathlib import Path
    nativa = Path("models") / "hybrid" / "rl_2m" / "modelo.zip"
    assert etiqueta_rl(nativa) == "hybrid/rl_2m"


def test_nombres_completos():
    assert nombre_rl("models/hybrid/rl_2m/modelo.zip") == "RL (hybrid/rl_2m)"
    assert nombre_reglas("maniac") == "Reglas (maniac)"


def test_duelo_y_la_gui_usan_el_mismo_nombre():
    import duelo
    import play_vs_rl

    # duelo.construir() cargaría el modelo (lento y puede no existir); basta con
    # comprobar que ambos módulos comparten la MISMA función, no una copia.
    assert duelo.nombre_rl is play_vs_rl.nombre_rl is nombre_rl
