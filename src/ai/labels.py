"""
Nombres legibles de agentes, compartidos por el arnés de duelos y la GUI.

Vive aquí y no en duelo.py para que un mismo modelo se llame igual en los dos
sitios: si la tabla de resultados dice "RL (hybrid/rl_2m)" y la ventana dice
otra cosa, no hay forma de cruzar lo que juegas con lo que mides.
"""

from __future__ import annotations

from pathlib import Path


def etiqueta_rl(ruta: str | Path) -> str:
    """models/hybrid/rl_2m/modelo.zip -> 'hybrid/rl_2m'.

    Recorta lo que no distingue a un modelo de otro: el prefijo models/ y el
    nombre de fichero 'modelo', que todos comparten.
    """
    partes = list(Path(ruta).with_suffix("").parts)
    if "models" in partes:
        partes = partes[partes.index("models") + 1:]
    if len(partes) > 1 and partes[-1] == "modelo":
        partes = partes[:-1]
    return "/".join(partes) or Path(ruta).stem


def nombre_rl(ruta: str | Path) -> str:
    """Nombre completo para mostrar: 'RL (hybrid/rl_2m)'."""
    return f"RL ({etiqueta_rl(ruta)})"


def nombre_reglas(perfil: str) -> str:
    """Nombre completo para mostrar: 'Reglas (maniac)'."""
    return f"Reglas ({perfil})"
