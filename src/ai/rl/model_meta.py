from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def meta_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    if p.suffix == ".zip":
        p = p.with_suffix("")
    return p.with_suffix(".meta.json")


def save_meta(
    model_path: str | Path,
    starting_stack: int,
    big_blind: int,
    small_blind: int,
    obs_dim: int,
    training_profiles: list[str] | None = None,
) -> None:
    meta = {
        "starting_stack": starting_stack,
        "big_blind"     : big_blind,
        "small_blind"   : small_blind,
        "obs_dim"       : obs_dim,
    }
    # Contra qué perfiles de reglas se entrenó. Opcional: los modelos anteriores
    # a los perfiles no lo tienen, y load_meta no debe exigirlo. Es lo que
    # distingue dos modelos por lo demás idénticos salvo por su sparring.
    if training_profiles is not None:
        meta["training_profiles"] = list(training_profiles)
    meta_path(model_path).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_meta(model_path: str | Path) -> dict[str, Any]:
    path = meta_path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No hay metadatos para el modelo: falta {path.name}.\n"
            f"Es un modelo entrenado antes de que existieran, así que no se sabe\n"
            f"con qué stack ni qué observación se entrenó. Reentrénalo:\n"
            f"    python -m src.ai.rl.train"
        )
    return json.loads(path.read_text(encoding="utf-8"))
