"""
Utilidad de checkpoints del agente RL.

Crea un checkpoint "sin entrenar" (PPO con pesos aleatorios) para poder usarlo
como suelo de comparación en los duelos, p.ej.:

    from src.ai.rl.checkpoints import asegurar_sin_entrenar
    asegurar_sin_entrenar(Path("models/rl"))
    # -> models/rl/sin_entrenar.zip  (cargable con duelo.py rl:...)
"""
from __future__ import annotations

from pathlib import Path

from src.engine.controller import GameController


def asegurar_sin_entrenar(models_dir: Path) -> Path:
    """Crea el checkpoint cero (PPO sin entrenar) si no existe, con su .meta.json
    para que RLAgent pueda cargarlo. Devuelve la ruta (sin extensión).

    Idempotente: si ya existe sin_entrenar.zip, no lo regenera.
    """
    models_dir = Path(models_dir)
    path = models_dir / "sin_entrenar"
    if (models_dir / "sin_entrenar.zip").exists():
        return path

    # Import perezoso: crear el modelo necesita torch/SB3, pero solo la 1ª vez.
    print("Creando checkpoint cero (PPO sin entrenar)...", flush=True)
    from stable_baselines3 import PPO
    from src.ai.rl.poker_env import PokerEnv
    from src.ai.rl.model_meta import save_meta

    models_dir.mkdir(parents=True, exist_ok=True)
    model = PPO("MlpPolicy", PokerEnv())   # sin .learn(): pesos aleatorios
    model.save(str(path))
    save_meta(
        str(path),
        starting_stack = PokerEnv.STARTING_STACK,
        big_blind      = GameController.BIG_BLIND,
        small_blind    = GameController.SMALL_BLIND,
        obs_dim        = PokerEnv.OBS_DIM,
    )
    return path
