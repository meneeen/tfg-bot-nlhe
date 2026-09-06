"""
Juega contra el agente RL entrenado, en la interfaz gráfica.

Necesita un modelo entrenado. Si no tienes ninguno:
    python -m src.ai.rl.train

Uso:
    python play_vs_rl.py                                   # usa models/rl/modelo.zip
    python play_vs_rl.py --model models/rl_v2/modelo.zip

NOTA: el stack no es configurable a propósito. El modelo se entrenó
normalizando la observación por STARTING_STACK; jugar con otro stack le haría
llegar features en una escala distinta a la que aprendió, y sus decisiones
dejarían de tener sentido. El stack está horneado en el modelo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ai.labels import nombre_rl
from src.engine.controller import STARTING_STACK
from src.gui.main_window import main as run_gui

_DEFAULT_MODEL = "models/rl/modelo.zip"


def main() -> int:
    p = argparse.ArgumentParser(description="Juega contra el agente RL en la GUI.")
    p.add_argument("--model", default=_DEFAULT_MODEL)
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"No existe el modelo: {args.model}")
        print("\nEntrena uno primero:")
        print("    python -m src.ai.rl.train")
        return 1

    from src.ai.rl_agent import RLAgent          # importa SB3: lento, solo si hace falta

    print(f"Cargando modelo: {args.model}")
    agent  = RLAgent(args.model)
    nombre = nombre_rl(args.model)               # mismo formato que en duelo.py
    print(f"Listo. Abriendo la partida contra {nombre}.\n")

    run_gui(agent, agent_name=nombre)
    return 0


if __name__ == "__main__":
    sys.exit(main())
