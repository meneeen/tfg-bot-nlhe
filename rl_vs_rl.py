"""
El agente RL contra sí mismo. Deja un log legible mano a mano.

Los dos lados juegan en modo estocástico, y no es opcional: si fueran
deterministas, con el mismo reparto tomarían siempre las mismas decisiones y el
enfrentamiento sería una única partida repetida N veces, sin ninguna
información. El muestreo es lo que hace que cada mano explore líneas distintas.

También sirve para enfrentar dos checkpoints, que es la forma de comprobar si el
self-play mejoró algo de verdad:

    python rl_vs_rl.py --model-b models/rl_v2/modelo.zip

(el modelo A es el final; si gana claramente al de la fase 1, el self-play sirvió)

Uso:
    python rl_vs_rl.py                  # 200 manos, log en logs/rl_vs_rl.log
    python rl_vs_rl.py --hands 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ai.match import play_match, summary

_DEFAULT_MODEL = "models/rl/modelo.zip"


def main() -> int:
    p = argparse.ArgumentParser(description="El agente RL contra sí mismo.")
    p.add_argument("--model-a", default=_DEFAULT_MODEL)
    p.add_argument("--model-b", default=None,
                   help="Segundo modelo. Por defecto, el mismo que A (self-play puro). "
                        "Apunta a otro checkpoint para comparar dos versiones.")
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="logs/rl_vs_rl.log")
    args = p.parse_args()

    model_b = args.model_b or args.model_a
    for m in (args.model_a, model_b):
        if not Path(m).exists():
            print(f"No existe el modelo: {m}")
            print("\nEntrena uno primero:\n    python -m src.ai.rl.train")
            return 1

    from src.ai.rl_agent import RLAgent

    same = (model_b == args.model_a)
    name_a, name_b = ("RL-1", "RL-2") if same else ("RL-A", "RL-B")

    print("Cargando modelos...")
    # deterministic=False siempre: ver el docstring de arriba.
    agent_a = RLAgent(args.model_a, deterministic=False)
    agent_b = RLAgent(model_b,      deterministic=False)
    print(f"  {name_a}: {args.model_a}")
    print(f"  {name_b}: {model_b}")
    print(f"\nJugando {args.hands} manos...\n")

    res = play_match(
        agent_a, agent_b,
        name_a=name_a, name_b=name_b,
        hands=args.hands, seed=args.seed,
        log_path=args.log,
    )

    print("\n" + summary(res))
    print(f"\nLog detallado: {Path(args.log).resolve()}")

    if same:
        print("\n  Nota: el mismo modelo a ambos lados deberia dar ~0 bb/100.\n"
              "  Una desviacion grande indica ventaja posicional no compensada\n"
              "  o simplemente falta de manos (varianza).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
