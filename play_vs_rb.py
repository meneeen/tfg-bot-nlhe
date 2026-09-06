"""
Juega contra el agente de reglas (RuleBasedAgent), en la interfaz gráfica.

Es el rival por defecto y el único que no necesita nada más: ni Ollama, ni un
modelo entrenado. Chen preflop, equity y pot odds postflop, con farol y
semi-farol.

El estilo se elige con --profile: la lógica de decisión es la misma, cambian los
umbrales (ver src/ai/rule_profiles.py).

Uso:
    python play_vs_rb.py                    # perfil baseline
    python play_vs_rb.py --profile maniac
    python play_vs_rb.py --list             # ver los perfiles disponibles
"""
from __future__ import annotations

import argparse
import sys

from src.ai.labels import nombre_reglas
from src.ai.rule_based_agent import RuleBasedAgent
from src.ai.rule_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from src.gui.main_window import main as run_gui


def main() -> int:
    p = argparse.ArgumentParser(
        description="Juega contra el agente de reglas en la GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--profile", "-p", default=DEFAULT_PROFILE.name, metavar="PERFIL",
        help=f"Estilo del rival. Disponibles: {', '.join(sorted(PROFILES))}.",
    )
    p.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="Semilla del rival. Fíjala para que sus faroles sean reproducibles.",
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="Lista los perfiles con su descripción y termina.",
    )
    args = p.parse_args()

    if args.list:
        ancho = max(len(n) for n in PROFILES)
        for nombre in sorted(PROFILES):
            print(f"  {nombre:<{ancho}}  {PROFILES[nombre].description}")
        return 0

    try:
        perfil = get_profile(args.profile)
    except KeyError as exc:
        # str() de un KeyError añade comillas alrededor del mensaje; args[0] no.
        print(exc.args[0])
        return 1

    print(f"Rival: {nombre_reglas(perfil.name)} — {perfil.description}\n")
    run_gui(RuleBasedAgent(perfil, seed=args.seed),
            agent_name=nombre_reglas(perfil.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
