"""
El agente RL contra el LLM. Deja un log legible mano a mano.

Necesita:
  - Un modelo RL entrenado:  python -m src.ai.rl.train
  - Ollama corriendo (ollama serve) con el modelo descargado.

AVISO DE TIEMPO: el LLM tarda ~5 s por decisión y toma ~2 decisiones por mano,
así que 200 manos son ~35 min y 1.000 manos unas 3 h. El RL responde en
milisegundos: el LLM es el cuello de botella. El script imprime el ETA.

Uso:
    python rl_vs_llm.py                  # 100 manos, log en logs/rl_vs_llm.log
    python rl_vs_llm.py --hands 500
    python rl_vs_llm.py --no-reasoning   # ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ai.llm_agent import LLMAgent
from src.ai.match import play_match, summary
from src.ai.rule_based_agent import RuleBasedAgent
from src.engine.controller import STARTING_STACK

_DEFAULT_MODEL = "models/rl/modelo.zip"


def main() -> int:
    p = argparse.ArgumentParser(description="El agente RL contra el LLM.")
    p.add_argument("--rl-model", default=_DEFAULT_MODEL)
    p.add_argument("--llm-model", default="llama3.1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-reasoning", action="store_true",
                   help="Ablation: no pedirle al LLM que razone.")
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="logs/rl_vs_llm.log")
    args = p.parse_args()

    if not Path(args.rl_model).exists():
        print(f"No existe el modelo RL: {args.rl_model}")
        print("\nEntrena uno primero:\n    python -m src.ai.rl.train")
        return 1

    from src.ai.rl_agent import RLAgent

    rl = RLAgent(args.rl_model, deterministic=False)
    llm = LLMAgent(
        model=args.llm_model,
        temperature=args.temperature,
        include_reasoning=not args.no_reasoning,
        # Sin agente de reserva, un timeout de red haría que el LLM se retirase:
        # perderia manos por fallos de infraestructura, no por jugar mal, y eso
        # contaminaria la medicion de su fuerza.
        fallback_agent=RuleBasedAgent(seed=0),
    )

    print(f"RL : {args.rl_model}")
    print(f"LLM: {args.llm_model}  (temperature={args.temperature}, "
          f"reasoning={not args.no_reasoning})")

    # La primera llamada carga el modelo en Ollama y puede tardar minutos.
    # Se hace ahora para que no contamine la medicion de latencia de la partida.
    print("\nPrecalentando el LLM (la carga en frio puede tardar minutos)...")
    from src.ai.base_agent import GameState, Position
    from src.engine.controller import GameController, Street
    from treys import Card
    sb, bb = GameController.SMALL_BLIND, GameController.BIG_BLIND
    warm = GameState(
        hole_cards=(Card.new("As"), Card.new("Ks")), community_cards=(),
        pot=sb + bb, my_stack=STARTING_STACK - sb, opp_stack=STARTING_STACK - bb,
        my_bet_this_street=sb, to_call=bb - sb, min_raise_to=2 * bb,
        max_raise_to=STARTING_STACK, position=Position.BTN, street=Street.PREFLOP,
        action_history=(), big_blind=bb, small_blind=sb,
    )
    llm.decide_action(warm)
    if llm.n_fallbacks:
        print("\n  El LLM no responde. Comprueba que Ollama está corriendo (ollama serve).")
        return 1
    print(f"  modelo confirmado por Ollama: {llm.stats()['server_model']}")

    # Contadores a cero: el precalentado no forma parte de la partida.
    llm.n_calls = llm.n_fallbacks = llm.n_retries = 0
    llm.n_parse_failures = llm.n_invalid_actions = llm.n_output_tokens = 0
    llm._total_latency = 0.0

    print(f"\nJugando {args.hands} manos (el LLM es el cuello de botella)...\n")
    res = play_match(
        rl, llm,
        name_a="RL", name_b="LLM",
        hands=args.hands, seed=args.seed,
        log_path=args.log,
    )

    print("\n" + summary(res))

    # Fiabilidad del LLM: si hubo muchos fallbacks, el resultado mide su red,
    # no su juego.
    s = llm.stats()
    print("\n--- Fiabilidad del LLM ---")
    print(f"  modelo confirmado    : {s['server_model']}")
    print(f"  decisiones           : {s['n_calls']}")
    print(f"  latencia media       : {s['avg_latency_s']:.2f} s")
    print(f"  tasa de malformadas  : {s['malformed_rate']:.1%}")
    print(f"  reintentos           : {s['n_retries']}")
    print(f"  fallbacks            : {s['n_fallbacks']}")
    if s["n_calls"] and s["n_fallbacks"] / s["n_calls"] > 0.05:
        print("\n  AVISO: mas del 5 % de decisiones fueron del agente de reserva.\n"
              "  El resultado mide la fiabilidad del LLM, no su juego.")

    print(f"\nLog detallado: {Path(args.log).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
