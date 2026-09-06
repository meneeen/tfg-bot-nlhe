"""
Juega contra el LLM (Ollama) en la interfaz gráfica.

Requiere Ollama corriendo (ollama serve) y el modelo descargado
(ollama pull llama3.1).

Uso:
    python play_vs_llm.py                          # llama3.1, temperature 0.7
    python play_vs_llm.py --model llama3.2:3b      # más rápido, juega peor
    python play_vs_llm.py --temperature 0.0        # decisiones casi deterministas
    python play_vs_llm.py --no-reasoning           # ablation: sin razonamiento
"""
from __future__ import annotations

import argparse
import sys
import time

from src.ai.llm_agent import LLMAgent
from src.ai.rule_based_agent import RuleBasedAgent
from src.engine.controller import STARTING_STACK
from src.gui.main_window import main as run_gui


def _warmup(agent: LLMAgent) -> bool:
    """Fuerza la carga del modelo en Ollama. True si respondió de verdad."""
    from treys import Card

    from src.ai.base_agent import GameState, Position
    from src.engine.controller import GameController, Street

    sb, bb = GameController.SMALL_BLIND, GameController.BIG_BLIND
    state = GameState(
        hole_cards=(Card.new("As"), Card.new("Ks")),
        community_cards=(),
        pot=sb + bb,
        my_stack=STARTING_STACK - sb,
        opp_stack=STARTING_STACK - bb,
        my_bet_this_street=sb,
        to_call=bb - sb,
        min_raise_to=2 * bb,
        max_raise_to=STARTING_STACK,
        position=Position.BTN,
        street=Street.PREFLOP,
        action_history=(),
        big_blind=bb,
        small_blind=sb,
    )
    t0 = time.perf_counter()
    agent.decide_action(state)
    dt = time.perf_counter() - t0
    ok = agent.n_fallbacks == 0
    print(f"  carga en {dt:.1f} s", end="")
    print("" if ok else "   [!] el modelo NO respondio (fallback)")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Juega contra el LLM en la GUI.")
    p.add_argument("--model", default="llama3.1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-reasoning", action="store_true",
                   help="Ablation: no pedirle al modelo que razone.")
    args = p.parse_args()

    agent = LLMAgent(
        model=args.model,
        temperature=args.temperature,
        include_reasoning=not args.no_reasoning,
        # Sin esto, un timeout de red haria que el LLM se retirase.
        fallback_agent=RuleBasedAgent(seed=0),
    )

    print(f"Precalentando '{args.model}' (puede tardar minutos la 1a vez)...")
    if not _warmup(agent):
        print("\n  Ollama no responde. Comprueba que está corriendo (ollama serve).")
        return 1

    s = agent.stats()
    print(f"  modelo confirmado por Ollama: {s['server_model']}")
    print(f"\nAbriendo la partida. El rival tarda ~5 s por decision:")

    # Contadores a cero: el precalentado no cuenta como partida.
    agent.n_calls = agent.n_fallbacks = agent.n_retries = 0
    agent.n_parse_failures = agent.n_invalid_actions = agent.n_output_tokens = 0
    agent._total_latency = 0.0

    try:
        # El modelo concreto forma parte del rival: 'LLM' a secas no distingue
        # una partida contra llama3.1 de una contra mistral.
        run_gui(agent, agent_name=f"LLM ({args.model})")
    finally:
        # SystemExit de pygame: aprovechamos para volcar las metricas.
        s = agent.stats()
        if s["n_calls"]:
            print("\n--- LLM en esta partida ---")
            print(f"  decisiones          : {s['n_calls']}")
            print(f"  latencia media      : {s['avg_latency_s']:.2f} s")
            print(f"  tasa de malformadas : {s['malformed_rate']:.1%}")
            print(f"  reintentos          : {s['n_retries']}")
            print(f"  fallbacks           : {s['n_fallbacks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
