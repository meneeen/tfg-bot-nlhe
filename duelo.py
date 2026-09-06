from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.ai.base_agent import PokerAgent
from src.ai.labels import nombre_reglas, nombre_rl
from src.ai.match import (
    MatchResult, SessionMatchResult, play_many_sessions, play_match,
    session_summary,
)
from src.ai.random_agent import RandomAgent
from src.ai.rule_based_agent import RuleBasedAgent
from src.ai.rule_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from src.engine.controller import GameController, STARTING_STACK

_JSON_OUT   = Path("datos/duelos.json")
_CURVAS_DIR = Path("datos/curvas")

# Modo partidas: salida separada de la del modo manos. Mezclarlas en el mismo
# JSON no funcionaría — un duelo de manos y una tanda de partidas no comparten
# ni métrica (bb/100 vs partidas ganadas) ni unidad (mano vs partida).
_PARTIDAS_JSON = Path("datos/partidas.json")

_AYUDA_AGENTE = (
    "random | reglas | reglas:<perfil> | llm | rl:<ruta.zip>   "
    f"(perfiles: {', '.join(sorted(PROFILES))})"
)

# ───────────────────────────── construir agentes ────────────────────────────

def construir(spec: str, seed: int) -> tuple[PokerAgent, str]:
    spec = spec.strip()

    if spec == "random":
        return RandomAgent(seed=seed), "Aleatorio"

    # "reglas" (perfil por defecto) y "reglas:<perfil>". El nombre SIEMPRE lleva
    # el perfil entre paréntesis, incluso el default: en datos/duelos.json una
    # fila "Reglas" a secas no dice contra qué estilo se jugó, y dos duelos con
    # rivales distintos acaban indistinguibles en la tabla de resultados.
    if spec == "reglas" or spec.startswith("reglas:"):
        nombre_perfil = spec[len("reglas:"):] if ":" in spec else DEFAULT_PROFILE.name
        try:
            perfil = get_profile(nombre_perfil)
        except KeyError as exc:
            raise SystemExit(str(exc)) from None
        return RuleBasedAgent(perfil, seed=seed), nombre_reglas(perfil.name)

    if spec == "llm":
        from src.ai.llm_agent import LLMAgent
        return LLMAgent(fallback_agent=RuleBasedAgent(seed=seed)), "LLM"

    if spec.startswith("rl:"):
        from src.ai.rl_agent import RLAgent
        ruta = spec[3:]
        if not Path(ruta).exists():
            raise SystemExit(f"No existe el modelo: {ruta}")
        return RLAgent(ruta), nombre_rl(ruta)

    raise SystemExit(
        f"Agente desconocido: '{spec}'\n"
        f"Usa: random | reglas | reglas:<perfil> | llm | rl:<ruta.zip>\n"
        f"Perfiles de reglas: {', '.join(sorted(PROFILES))}"
    )


# ──────────────────────────────── informe ───────────────────────────────────

def _fmt(valor, formato: str, si_none: str = "n/d") -> str:
    return si_none if valor is None else format(valor, formato)


def informe(res: MatchResult, seed: int) -> str:
    a, b = res.name_a, res.name_b
    marca = "" if res.significant else "   (NO significativo)"

    bb_b = -res.bb_per_100 or 0.0

    if res.net_chips_a > 0:
        veredicto = f"gana {a} (+{res.net_chips_a:,})".replace(",", ".")
    elif res.net_chips_a < 0:
        veredicto = f"gana {b} (+{-res.net_chips_a:,})".replace(",", ".")
    else:
        veredicto = "empate (0 fichas)"

    lineas = [
        "",
        "=" * 66,
        f"  {a}  vs  {b}",
        f"  {res.hands:,} manos  |  semilla {seed}  |  "
        f"{res.elapsed_s / 60:.1f} min".replace(",", "."),
        "=" * 66,
        "",
        "  RENDIMIENTO",
        f"    bb/100            : {a} {res.bb_per_100:+.2f}  |  {b} {bb_b:+.2f}"
        f"   +/-{res.ci95:.2f}{marca}",
        f"    Fichas netas      : {veredicto}",
        f"    Manos ganadas     : {a} {res.wins_a:,} ({res.wins_a/res.hands:.1%})  |  "
        f"{b} {res.wins_b:,} ({res.wins_b/res.hands:.1%})  |  "
        f"empates {res.ties:,}".replace(",", "."),
        "",
        "  ESTILO",
        f"    AF postflop       : {a} {_fmt(res.af_a, '.2f')}  |  "
        f"{b} {_fmt(res.af_b, '.2f')}",
        f"    WTSD (compartido) : {_fmt(res.wtsd, '.1%')}   "
        f"({res.n_showdowns:,} showdowns de {res.n_saw_flop:,} manos "
        f"que vieron flop)".replace(",", "."),
        "",
    ]

    if not res.significant:
        lineas += [
            "  AVISO: El intervalo de confianza cruza el cero: con estas manos no",
            "    puedes afirmar que un agente sea mejor que el otro.",
        ]

    lineas.append("=" * 66)
    return "\n".join(lineas)


# ──────────────────────────────── salida JSON ───────────────────────────────

def a_dict(res: MatchResult, seed: int, n_duelo: int = 0) -> dict:
    d = {
        "n_duelo": n_duelo,
        "a": res.name_a,
        "b": res.name_b,
        "manos": res.hands,
        "seed": seed,

        "bb100_a": round(res.bb_per_100, 3),
        "bb100_b": round(-res.bb_per_100 or 0.0, 3),
        "fichas_netas_a": res.net_chips_a,

        "manos_ganadas_a": res.wins_a,
        "manos_ganadas_b": res.wins_b,
        "empates": res.ties,

        "analisis": _analisis(res),
    }
    return d


def _analisis(res: MatchResult) -> dict:
    return {
        "af_a"       : None if res.af_a is None else round(res.af_a, 3),
        "af_b"       : None if res.af_b is None else round(res.af_b, 3),
        "n_showdowns": res.n_showdowns,
        "n_flop"     : res.n_saw_flop,
        "wtsd"       : None if res.wtsd is None else round(res.wtsd, 4),
        "n_preflop"  : res.hands - res.n_saw_flop,
        "preflop_a"  : res.wins_preflop_a,
        "preflop_b"  : res.wins_preflop_b,
    }


def escribir_curva(res: MatchResult, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    acum = 0.0
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mano", "delta_bb_a", "acum_bb_a", "acum_bb_b", "vio_flop"])
        for i, (d, vio) in enumerate(zip(res.deltas_bb, res.saw_flop_flags), start=1):
            acum += d
            w.writerow([i, f"{d:.3f}", f"{acum:.3f}", f"{-acum:.3f}", int(vio)])


def a_dict_partidas(res: SessionMatchResult, n: int = 0) -> dict:
    """Solo los datos crudos de la tanda.

    Nada de win rate ni IC: se derivan de ganadas_a / partidas. Guardar un valor
    derivado solo crea la posibilidad de que contradiga al que lo genera.
    """
    return {
        "n_tanda"   : n,
        "a"         : res.name_a,
        "b"         : res.name_b,
        "partidas"  : res.n_sessions,
        "seed"      : res.seed,
        "ganadas_a" : res.wins_a,
        "ganadas_b" : res.wins_b,
        "empates"   : res.draws,
        "af_a"      : None if res.af_a is None else round(res.af_a, 3),
        "af_b"      : None if res.af_b is None else round(res.af_b, 3),
        "wtsd"      : None if res.wtsd is None else round(res.wtsd, 4),
    }


def _cargar_duelos(out: Path) -> list[dict]:
    if not out.exists():
        return []
    texto = out.read_text(encoding="utf-8").strip()
    if not texto:
        return []
    previo = json.loads(texto)
    return previo if isinstance(previo, list) else [previo]


def _guardar_duelos(lista: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(lista, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ──────────────────────────────────── CLI ───────────────────────────────────

def _modo_partidas(args, agente_a, agente_b, nombre_a, nombre_b) -> int:
    """Rama --modo partidas: tanda de partidas al KO en vez de manos sueltas."""
    if nombre_a == nombre_b:
        print(f"\n  ERROR: los dos agentes se llaman '{nombre_a}'. En modo "
              f"partidas el marcador se atribuye por nombre, así que tienen "
              f"que ser distinguibles.")
        return 1

    print(f"\n{nombre_a}  vs  {nombre_b}   |   {args.sessions} partidas al KO"
          f"   |   semilla {args.seed}")
    print(f"stack {STARTING_STACK} ({STARTING_STACK // GameController.BIG_BLIND} "
          f"ciegas)  |  ciegas {GameController.SMALL_BLIND}/"
          f"{GameController.BIG_BLIND}  |  tope {args.max_hands} manos/partida\n")

    res = play_many_sessions(
        agente_a, agente_b, nombre_a, nombre_b,
        n_sessions=args.sessions,
        seed=args.seed,
        max_hands=args.max_hands,
    )
    print(session_summary(res))

    lista = _cargar_duelos(_PARTIDAS_JSON)
    n     = len(lista) + 1
    lista.append(a_dict_partidas(res, n))
    _guardar_duelos(lista, _PARTIDAS_JSON)

    print(f"\n  Tanda #{n} registrada:")
    print(f"    {_PARTIDAS_JSON}  (append)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("agente_a", help=_AYUDA_AGENTE)
    p.add_argument("agente_b", help=_AYUDA_AGENTE)
    p.add_argument("--hands", "-n", type=int, default=1000,
                   help="Número de manos (default: 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla del reparto (default: 42)")
    p.add_argument("--log", type=str, default=None,
                   help="(Opcional) Volcar el log mano a mano a este fichero de texto")
    p.add_argument("--modo", choices=("manos", "partidas"), default="manos",
                   help="'manos': N manos independientes reseteando stacks, "
                        "mide bb/100 (default). 'partidas': partidas al KO con "
                        "stacks arrastrados, mide partidas ganadas.")
    p.add_argument("--sessions", type=int, default=500,
                   help="Solo con --modo partidas: cuántas partidas jugar. Se "
                        "redondea al par siguiente porque van en parejas espejo "
                        "(default: 500)")
    p.add_argument("--max-hands", type=int, default=200,
                   help="Solo con --modo partidas: tope de manos por partida; "
                        "al alcanzarlo la partida cuenta como empate "
                        "(default: 200)")
    args = p.parse_args()

    agente_a, nombre_a = construir(args.agente_a, args.seed)
    agente_b, nombre_b = construir(args.agente_b, args.seed + 1)

    if args.modo == "partidas":
        return _modo_partidas(args, agente_a, agente_b, nombre_a, nombre_b)

    print(f"\n{nombre_a}  vs  {nombre_b}   |   {args.hands:,} manos   |   "
          f"semilla {args.seed}".replace(",", "."))
    print(f"stack {STARTING_STACK}  |  ciegas "
          f"{GameController.SMALL_BLIND}/{GameController.BIG_BLIND}\n")

    res = play_match(
        agente_a, agente_b, nombre_a, nombre_b,
        hands=args.hands,
        seed=args.seed,
        log_path=args.log,
    )

    if res.net_chips_a + res.net_chips_b != 0:
        print(f"\n  ERROR - FUGA DE FICHAS: {nombre_a} {res.net_chips_a:+} y "
              f"{nombre_b} {res.net_chips_b:+} no suman cero.")
        print("    Los resultados de este duelo NO son válidos.")
        return 1

    print(informe(res, args.seed))

    # Salida automática: append al JSON + CSV con el mismo n_duelo correlativo.
    lista   = _cargar_duelos(_JSON_OUT)
    n_duelo = len(lista) + 1
    lista.append(a_dict(res, args.seed, n_duelo))
    _guardar_duelos(lista, _JSON_OUT)

    curva = _CURVAS_DIR / f"duelo{n_duelo}.csv"
    escribir_curva(res, curva)

    print(f"\n  Duelo #{n_duelo} registrado:")
    print(f"    {_JSON_OUT}  (append)")
    print(f"    {curva}  ({res.hands:,} filas)".replace(",", "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())