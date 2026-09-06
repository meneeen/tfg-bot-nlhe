"""
Perfiles de estilo para RuleBasedAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

_NUNCA_CHEN   = 99.0
_SIEMPRE_CHEN = -99.0
_NUNCA_EQUITY = 1.01


@dataclass(frozen=True)
class RuleProfile:

    name: str
    description: str

    # ───────────────────────────── preflop ──────────────────────────────────
    btn_open_raise: float   # Chen mínimo para abrir subiendo desde BTN
    btn_limp: float         # Chen mínimo para entrar pagando (por debajo: fold)
    bb_threebet: float      # Chen mínimo para 3-bet desde BB
    bb_call_raise: float    # Chen mínimo para pagar una subida desde BB

    open_bb_mult: float     # tamaño de apertura, en ciegas grandes
    threebet_mult: float    # tamaño del 3-bet, en ciegas grandes

    # ──────────────────────────── postflop ──────────────────────────────────
    equity_value_bet: float      # equity mínima para apostar de primeras
    equity_raise_vs_bet: float   # equity mínima para re-subir una apuesta
    bet_pot_fraction: float      # tamaño de apuesta como fracción del bote

    semi_bluff_freq: float       # P(apostar) con proyecto y sin mano hecha
    bluff_freq: float            # P(apostar) sin nada en absoluto

    facing_bet_discount: float   # descuento por rango al enfrentar una apuesta
    draw_call_equity: float      # equity implícita atribuida a un proyecto
    call_equity_floor: float     # suelo de equity al pagar (1.0 = paga siempre)

    def __post_init__(self) -> None:
        for campo in ("semi_bluff_freq", "bluff_freq", "facing_bet_discount",
                      "draw_call_equity", "call_equity_floor"):
            valor = getattr(self, campo)
            if not 0.0 <= valor <= 1.0:
                raise ValueError(
                    f"Perfil '{self.name}': {campo}={valor} fuera de [0, 1]."
                )
        if self.bet_pot_fraction <= 0.0:
            raise ValueError(
                f"Perfil '{self.name}': bet_pot_fraction debe ser > 0, "
                f"recibido {self.bet_pot_fraction}."
            )
        for campo in ("open_bb_mult", "threebet_mult"):
            valor = getattr(self, campo)
            if valor <= 1.0:
                raise ValueError(
                    f"Perfil '{self.name}': {campo}={valor} debe ser > 1 "
                    f"(una subida tiene que superar la ciega grande)."
                )

    def variante(self, **cambios) -> "RuleProfile":
        cambios.setdefault("name", f"{self.name}*")
        return replace(self, **cambios)


# ═════════════════════════════ perfiles ══════════════════════════

# Baseline: 
#   - equity_value_bet estaba en 0.65, por encima del trío: el agente hacía
#     check el 82% del tiempo postflop. Un rival tan pasivo no sirve de baseline
#     y le enseña al RL que "el rival nunca apuesta", estrategia que se hunde en
#     self-play.
#   - equity_raise_vs_bet es un umbral SEPARADO y más alto. Sin él, el agente
#     re-subía con cualquier mano que apostaría por valor, y dos de estos
#     agentes entraban en guerras de subidas (72% de re-subidas postflop). Un
#     maníaco es tan mal baseline como un nit, solo que explotable al revés.
#
BASELINE = RuleProfile(
    name        = "baseline",
    description = "Perfil histórico calibrado: tight-passive moderado.",
    btn_open_raise      = 8.0,    # 10.4% de las manos
    btn_limp            = 4.0,    # +39.2% limpeadas
    bb_threebet         = 14.0,   # 1.4%
    bb_call_raise       = 7.0,    # 17.2%
    open_bb_mult        = 3.0,
    threebet_mult       = 6.0,
    equity_value_bet    = 0.55,   # doble par / overpair
    equity_raise_vs_bet = 0.68,   # solo trío o mejor re-sube
    bet_pot_fraction    = 0.50,
    semi_bluff_freq     = 0.30,
    bluff_freq          = 0.06,
    facing_bet_discount = 0.78,
    draw_call_equity    = 0.28,
    call_equity_floor   = 0.0,
)

# Nit: extremo del jugador tight-passive. Solo juega manos premium y solo
# mete fichas con la mano hecha.
NIT = RuleProfile(
    name        = "nit",
    description = "Tight-passive extremo: abre 6.6% y solo sube con trío o mejor.",
    btn_open_raise      = 9.0,    # 6.6%
    btn_limp            = 6.5,    # +11.2% limpeadas
    bb_threebet         = 16.0,   # 0.9% (KK+)
    bb_call_raise       = 9.0,    # 6.6%
    open_bb_mult        = 3.0,
    threebet_mult       = 6.0,
    equity_value_bet    = 0.68,   # trío+
    equity_raise_vs_bet = 0.83,   # color+
    bet_pot_fraction    = 0.50,
    semi_bluff_freq     = 0.05,
    bluff_freq          = 0.0,
    facing_bet_discount = 0.70,   # da mucho crédito a quien apuesta
    draw_call_equity    = 0.20,
    call_equity_floor   = 0.0,
)

# Tight-aggressive: el arquetipo "correcto" en heads-up. Rango de apertura ancho,
# nunca limpea, y postflop apuesta por valor con par decente además de farolear 
# con frecuencia sana.
TAG = RuleProfile(
    name        = "tag",
    description = "Tight-aggressive: abre 76.8% desde BTN, nunca limpea, farolea un 12%.",
    btn_open_raise      = 2.0,    # 76.8%
    btn_limp            = _NUNCA_CHEN,
    bb_threebet         = 8.0,    # 10.4%
    bb_call_raise       = 3.5,    # 51.7% (defiende el 62% del total)
    open_bb_mult        = 2.5,
    threebet_mult       = 7.5,
    equity_value_bet    = 0.50,   # par fuerte+
    equity_raise_vs_bet = 0.68,
    bet_pot_fraction    = 0.60,
    semi_bluff_freq     = 0.45,
    bluff_freq          = 0.12,
    facing_bet_discount = 0.80,
    draw_call_equity    = 0.30,
    call_equity_floor   = 0.0,
)

# Loose-aggressive: juega casi todo y presiona con casi todo. Distinto del
# maníaco en que sigue teniendo umbrales — no sube el 100% de las veces.
LAG = RuleProfile(
    name        = "lag",
    description = "Loose-aggressive: abre 92.8%, 3-betea 24% y farolea un 30%.",
    btn_open_raise      = 0.0,    # 92.8%
    btn_limp            = _NUNCA_CHEN,
    bb_threebet         = 6.0,    # 24.0%
    bb_call_raise       = 0.0,    # 92.8%
    open_bb_mult        = 2.5,
    threebet_mult       = 9.0,
    equity_value_bet    = 0.38,   # cualquier par
    equity_raise_vs_bet = 0.58,   # doble par+
    bet_pot_fraction    = 0.75,
    semi_bluff_freq     = 0.70,
    bluff_freq          = 0.30,
    facing_bet_discount = 0.85,
    draw_call_equity    = 0.35,
    call_equity_floor   = 0.0,
)

# Calling station: juega el 100% de las manos, NUNCA sube y NUNCA se tira.
STATION = RuleProfile(
    name        = "station",
    description = "Calling station: paga el 100% de las manos y nunca sube.",
    btn_open_raise      = _NUNCA_CHEN,
    btn_limp            = _SIEMPRE_CHEN,
    bb_threebet         = _NUNCA_CHEN,
    bb_call_raise       = _SIEMPRE_CHEN,
    open_bb_mult        = 3.0,    # inalcanzable: nunca sube
    threebet_mult       = 6.0,    # inalcanzable: nunca sube
    equity_value_bet    = _NUNCA_EQUITY,
    equity_raise_vs_bet = _NUNCA_EQUITY,
    bet_pot_fraction    = 0.50,   # inalcanzable: nunca apuesta
    semi_bluff_freq     = 0.0,
    bluff_freq          = 0.0,
    facing_bet_discount = 1.0,
    draw_call_equity    = 1.0,
    call_equity_floor   = 1.0,    # paga siempre
)

# Maníaco: sube el 100% preflop y apuesta casi siempre postflop.
MANIAC = RuleProfile(
    name        = "maniac",
    description = "Maníaco: sube el 100% preflop y farolea el 65% postflop.",
    btn_open_raise      = _SIEMPRE_CHEN,
    btn_limp            = _SIEMPRE_CHEN,   # inalcanzable: siempre sube antes
    bb_threebet         = 1.0,    # 86.1%
    bb_call_raise       = _SIEMPRE_CHEN,
    open_bb_mult        = 5.0,
    threebet_mult       = 12.0,
    equity_value_bet    = 0.25,   # carta alta decente ya apuesta
    equity_raise_vs_bet = 0.35,
    bet_pot_fraction    = 1.00,
    semi_bluff_freq     = 0.90,
    bluff_freq          = 0.65,
    facing_bet_discount = 0.95,
    draw_call_equity    = 0.40,
    call_equity_floor   = 0.0,
)


PROFILES: dict[str, RuleProfile] = {
    p.name: p for p in (BASELINE, NIT, TAG, LAG, STATION, MANIAC)
}

DEFAULT_PROFILE = BASELINE


def get_profile(name: str) -> RuleProfile:
    """Busca un perfil por nombre (sin distinguir mayúsculas)."""
    try:
        return PROFILES[name.strip().lower()]
    except KeyError:
        raise KeyError(
            f"Perfil de reglas desconocido: '{name}'. "
            f"Disponibles: {', '.join(sorted(PROFILES))}."
        ) from None
