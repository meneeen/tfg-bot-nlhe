"""
Tests para el sistema de perfiles de RuleBasedAgent.

Cobertura:
1. Registro e integridad  — nombres, lookup, errores claros.
2. Validación            — un perfil mal construido falla al construirse.
3. Inmutabilidad         — variante() no contamina el perfil original.
4. Cableado en el agente — profile por nombre, por objeto y overrides sueltos.
5. Comportamiento        — cada perfil hace lo que promete su descripción.
6. Compatibilidad        — el default sigue siendo el baseline histórico.

Ejecutar con:
    pytest tests/ai/test_rule_profiles.py -v
"""
import dataclasses

import pytest
from treys import Card

from src.ai.base_agent import ActionType, Position
from src.ai.rule_based_agent import RuleBasedAgent
from src.ai.rule_profiles import (
    BASELINE, DEFAULT_PROFILE, LAG, MANIAC, NIT, PROFILES, STATION, TAG,
    RuleProfile, get_profile,
)
from src.engine.controller import Street

from tests.ai.test_rule_based_agent import assert_valid, make_state

# Manos de referencia para los tests de comportamiento.
BASURA   = (Card.new("7d"), Card.new("2c"))   # 72o, Chen ≈ -1.5: la peor mano
MEDIA    = (Card.new("9s"), Card.new("7h"))   # 97o, Chen ≈ 3.5
PREMIUM  = (Card.new("As"), Card.new("Ah"))   # AA,  Chen = 20

TODOS = [NIT, BASELINE, TAG, LAG, STATION, MANIAC]


# ────────────────────────── 1. registro e integridad ─────────────────────────

def test_la_clave_del_registro_coincide_con_el_nombre():
    for clave, perfil in PROFILES.items():
        assert clave == perfil.name


def test_todos_los_perfiles_estan_registrados():
    assert {p.name for p in TODOS} == set(PROFILES)


def test_get_profile_ignora_mayusculas_y_espacios():
    assert get_profile("  TAG ") is TAG


def test_get_profile_desconocido_lista_los_disponibles():
    with pytest.raises(KeyError) as exc:
        get_profile("marciano")
    mensaje = str(exc.value)
    assert "marciano" in mensaje
    for nombre in PROFILES:
        assert nombre in mensaje


def test_cada_perfil_tiene_descripcion():
    for perfil in TODOS:
        assert perfil.description.strip()


# ───────────────────────────── 2. validación ─────────────────────────────────

@pytest.mark.parametrize("campo", [
    "semi_bluff_freq", "bluff_freq", "facing_bet_discount",
    "draw_call_equity", "call_equity_floor",
])
def test_frecuencia_fuera_de_rango_falla(campo):
    with pytest.raises(ValueError, match=campo):
        BASELINE.variante(**{campo: 1.5})


def test_bet_pot_fraction_cero_falla():
    # Apostar el 0 % del bote no es una apuesta: sería un RAISE de importe nulo
    # que el motor rechazaría a mitad de partida en vez de al construirlo.
    with pytest.raises(ValueError, match="bet_pot_fraction"):
        BASELINE.variante(bet_pot_fraction=0.0)


def test_multiplicador_de_subida_menor_que_la_ciega_falla():
    with pytest.raises(ValueError, match="open_bb_mult"):
        BASELINE.variante(open_bb_mult=0.5)


def test_el_mensaje_de_error_identifica_el_perfil():
    with pytest.raises(ValueError, match="nit"):
        NIT.variante(name="nit", bluff_freq=99.0)


# ──────────────────────────── 3. inmutabilidad ───────────────────────────────

def test_el_perfil_es_inmutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        BASELINE.bluff_freq = 0.9


def test_variante_no_modifica_el_original():
    antes = BASELINE.bluff_freq
    v = BASELINE.variante(bluff_freq=0.9)
    assert v.bluff_freq == 0.9
    assert BASELINE.bluff_freq == antes


def test_variante_marca_el_nombre():
    # Sin la marca, una tabla de resultados tendría dos filas "tag" con números
    # distintos y no habría forma de saber cuál es cuál.
    assert TAG.variante(bluff_freq=0.0).name != TAG.name


def test_variante_respeta_un_nombre_explicito():
    assert TAG.variante(name="tag_sin_farol", bluff_freq=0.0).name == "tag_sin_farol"


# ─────────────────────── 4. cableado en el agente ────────────────────────────

def test_default_es_el_baseline_historico():
    # Los duelos ya registrados y los modelos entrenados contra "reglas" asumen
    # estos números: cambiarlos invalidaría las comparaciones publicadas.
    assert RuleBasedAgent().profile is BASELINE
    assert DEFAULT_PROFILE is BASELINE


def test_perfil_por_nombre_y_por_objeto_son_equivalentes():
    assert RuleBasedAgent("lag").profile is RuleBasedAgent(LAG).profile is LAG


def test_perfil_desconocido_falla_al_construir():
    with pytest.raises(KeyError):
        RuleBasedAgent("marciano")


def test_overrides_sueltos_siguen_funcionando():
    # Firma histórica, usada por los tests que necesitan un agente determinista.
    a = RuleBasedAgent(bluff_freq=0.0, semi_bluff_freq=0.0)
    assert a.bluff_freq == 0.0
    assert a.semi_bluff_freq == 0.0
    assert BASELINE.bluff_freq > 0.0        # el perfil original, intacto


def test_overrides_se_aplican_sobre_el_perfil_elegido():
    a = RuleBasedAgent("maniac", bluff_freq=0.0)
    assert a.bluff_freq == 0.0
    assert a.profile.equity_value_bet == MANIAC.equity_value_bet


def test_repr_identifica_el_perfil():
    assert "maniac" in repr(RuleBasedAgent("maniac"))


# ─────────────────────── 5. comportamiento por perfil ────────────────────────

def _accion_btn(perfil, hole, **kw):
    agente = RuleBasedAgent(perfil, seed=0)
    estado = make_state(hole_cards=hole, position=Position.BTN, to_call=10, **kw)
    accion = agente.decide_action(estado)
    assert_valid(accion, estado)
    return accion.action_type


_SUBIDA = (ActionType.RAISE, ActionType.ALL_IN)


def test_station_nunca_sube_preflop():
    for hole in (BASURA, MEDIA, PREMIUM):
        assert _accion_btn(STATION, hole) == ActionType.CALL


def test_station_nunca_se_tira_ante_una_apuesta():
    # Es el perfil que castiga al agente que solo gana faroleando.
    agente = RuleBasedAgent(STATION, seed=0)
    estado = make_state(
        hole_cards=BASURA,
        community_cards=(Card.new("Ks"), Card.new("Qd"), Card.new("Jh")),
        street=Street.FLOP, to_call=500, pot=100, min_raise_to=1000,
    )
    assert agente.decide_action(estado).action_type == ActionType.CALL


def test_station_nunca_apuesta_postflop():
    agente = RuleBasedAgent(STATION, seed=0)
    estado = make_state(
        hole_cards=PREMIUM,
        community_cards=(Card.new("Ad"), Card.new("Ac"), Card.new("Jh")),
        street=Street.FLOP, to_call=0,
    )
    # Póker de ases servido y aun así pasa: la station no sube nunca.
    assert agente.decide_action(estado).action_type == ActionType.CHECK


def test_maniac_sube_incluso_la_peor_mano():
    assert _accion_btn(MANIAC, BASURA) in _SUBIDA


def test_nit_se_tira_una_mano_media_que_el_lag_sube():
    assert _accion_btn(NIT, MEDIA) == ActionType.FOLD
    assert _accion_btn(LAG, MEDIA) in _SUBIDA


def test_todos_los_perfiles_suben_con_ases():
    for perfil in (NIT, BASELINE, TAG, LAG, MANIAC):
        assert _accion_btn(perfil, PREMIUM) in _SUBIDA, perfil.name


def test_tag_y_lag_nunca_limpean():
    # Entrar pagando desde el botón regala la iniciativa sin cerrar el bote:
    # es precisamente el error que estos dos perfiles NO deben cometer.
    for perfil in (TAG, LAG):
        for hole in (BASURA, MEDIA, PREMIUM):
            assert _accion_btn(perfil, hole) != ActionType.CALL, perfil.name


def test_los_rangos_de_apertura_estan_ordenados():
    # nit ⊂ baseline ⊂ tag ⊂ lag ⊂ maniac. Si este orden se rompe, la batería
    # deja de cubrir el eje tight↔loose y los duelos dejan de ser interpretables.
    orden = [NIT, BASELINE, TAG, LAG, MANIAC]
    umbrales = [p.btn_open_raise for p in orden]
    assert umbrales == sorted(umbrales, reverse=True), umbrales


def test_la_agresion_postflop_esta_ordenada():
    orden = [STATION, NIT, BASELINE, TAG, LAG, MANIAC]
    for suave, fuerte in zip(orden, orden[1:]):
        assert suave.equity_value_bet >= fuerte.equity_value_bet, (
            f"{suave.name} debería apostar menos que {fuerte.name}"
        )
        assert suave.bluff_freq <= fuerte.bluff_freq, (
            f"{suave.name} debería farolear menos que {fuerte.name}"
        )


def test_resubir_nunca_es_mas_barato_que_apostar():
    # Si re-subir exigiera menos que apostar de primeras, dos agentes iguales
    # entrarían en guerras de subidas (el fallo documentado en el baseline).
    for perfil in TODOS:
        assert perfil.equity_raise_vs_bet >= perfil.equity_value_bet, perfil.name


# ────────────────────── 6. invariantes en todos los perfiles ─────────────────

@pytest.mark.parametrize("perfil", TODOS, ids=lambda p: p.name)
@pytest.mark.parametrize("street", [Street.PREFLOP, Street.FLOP, Street.RIVER])
def test_ningun_perfil_produce_acciones_invalidas(perfil, street):
    board = {
        Street.PREFLOP: (),
        Street.FLOP: (Card.new("Ks"), Card.new("Qd"), Card.new("7h")),
        Street.RIVER: (Card.new("Ks"), Card.new("Qd"), Card.new("7h"),
                       Card.new("2c"), Card.new("9s")),
    }[street]

    agente = RuleBasedAgent(perfil, seed=3)
    for hole in (BASURA, MEDIA, PREMIUM):
        for to_call in (0, 20, 500):
            for posicion in (Position.BTN, Position.BB):
                estado = make_state(
                    hole_cards=hole, community_cards=board, street=street,
                    to_call=to_call, position=posicion,
                )
                assert_valid(agente.decide_action(estado), estado)


@pytest.mark.parametrize("perfil", TODOS, ids=lambda p: p.name)
def test_ningun_perfil_se_rompe_sin_fichas_para_subir(perfil):
    # min_raise_to > max_raise_to: no hay subida legal posible.
    agente = RuleBasedAgent(perfil, seed=3)
    estado = make_state(
        hole_cards=PREMIUM, to_call=15, my_stack=15,
        min_raise_to=1000, max_raise_to=15,
    )
    assert_valid(agente.decide_action(estado), estado)


@pytest.mark.parametrize("perfil", TODOS, ids=lambda p: p.name)
def test_mismo_seed_mismo_juego(perfil):
    estado = make_state(
        hole_cards=MEDIA,
        community_cards=(Card.new("Ks"), Card.new("Qd"), Card.new("7h")),
        street=Street.FLOP,
    )
    a1 = [RuleBasedAgent(perfil, seed=7).decide_action(estado) for _ in range(5)]
    a2 = [RuleBasedAgent(perfil, seed=7).decide_action(estado) for _ in range(5)]
    assert a1 == a2
