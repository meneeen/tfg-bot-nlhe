"""
Tests de duelo.py.

Cobertura:
  - construir() devuelve el agente correcto (y su nombre) para cada spec.
  - construir() falla con mensaje claro ante un spec desconocido o una ruta de
    modelo inexistente.
  - a_dict() produce un dict serializable a JSON sin infinitos ni NaN, incluido
    el caso af=None (denominador de calls postflop = 0).

Ejecutar con:
    pytest tests/test_duelo.py -v
"""
import json
from pathlib import Path

import pytest

from duelo import construir, a_dict
from src.ai.match import MatchResult
from src.ai.random_agent import RandomAgent
from src.ai.rule_based_agent import RuleBasedAgent
from src.ai.rule_profiles import BASELINE, PROFILES, TAG

# Modelo real para el caso rl:<ruta>. Si no está (checkout sin modelos), se
# salta el test en vez de fallar.
_RL_MODEL = (Path(__file__).resolve().parents[1]
             / "models" / "hybrid" / "rl_2m" / "modelo.zip")


# ─────────────────────────────── construir() ────────────────────────────────

def test_construir_random():
    agente, nombre = construir("random", seed=1)
    assert isinstance(agente, RandomAgent)
    assert nombre == "Aleatorio"


def test_construir_reglas():
    # "reglas" a secas sigue siendo el perfil histórico (los duelos ya
    # registrados se jugaron contra ESE rival), pero el nombre lo hace
    # explícito: una fila "Reglas" sin perfil no dice contra qué se jugó.
    agente, nombre = construir("reglas", seed=1)
    assert isinstance(agente, RuleBasedAgent)
    assert agente.profile is BASELINE
    assert nombre == "Reglas (baseline)"


def test_construir_reglas_con_perfil():
    agente, nombre = construir("reglas:tag", seed=1)
    assert isinstance(agente, RuleBasedAgent)
    assert agente.profile is TAG
    assert nombre == "Reglas (tag)"


@pytest.mark.parametrize("perfil", sorted(PROFILES))
def test_construir_todos_los_perfiles(perfil):
    agente, nombre = construir(f"reglas:{perfil}", seed=1)
    assert agente.profile.name == perfil
    assert perfil in nombre


def test_construir_perfil_desconocido_falla_claro():
    with pytest.raises(SystemExit) as exc:
        construir("reglas:marciano", seed=1)
    mensaje = str(exc.value)
    assert "marciano" in mensaje
    assert "tag" in mensaje          # lista los disponibles


def test_construir_llm_no_contacta_ollama():
    # Construir el LLM solo configura el agente; no hace ninguna llamada de red.
    agente, nombre = construir("llm", seed=1)
    assert type(agente).__name__ == "LLMAgent"
    assert nombre == "LLM"


@pytest.mark.skipif(not _RL_MODEL.exists(), reason="no hay modelo RL entrenado")
def test_construir_rl():
    agente, nombre = construir(f"rl:{_RL_MODEL}", seed=1)
    assert type(agente).__name__ == "RLAgent"
    assert nombre == "RL (hybrid/rl_2m)"


def test_construir_spec_desconocido_falla_claro():
    with pytest.raises(SystemExit) as exc:
        construir("marciano", seed=1)
    assert "desconocido" in str(exc.value).lower()


def test_construir_rl_ruta_inexistente_falla_claro():
    with pytest.raises(SystemExit) as exc:
        construir("rl:no/existe/modelo.zip", seed=1)
    assert "no existe" in str(exc.value).lower()


def test_construir_no_importa_torch_para_agentes_ligeros():
    # Un duelo que no usa el RL no debe cargar stable_baselines3 / torch. Se
    # comprueba en un SUBPROCESO: dentro de la misma sesión de pytest otro test
    # ya podría haber importado torch, así que sys.modules del proceso actual no
    # sirve. Un proceso limpio aísla el efecto de construir().
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys;"
        "from duelo import construir;"
        "construir('reglas', 1); construir('random', 1);"
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"construir() cargó torch para agentes ligeros:\n{r.stderr}"
    )


# ─────────────────────────────── a_dict() ───────────────────────────────────

def _match_con_af_none() -> MatchResult:
    """MatchResult donde af_a es None (0 calls postflop de A) y af_b definido."""
    res = MatchResult("A", "B", hands=10, big_blind=50)
    res.net_chips_a = 100
    res.net_chips_b = -100
    res.deltas_bb = [2.0, -1.0, 0.0, 3.0, -2.0, 1.0, 0.0, 0.5, -0.5, 1.0]
    res.wins_a, res.wins_b, res.ties = 5, 4, 1
    res.n_saw_flop, res.n_showdowns = 8, 5
    res.raises_postflop_a = 3          # calls_a = 0  -> af_a None
    res.raises_postflop_b, res.calls_postflop_b = 2, 4   # af_b = 0.5
    return res


def test_a_dict_serializable_con_af_none():
    d = a_dict(_match_con_af_none(), seed=42)
    an = d["analisis"]

    assert an["af_a"] is None
    assert an["af_b"] == 0.5
    assert an["wtsd"] == pytest.approx(0.625)
    # El objeto analisis lleva sus claves en el orden acordado.
    assert list(an) == ["af_a", "af_b", "n_showdowns", "n_flop",
                        "wtsd", "n_preflop", "preflop_a", "preflop_b"]

    # bb/100 de ambos (b = -a) y sin los campos que se quitaron del --out.
    assert d["bb100_a"] == pytest.approx(20.0)
    assert d["bb100_b"] == pytest.approx(-20.0)
    # Nada de esto va ya en el nivel superior.
    for k in ("bb100", "ic", "significativo", "starting_stack", "big_blind",
              "fecha", "segundos", "af_a", "wtsd", "n_showdowns", "manos_preflop"):
        assert k not in d

    # allow_nan=False hace que json.dumps LANCE si hay inf o NaN: es la forma
    # limpia de exigir "sin infinitos ni NaN".
    texto = json.dumps(d, allow_nan=False)
    assert json.loads(texto)["analisis"]["af_a"] is None   # None -> null -> None


def test_a_dict_serializable_todo_none():
    # Ninguna mano vio flop: wtsd None y ambos AF None. Debe seguir serializando.
    res = MatchResult("A", "B", hands=4, big_blind=50)
    res.net_chips_a, res.net_chips_b = 0, 0
    res.deltas_bb = [1.0, -1.0, 2.0, -2.0]
    d = a_dict(res, seed=1)
    an = d["analisis"]

    assert an["af_a"] is None and an["af_b"] is None and an["wtsd"] is None
    json.dumps(d, allow_nan=False)   # no debe lanzar


def test_a_dict_no_incluye_serie():
    # La serie mano a mano ya no va en el JSON: se vuelca aparte con --curva.
    d = a_dict(_match_con_af_none(), seed=1)
    assert "deltas_bb" not in d


def test_a_dict_analisis_flop_preflop():
    res = MatchResult("A", "B", hands=10, big_blind=50)
    res.n_saw_flop = 6                          # -> 4 manos acaban preflop
    res.wins_preflop_a, res.wins_preflop_b = 3, 1
    an = a_dict(res, seed=1)["analisis"]

    assert an["n_flop"] == 6
    assert an["n_preflop"] == 4
    assert an["preflop_a"] == 3 and an["preflop_b"] == 1


def test_a_dict_incluye_n_duelo():
    d = a_dict(_match_con_af_none(), seed=1, n_duelo=7)
    assert d["n_duelo"] == 7


def test_a_dict_analisis_preflop_cero():
    # Todas las manos ven flop: 0 manos preflop.
    res = MatchResult("A", "B", hands=5, big_blind=50)
    res.n_saw_flop = 5
    an = a_dict(res, seed=1)["analisis"]

    assert an["n_flop"] == 5 and an["n_preflop"] == 0
    assert an["preflop_a"] == 0 and an["preflop_b"] == 0
    json.dumps(a_dict(res, seed=1), allow_nan=False)
