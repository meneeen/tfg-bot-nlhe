"""
Tests para la selección de perfiles de sparring en el entrenamiento.

Cubre las dos piezas que deciden CONTRA QUIÉN entrena el agente:
1. _parse_profiles   — traduce --baseline-profile a una lista de perfiles.
2. _refresh_pool     — reparte las ranuras del pool entre reglas y yos pasados.

No entrena nada: importar train trae PPO, pero ninguna de estas dos funciones
lo toca.

Ejecutar con:
    pytest tests/ai/test_train_baselines.py -v
"""
import pytest

from src.ai.rule_based_agent import RuleBasedAgent
from src.ai.rule_profiles import BASELINE, LAG, PROFILES, STATION, TAG

train = pytest.importorskip(
    "src.ai.rl.train", reason="requiere stable-baselines3"
)


# ─────────────────────────── 1. _parse_profiles ──────────────────────────────

def test_un_solo_perfil():
    assert train._parse_profiles("tag") == [TAG]


def test_varios_perfiles_conservan_el_orden():
    assert train._parse_profiles("lag,tag,station") == [LAG, TAG, STATION]


def test_all_devuelve_todos():
    assert len(train._parse_profiles("all")) == len(PROFILES)


def test_tolera_espacios_y_mayusculas():
    assert train._parse_profiles(" TAG , lag ") == [TAG, LAG]


def test_descarta_duplicados():
    # Repetir un perfil duplicaría sus ranuras y sesgaría el reparto sin que el
    # informe lo dijera.
    assert train._parse_profiles("tag,tag,lag") == [TAG, LAG]


def test_perfil_desconocido_falla_claro():
    with pytest.raises(SystemExit) as exc:
        train._parse_profiles("tag,marciano")
    assert "marciano" in str(exc.value)


@pytest.mark.parametrize("spec", ["", "   ", ",,"])
def test_spec_vacio_falla_claro(spec):
    with pytest.raises(SystemExit) as exc:
        train._parse_profiles(spec)
    assert "vacío" in str(exc.value)


# ─────────────────────────── 2. _refresh_pool ────────────────────────────────

class _EnvEspia:
    """Sustituto de PokerEnv que solo recuerda el último pool recibido."""

    def __init__(self):
        self.pool = None

    def set_opponent_pool(self, agents):
        self.pool = list(agents)


def _callback(baselines, share, n_yos):
    env = _EnvEspia()
    cb = train.SelfPlayCallback(
        env, out_dir=".", interval=1, pool_size=8,
        baselines=baselines, baseline_share=share, verbose=0,
    )
    cb._pool = [f"yo{i}" for i in range(n_yos)]   # RLAgent falsos: solo se cuentan
    cb._refresh_pool()
    return env.pool


def _reglas(*perfiles):
    return [RuleBasedAgent(p, seed=0) for p in perfiles]


def _nombres(pool):
    return [a.profile.name for a in pool if isinstance(a, RuleBasedAgent)]


def test_share_cero_es_self_play_puro():
    pool = _callback(_reglas(TAG), share=0.0, n_yos=4)
    assert _nombres(pool) == []
    assert len(pool) == 4


def test_share_uno_es_solo_reglas():
    pool = _callback(_reglas(TAG, LAG), share=1.0, n_yos=4)
    assert sorted(_nombres(pool)) == ["lag", "tag"]
    assert not any(isinstance(a, str) for a in pool)


def test_reparto_equitativo_entre_perfiles():
    pool = _callback(_reglas(TAG, LAG, STATION), share=0.5, n_yos=6)
    from collections import Counter
    cuenta = Counter(_nombres(pool))
    assert set(cuenta) == {"tag", "lag", "station"}
    assert max(cuenta.values()) - min(cuenta.values()) <= 1


def test_share_de_la_mitad_da_mitad_de_ranuras():
    pool = _callback(_reglas(BASELINE), share=0.5, n_yos=8)
    assert len(_nombres(pool)) == 8      # 8 reglas + 8 yos
    assert len(pool) == 16


def test_ningun_perfil_desaparece_con_share_bajo():
    # 4 perfiles y un share que daría menos de 4 ranuras: aun así todos deben
    # aparecer, o el modelo se anunciaría como entrenado contra un rival que
    # nunca vio.
    pool = _callback(_reglas(TAG, LAG, STATION, BASELINE), share=0.1, n_yos=4)
    assert sorted(set(_nombres(pool))) == ["baseline", "lag", "station", "tag"]


def test_sin_yos_el_pool_es_solo_reglas():
    # Al arrancar el entrenamiento todavía no hay checkpoints propios.
    pool = _callback(_reglas(TAG), share=0.5, n_yos=0)
    assert _nombres(pool) == ["tag"]


def test_sin_baselines_el_pool_es_solo_yos():
    pool = _callback([], share=0.5, n_yos=3)
    assert pool == ["yo0", "yo1", "yo2"]


# ─────────────────────── 3. metadatos del modelo ─────────────────────────────

def test_los_perfiles_se_guardan_en_los_metadatos(tmp_path):
    from src.ai.rl.model_meta import load_meta, save_meta

    ruta = tmp_path / "modelo"
    save_meta(str(ruta), starting_stack=1000, big_blind=20, small_blind=10,
              obs_dim=32, training_profiles=["tag", "lag"])
    assert load_meta(str(ruta))["training_profiles"] == ["tag", "lag"]


def test_los_metadatos_sin_perfiles_siguen_siendo_validos(tmp_path):
    # Los modelos entrenados antes de que existieran los perfiles no llevan el
    # campo; load_meta no debe exigirlo.
    from src.ai.rl.model_meta import load_meta, save_meta

    ruta = tmp_path / "modelo"
    save_meta(str(ruta), starting_stack=1000, big_blind=20, small_blind=10,
              obs_dim=32)
    meta = load_meta(str(ruta))
    assert "training_profiles" not in meta
    assert meta["obs_dim"] == 32
