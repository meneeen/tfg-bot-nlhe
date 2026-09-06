"""
Tests para PokerEnv.

Cubren:
- Shapes de observation_space y action_space
- Contrato de Gymnasium (reset/step API)
- Que todas las observaciones devueltas están dentro del espacio
- Que la recompensa intermedia es 0 y la terminal es finita
- Terminación en tiempo finito con acciones aleatorias
- Configuración del oponente y set_opponent()
- Invariantes de la recompensa (fold desde BTN siempre pierde ≤ 0.5 BB)
"""
from __future__ import annotations

import math
import pytest

from src.ai.rl.poker_env import PokerEnv
from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent


# ───────────────────────── agentes auxiliares de test ────────────────────────

class _AlwaysCallAgent(PokerAgent):
    """Agente que siempre iguala (o pasa si puede)."""
    def decide_action(self, state: GameState) -> AgentAction:
        if state.to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.CALL)


class _AlwaysFoldAgent(PokerAgent):
    """Agente que siempre se retira (o pasa si no hay apuesta)."""
    def decide_action(self, state: GameState) -> AgentAction:
        if state.to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.FOLD)


class _AlwaysAllInAgent(PokerAgent):
    """Agente que siempre va all-in."""
    def decide_action(self, state: GameState) -> AgentAction:
        return AgentAction(ActionType.ALL_IN)


# ──────────────────────────── fixtures ───────────────────────────────────────

@pytest.fixture
def env():
    return PokerEnv(opponent=_AlwaysCallAgent())


@pytest.fixture
def fold_opponent_env():
    return PokerEnv(opponent=_AlwaysFoldAgent())


# ══════════════════════ TestObservationSpace ══════════════════════════════════

class TestObservationSpace:
    def test_obs_dim_constant(self):
        # 18 -> 20 al añadir las features de proyecto (flush_draw, straight_draw)
        assert PokerEnv.OBS_DIM == 20

    def test_obs_shape_from_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == (PokerEnv.OBS_DIM,)

    def test_obs_dtype_float32(self, env):
        import numpy as np
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_in_observation_space_after_reset(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            f"obs fuera del espacio: min={obs.min():.4f}, max={obs.max():.4f}"
        )

    def test_obs_in_space_after_every_step(self, env):
        obs, _ = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            assert env.observation_space.contains(obs), (
                f"obs fuera del espacio en step: min={obs.min():.4f}, max={obs.max():.4f}"
            )
            if terminated:
                obs, _ = env.reset()
                assert env.observation_space.contains(obs)

    def test_obs_bounds_are_zero_one(self):
        space = PokerEnv().observation_space
        import numpy as np
        assert (space.low  == 0.0).all()
        assert (space.high == 1.0).all()


# ══════════════════════ TestActionSpace ══════════════════════════════════════

class TestActionSpace:
    def test_n_actions_constant(self):
        assert PokerEnv.N_ACTIONS == 6

    def test_action_space_is_discrete_6(self, env):
        import gymnasium
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 6


# ══════════════════════ TestResetAPI ═════════════════════════════════════════

class TestResetAPI:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_info_is_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_with_seed_is_reproducible(self, env):
        import numpy as np
        # Misma semilla -> mismas cartas repartidas -> misma observación.
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)

    def test_different_seeds_differ(self, env):
        import numpy as np
        # Semillas distintas deben (casi siempre) producir observaciones distintas.
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        assert not np.array_equal(obs_a, obs_b)

    def test_seeded_hand_sequence_is_reproducible(self, env):
        # Una secuencia completa de manos (sin re-sembrar entre ellas) debe
        # ser idéntica al reanudar con la misma semilla inicial.
        def rollout():
            env.reset(seed=123)
            hands = []
            for _ in range(5):
                hands.append(tuple(env._rl_player.hole_cards))
                env.reset()
            return hands
        assert rollout() == rollout()

    def test_multiple_resets_dont_crash(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        for _ in range(10):
            env.reset()


# ══════════════════════ TestStepAPI ══════════════════════════════════════════

class TestStepAPI:
    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step(0)   # FOLD
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (PokerEnv.OBS_DIM,)

    def test_step_reward_is_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, env):
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_always_false(self, env):
        env.reset()
        for action in range(6):
            env.reset()
            _, _, _, truncated, _ = env.step(action)
            assert truncated is False

    def test_step_info_is_dict(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert isinstance(info, dict)


# ══════════════════════ TestEpisodeTermination ════════════════════════════════

class TestEpisodeTermination:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
    def test_random_episode_terminates(self, seed):
        env   = PokerEnv(opponent=_AlwaysCallAgent())
        env.reset(seed=seed)
        max_steps = 200
        for i in range(max_steps):
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                return
        pytest.fail(f"Episodio no terminó en {max_steps} pasos (seed={seed})")

    def test_fold_terminates_when_facing_a_bet(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        for _ in range(30):
            env.reset()
            if env._gc.human_to_call > 0:
                _, _, terminated, _, _ = env.step(0)   # FOLD real
                assert terminated
                assert env._rl_player.folded
                return
        pytest.skip("no se alcanzó un estado con to_call > 0")

    def test_allin_episode_terminates(self):
        env = PokerEnv(opponent=_AlwaysAllInAgent())
        for _ in range(5):
            env.reset()
            for _ in range(20):
                action = 5   # RL también va all-in
                _, _, terminated, _, _ = env.step(action)
                if terminated:
                    break
            else:
                pytest.fail("Episodio all-in vs all-in no terminó en 20 pasos")

    def test_multiple_complete_episodes(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        for ep in range(20):
            env.reset()
            steps = 0
            while steps < 200:
                _, _, terminated, _, _ = env.step(env.action_space.sample())
                steps += 1
                if terminated:
                    break
            else:
                pytest.fail(f"Episodio {ep} no terminó en 200 pasos")


# ══════════════════════ TestReward ═══════════════════════════════════════════

class TestReward:
    def test_intermediate_reward_is_zero(self):
        """Pasos no terminales deben dar recompensa 0."""
        env = PokerEnv(opponent=_AlwaysCallAgent())
        found_nonterminal = False
        for _ in range(20):
            env.reset()
            # CHECK/CALL raramente termina inmediatamente
            _, reward, terminated, _, _ = env.step(1)
            if not terminated:
                assert reward == 0.0
                found_nonterminal = True
                break
        # Si todos terminaron, el test es inconcluso pero no falla
        # (puede ocurrir si siempre hay fold del oponente)
        if not found_nonterminal:
            pytest.skip("No se encontró paso no terminal en 20 intentos")

    def test_terminal_reward_is_finite(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        for _ in range(20):
            env.reset()
            for _ in range(200):
                _, reward, terminated, _, _ = env.step(env.action_space.sample())
                if terminated:
                    assert math.isfinite(reward), f"reward no finito: {reward}"
                    break

    def test_fold_from_btn_loses_small_blind(self):
        """
        Si el agente RL es BTN/SB y se retira preflop, pierde exactamente su
        ciega pequeña. Expresado en BB (que es la unidad de la recompensa), eso
        es −SB/BB — se deriva de las constantes, no se hardcodea, para que el
        test sobreviva a un cambio de estructura de ciegas.
        """
        from src.engine.controller import GameController
        expected = -GameController.SMALL_BLIND / GameController.BIG_BLIND

        env = PokerEnv(opponent=_AlwaysCallAgent())
        for _ in range(20):
            env.reset()
            if env._gc.dealer is env._rl_player:   # RL es BTN/SB
                _, reward, terminated, _, _ = env.step(0)   # FOLD
                assert terminated
                assert reward == pytest.approx(expected), (
                    f"Fold como SB debería costar {expected} BB, got {reward}"
                )
                return
        pytest.skip("RL nunca fue BTN en 20 intentos")

    def test_no_reward_is_lost_to_skipped_hands(self):
        """
        Invariante clave del arreglo de recompensa: la suma de las recompensas
        devueltas iguala las fichas ganadas en TODAS las manos repartidas,
        incluidas aquellas que terminaron sin que el RL llegara a decidir
        (el rival se retiró desde BTN). Antes esas manos se descartaban y su
        recompensa se perdía.
        """
        env = PokerEnv(opponent=_AlwaysFoldAgent())

        deltas: list[float] = []
        real_delta = env._chip_delta_bb

        def spy() -> float:                 # una llamada = una mano terminada
            d = real_delta()
            deltas.append(d)
            return d
        env._chip_delta_bb = spy

        total_reward = 0.0
        for _ in range(30):
            env.reset()
            done = False
            while not done:
                _, r, done, _, _ = env.step(1)   # check/call
            total_reward += r

        assert total_reward == pytest.approx(sum(deltas), abs=1e-6), (
            "Se perdió recompensa de manos sin decisión del RL"
        )


# ══════════════════════ TestPositionAndFreeFold ═══════════════════════════════

class TestPositionAndFreeFold:
    """Regresión de los dos bugs bloqueantes del entorno."""

    def test_position_is_exactly_balanced(self):
        """
        La posición se fija por episodio (par=BTN, impar=BB), no se deja a la
        rotación de dealer del motor. Antes salía 67 % BTN / 33 % BB, porque
        las manos descartadas (rival se retira desde BTN) rotaban el dealer y
        caían sistemáticamente sobre las manos en que el RL era BB.
        """
        env = PokerEnv(opponent=_AlwaysCallAgent())   # no descarta manos
        btn, N = 0, 200
        for _ in range(N):
            env.reset()
            if env._gc.dealer is env._rl_player:
                btn += 1
            done = False
            while not done:
                _, _, done, _, _ = env.step(1)
        assert btn == N // 2, f"Posición desbalanceada: {btn}/{N} BTN"

    def test_action_zero_never_folds_for_free(self):
        """
        Retirarse pudiendo pasar gratis es una acción estrictamente dominada.
        La acción 0 debe sanearse a CHECK cuando to_call == 0, igual que hacen
        RuleBasedAgent y LLMAgent.
        """
        env = PokerEnv(opponent=_AlwaysCallAgent())
        checked_any = False
        for _ in range(40):
            env.reset()
            if env._gc.human_to_call == 0:
                env.step(0)
                assert not env._rl_player.folded, (
                    "La acción 0 se retiró pudiendo pasar gratis"
                )
                checked_any = True
        if not checked_any:
            pytest.skip("no se alcanzó un estado con to_call == 0")


# ══════════════════════ TestOpponentPool ══════════════════════════════════════

class TestOpponentPool:
    """
    El rival debe rotar en CADA episodio, no cada N steps.

    Si el rival solo cambiara al guardar checkpoint, el agente pasaría decenas
    de miles de steps seguidos contra un rival fijo y se sobreajustaría a él
    antes de que cambiase.
    """

    def _play_episode(self, env) -> None:
        done = False
        while not done:
            _, _, done, _, _ = env.step(1)

    def _opponent_sequence(self, env, n: int, seed: int | None = None) -> list[str]:
        """Tipos de rival efectivamente usados en n episodios consecutivos."""
        seen: list[str] = []
        for i in range(n):
            env.reset(seed=seed if i == 0 else None)
            seen.append(type(env._opponent_agent).__name__)
            self._play_episode(env)
        return seen

    def test_opponent_varies_between_episodes(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        env.set_opponent_pool([
            _AlwaysCallAgent(), _AlwaysFoldAgent(), _AlwaysAllInAgent(),
        ])
        seen = self._opponent_sequence(env, 60, seed=11)
        assert len(set(seen)) > 1, (
            "El rival no rotó entre episodios: el pool no se está muestreando"
        )

    def test_all_pool_members_get_used(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        env.set_opponent_pool([
            _AlwaysCallAgent(), _AlwaysFoldAgent(), _AlwaysAllInAgent(),
        ])
        seen = set(self._opponent_sequence(env, 120, seed=3))
        assert seen == {
            "_AlwaysCallAgent", "_AlwaysFoldAgent", "_AlwaysAllInAgent",
        }, f"Algún rival del pool nunca se usó: {seen}"

    def test_pool_sampling_is_reproducible_with_seed(self):
        def run() -> list[str]:
            env = PokerEnv(opponent=_AlwaysCallAgent())
            env.set_opponent_pool([
                _AlwaysCallAgent(), _AlwaysFoldAgent(), _AlwaysAllInAgent(),
            ])
            return self._opponent_sequence(env, 30, seed=42)

        assert run() == run(), (
            "La secuencia de rivales no es reproducible con la misma semilla"
        )

    def test_different_seeds_give_different_sequences(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        pool = [_AlwaysCallAgent(), _AlwaysFoldAgent(), _AlwaysAllInAgent()]
        env.set_opponent_pool(pool)
        a = self._opponent_sequence(env, 30, seed=1)
        env.set_opponent_pool(pool)
        b = self._opponent_sequence(env, 30, seed=2)
        assert a != b

    def test_empty_pool_is_rejected(self):
        """
        El pool es el ÚNICO mecanismo de rival, así que no puede estar vacío:
        el entorno necesita contra quién jugar. Antes había una rama de fallback
        al «rival único»; ahora un rival fijo ES un pool de tamaño 1.
        """
        env = PokerEnv(opponent=_AlwaysCallAgent())
        with pytest.raises(ValueError, match="vacío"):
            env.set_opponent_pool([])

    def test_single_opponent_is_a_pool_of_one(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        assert len(env._opponent_pool) == 1
        env.set_opponent(_AlwaysFoldAgent())
        assert len(env._opponent_pool) == 1

    def test_set_opponent_clears_the_pool(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        env.set_opponent_pool([_AlwaysFoldAgent(), _AlwaysAllInAgent()])
        env.set_opponent(_AlwaysCallAgent())          # vuelve a rival único
        seen = self._opponent_sequence(env, 10)
        assert set(seen) == {"_AlwaysCallAgent"}

    def test_pool_agents_are_not_reinstantiated(self):
        """
        El entorno debe reutilizar las instancias del pool por referencia.
        Los RLAgent reales cargan su .zip en __init__; recrearlos en cada
        reset() sería prohibitivamente lento.
        """
        env  = PokerEnv(opponent=_AlwaysCallAgent())
        pool = [_AlwaysCallAgent(), _AlwaysFoldAgent()]
        env.set_opponent_pool(pool)

        used_ids = set()
        for _ in range(30):
            env.reset()
            used_ids.add(id(env._opponent_agent))
            self._play_episode(env)

        assert used_ids <= {id(a) for a in pool}, (
            "El entorno usó instancias de agente que no son las del pool"
        )


# ══════════════════════ TestOpponentConfig ════════════════════════════════════

class TestOpponentConfig:
    def test_default_opponent_is_rule_based(self):
        from src.ai.rule_based_agent import RuleBasedAgent
        env = PokerEnv()
        assert isinstance(env._opponent_agent, RuleBasedAgent)

    def test_custom_opponent_accepted(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        assert isinstance(env._opponent_agent, _AlwaysCallAgent)
        obs, _ = env.reset()
        assert obs.shape == (PokerEnv.OBS_DIM,)

    def test_set_opponent_takes_effect_on_next_reset(self):
        """
        Cambiar de rival afecta al SIGUIENTE episodio, no al que está en curso:
        el rival se muestrea del pool en reset(). No se cambia de rival a mitad
        de mano.
        """
        env = PokerEnv(opponent=_AlwaysCallAgent())
        env.set_opponent(_AlwaysFoldAgent())
        obs, _ = env.reset()
        assert isinstance(env._opponent_agent, _AlwaysFoldAgent)
        assert obs.shape == (PokerEnv.OBS_DIM,)

    def test_set_opponent_mid_episode(self, env):
        """Cambiar de oponente entre episodios no rompe nada."""
        env.reset()
        env.step(1)   # al menos un step
        env.set_opponent(_AlwaysFoldAgent())
        # El nuevo oponente actúa en el siguiente reset
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_allin_opponent_finishes_hands(self):
        env = PokerEnv(opponent=_AlwaysAllInAgent())
        for _ in range(5):
            env.reset()
            for _ in range(20):
                _, _, terminated, _, _ = env.step(1)
                if terminated:
                    break
            else:
                pytest.fail("Mano contra all-in no terminó")


# ══════════════════════ TestObsSemantics ═════════════════════════════════════

class TestObsSemantics:
    def test_position_alternates_between_episodes(self):
        """El botón rota, así que la posición debe cambiar entre manos."""
        env = PokerEnv(opponent=_AlwaysCallAgent())
        positions = set()
        for _ in range(10):
            obs, _ = env.reset()
            positions.add(float(obs[6]))   # índice 6 = posición
            # Terminar la mano para que rote el botón
            for _ in range(100):
                _, _, terminated, _, _ = env.step(env.action_space.sample())
                if terminated:
                    break
        assert len(positions) == 2, (
            "La posición debería alternar entre 0.0 (BB) y 1.0 (BTN)"
        )

    def test_street_one_hot_is_valid(self):
        """El one-hot de calle debe tener exactamente un 1."""
        env = PokerEnv(opponent=_AlwaysCallAgent())
        for _ in range(5):
            obs, _ = env.reset()
            _check_street_one_hot(obs)
            for _ in range(50):
                obs, _, terminated, _, _ = env.step(1)
                _check_street_one_hot(obs)
                if terminated:
                    break

    def test_hand_strength_in_range(self):
        env = PokerEnv(opponent=_AlwaysCallAgent())
        for _ in range(10):
            obs, _ = env.reset()
            assert 0.0 <= obs[0] <= 1.0, f"fuerza_mano fuera de rango: {obs[0]}"


def _check_street_one_hot(obs):
    street_oh = obs[1:5]
    assert street_oh.sum() == pytest.approx(1.0), (
        f"one-hot de calle inválido: {street_oh}"
    )
    assert set(street_oh.tolist()).issubset({0.0, 1.0}), (
        f"one-hot de calle tiene valores no binarios: {street_oh}"
    )
