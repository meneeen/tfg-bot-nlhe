"""
Espacio de observación  (OBS_DIM = 20 floats en [0, 1])
--------------------------------------------------------
  [0]     fuerza_mano         preflop_strength o postflop_equity_estimate
                              (continua: distingue AA overpair de top pair)
  [1-4]   street_one_hot      {PREFLOP, FLOP, TURN, RIVER}
  [5]     pot_to_stack_ratio  pot / (2 × STARTING_STACK)
  [6]     position            1=BTN/SB, 0=BB
  [7]     my_spr              min(stack / pot, SPR_CAP) / SPR_CAP
  [8]     opp_spr             ídem para el rival
  [9]     to_call_ratio       to_call / STARTING_STACK
  [10]    flush_draw          1.0 si faltan 1 carta para color
  [11]    straight_draw       1.0 si hay proyecto de escalera
  [12-19] action_history      N_ACTION_HISTORY × (tipo_norm, importe_norm)
                              0-relleno al inicio si hay menos de N acciones

Espacio de acciones (Discrete 6)
---------------------------------
  0  FOLD
  1  CHECK / CALL  (automático según to_call)
  2  RAISE 33 % del bote efectivo (pot tras igualar)
  3  RAISE 66 % del bote efectivo
  4  RAISE 100 % del bote efectivo
  5  ALL-IN

Recompensa
----------
  0.0 en cada step intermedio.
  (stack_final − STARTING_STACK) / BIG_BLIND al final de la mano.
  Ejemplo: ganar el blind del rival (+1 BB) → reward = +1.0
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import gymnasium

from src.engine.controller import GameController, STARTING_STACK
from src.engine.player import HumanPlayer, Action
from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent
from src.ai.agent_player import AgentPlayer, to_engine_action
from src.ai.hand_strength import (
    preflop_strength, postflop_equity_estimate, has_flush_draw, has_straight_draw,
)


# ────────────────────────────── constantes ────────────────────────────────────

_SPR_CAP        = 20     # stack/pot máximo antes de clampar a 1.0
_N_HIST         = 4      # últimas N acciones en la observación
_OBS_DIM        = 12 + _N_HIST * 2


_ACTION_TYPE_NORM: dict[str, float] = {
    "fold" : 0.00,
    "check": 0.25,
    "call" : 0.50,
    "raise": 1.00,   # all-ins históricas aparecen como raise en el motor
}

# Porcentajes de bote para las acciones de raise discreta
_RAISE_PCTS: dict[int, float] = {2: 0.33, 3: 0.66, 4: 1.00}

# Repartos máximos por posición en reset() antes de rendirse y cambiar de sitio.
# Salvaguarda contra rivales que SIEMPRE abandonan desde BTN y bloquean el episodio indefinidamente.
_MAX_DEALS_PER_POSITION = 50


def build_obs(state: GameState) -> np.ndarray:
    starting_stack = STARTING_STACK
    hole  = list(state.hole_cards)
    board = list(state.community_cards)

    # [0] fuerza de mano
    if hole:
        strength = (
            float(postflop_equity_estimate(hole, board))
            if board else
            float(preflop_strength(hole))
        )
    else:
        strength = 0.5   # valor neutro si aun no se han repartido cartas

    # [1-4] calle one-hot
    street_oh = [0.0, 0.0, 0.0, 0.0]
    street_oh[state.street.value] = 1.0

    # [5] pot-to-stack ratio
    pot_ratio = state.pot / (2 * starting_stack)

    # [6] posición: 1=BTN/SB, 0=BB
    position = 1.0 if state.position.value == "btn" else 0.0

    # [7-8] stack-to-pot ratio (clampado a _SPR_CAP)
    safe_pot = max(state.pot, 1)
    my_spr   = min(state.my_stack  / safe_pot, _SPR_CAP) / _SPR_CAP
    opp_spr  = min(state.opp_stack / safe_pot, _SPR_CAP) / _SPR_CAP

    # [9] to_call ratio
    to_call_ratio = min(state.to_call / starting_stack, 1.0)

    # [10-11] proyectos
    if hole and board:
        flush_draw    = 1.0 if has_flush_draw(hole, board)    else 0.0
        straight_draw = 1.0 if has_straight_draw(hole, board) else 0.0
    else:
        flush_draw = straight_draw = 0.0

    # [12-19] últimas _N_HIST acciones
    history    = state.action_history[-_N_HIST:]
    n_pad      = _N_HIST - len(history)
    hist_feats = [0.0, 0.0] * n_pad
    for rec in history:
        hist_feats.append(_ACTION_TYPE_NORM.get(rec.action_type.value, 0.75))
        hist_feats.append(min(rec.amount / starting_stack, 1.0))

    obs = np.array(
        [strength] + street_oh + [
            pot_ratio, position, my_spr, opp_spr, to_call_ratio,
            flush_draw, straight_draw,
        ] + hist_feats,
        dtype=np.float32,
    )
    return np.clip(obs, 0.0, 1.0)


def decode_action(action_int: int, state: GameState) -> AgentAction:
    to_call = state.to_call

    if action_int == 0:
        if to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.FOLD)

    if action_int == 1:
        return (
            AgentAction(ActionType.CHECK)
            if to_call == 0
            else AgentAction(ActionType.CALL)
        )

    if action_int in _RAISE_PCTS:
        pct      = _RAISE_PCTS[action_int]
        
        # bote despues de igualar
        eff_pot  = state.pot + to_call 
        
        raise_by = max(int(eff_pot * pct), state.big_blind)
        raise_to = state.my_bet_this_street + to_call + raise_by
        raise_to = max(raise_to, state.min_raise_to)
        if raise_to >= state.max_raise_to:
            return AgentAction(ActionType.ALL_IN)
        return AgentAction(ActionType.RAISE, raise_to)

    # action_int == 5
    return AgentAction(ActionType.ALL_IN)


# ═══════════════════════════════ PokerEnv ═════════════════════════════════════

class PokerEnv(gymnasium.Env):
    metadata       = {"render_modes": []}
    STARTING_STACK = STARTING_STACK     # constante del motor
    OBS_DIM        = _OBS_DIM
    N_ACTIONS      = 6

    # ──────────────────────────── __init__ ───────────────────────────────────

    def __init__(self, opponent: PokerAgent | None = None) -> None:
        super().__init__()

        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(self.N_ACTIONS)

        if opponent is None:
            from src.ai.rule_based_agent import RuleBasedAgent
            opponent = RuleBasedAgent()

        self._opponent_pool: list[PokerAgent] = [opponent]
        self._opponent_agent = opponent      # el muestreado ahora mismo

        self._rng = random.Random()

        self._rl_player  = HumanPlayer("RL",  self.STARTING_STACK)
        self._opp_player = AgentPlayer("Opp", self.STARTING_STACK, self._opponent_agent)
        self._gc         = GameController(self._rl_player, self._opp_player)
        self._opp_player.bind(self._gc)

        self._episode_idx = 0
        
        # Recompensa de manos que terminaron sin que el RL llegara a decidir
        self._pending_reward = 0.0

    # ─────────────────────────── API pública ─────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._gc.seed(seed)
            self._rng.seed(seed)      # y la secuencia de rivales del pool
            self._episode_idx = 0     # misma semilla -> misma secuencia de posiciones

        self._pick_opponent()

        deals = 0
        while True:
            self._rl_player.stack  = self.STARTING_STACK
            self._opp_player.stack = self.STARTING_STACK
            self._set_dealer()        # posición del episodio, antes de repartir
            self._gc.start_hand()
            self._advance_to_rl_turn()
            if not self._gc.hand_over:
                break

            self._pending_reward += self._chip_delta_bb()
            deals += 1

            if deals >= _MAX_DEALS_PER_POSITION: # por si el rival foldea siempre desde BTN
                self._episode_idx += 1
                deals = 0

        self._episode_idx += 1
        return self._observe(), {"position": self._position_str()}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        engine_action, amount = self._decode_action(action)
        self._gc.submit_human_action(engine_action, amount)
        self._advance_to_rl_turn()

        terminated = self._gc.hand_over
        if terminated:
            reward = self._chip_delta_bb() + self._pending_reward
            self._pending_reward = 0.0
        else:
            reward = 0.0

        info = {"position": self._position_str()} if terminated else {}
        return self._observe(), reward, terminated, False, info

    def set_opponent_pool(self, agents: list[PokerAgent]) -> None:
        if not agents:
            raise ValueError(
                "El pool de rivales no puede estar vacío: el entorno necesita "
                "contra quién jugar. Para un rival fijo, usa set_opponent(agent)."
            )
        self._opponent_pool = list(agents)

    def set_opponent(self, agent: PokerAgent) -> None:
        """Rival fijo. Azúcar para un pool de un solo elemento."""
        self.set_opponent_pool([agent])

    def _pick_opponent(self) -> None:
        """Muestrea el rival de este episodio del pool (nunca está vacío)."""
        agent = self._rng.choice(self._opponent_pool)
        self._opponent_agent = agent
        self._opp_player.set_agent(agent)

    # ─────────────────────── lógica interna ─────────────────────────────────

    def _set_dealer(self) -> None:
        rl_is_btn = (self._episode_idx % 2 == 0)
        self._gc.dealer = self._rl_player if rl_is_btn else self._opp_player

    def _position_str(self) -> str:
        return "btn" if self._gc.dealer is self._rl_player else "bb"

    def _chip_delta_bb(self) -> float:
        # Ganancia/pérdida del RL en la mano actual, en ciegas grandes.
        return float(self._rl_player.stack - self.STARTING_STACK) / self._gc.BIG_BLIND

    def _advance_to_rl_turn(self) -> None:
        # Llama gc.step() hasta que le toque al agente RL o acabe la mano.
        while not self._gc.hand_over and not self._gc.awaiting_human:
            self._gc.step()

    def _decode_action(self, action: int) -> tuple[Action, int]:
        state = GameState.from_engine_state(self._gc, self._rl_player)
        return to_engine_action(decode_action(action, state), state)

    def _observe(self) -> np.ndarray:
        state = GameState.from_engine_state(self._gc, self._rl_player)
        return build_obs(state)
