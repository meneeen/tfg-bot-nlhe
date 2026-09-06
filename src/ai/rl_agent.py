from __future__ import annotations

from src.ai.base_agent import AgentAction, GameState, PokerAgent
from src.ai.rl.model_meta import load_meta
from src.ai.rl.poker_env import PokerEnv, build_obs, decode_action
from src.engine.controller import GameController


class RLAgent(PokerAgent):

    def __init__(self, model_path: str, deterministic: bool = True) -> None:
        from stable_baselines3 import PPO

        meta = load_meta(model_path)
        self._check_compatible(model_path, meta)

        self._model        = PPO.load(model_path)
        self.deterministic = deterministic

    @staticmethod
    def _check_compatible(model_path: str, meta: dict) -> None:

        if meta["obs_dim"] != PokerEnv.OBS_DIM:
            raise ValueError(
                f"Modelo incompatible: {model_path}\n"
                f"  se entrenó con una observación de {meta['obs_dim']} features, "
                f"pero el código actual usa {PokerEnv.OBS_DIM}.\n"
                f"  Reentrénalo:  python -m src.ai.rl.train"
            )
        if meta["big_blind"] != GameController.BIG_BLIND:
            raise ValueError(
                f"Modelo incompatible: {model_path}\n"
                f"  se entrenó con ciega grande {meta['big_blind']}, "
                f"pero ahora es {GameController.BIG_BLIND}.\n"
                f"  La recompensa se normaliza por la ciega. Reentrénalo:\n"
                f"    python -m src.ai.rl.train"
            )
        if meta["starting_stack"] != PokerEnv.STARTING_STACK:
            raise ValueError(
                f"Modelo incompatible: {model_path}\n"
                f"  se entrenó con stack inicial {meta['starting_stack']}, "
                f"pero ahora es {PokerEnv.STARTING_STACK}.\n"
                f"  Reentrénalo:  python -m src.ai.rl.train"
            )

    def decide_action(self, state: GameState) -> AgentAction:

        obs = build_obs(state)
        action_int, _ = self._model.predict(obs, deterministic=self.deterministic)
        return decode_action(int(action_int), state)
