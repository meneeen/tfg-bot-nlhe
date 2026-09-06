from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.ai.rl.model_meta import save_meta
from src.ai.rl.poker_env import PokerEnv
from src.ai.rule_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from src.engine.controller import GameController


# ───────────────────────── receta de entrenamiento ───────────────────────────

@dataclass(frozen=True)
class Recipe:

    steps              : int    # pasos de entrenamiento (DEFAULT; override --steps)
    checkpoint_interval: int    # cada cuántos steps se guarda un rival al pool
    pool_size          : int    # rivales pasados que se conservan
    learning_rate      : float
    n_steps            : int    # buffer de rollout de PPO
    batch_size         : int

RECIPE = Recipe(
    steps               = 20_000,
    checkpoint_interval =  10_000,
    pool_size           =       8,
    learning_rate       =    3e-4,
    n_steps             =   2_048,
    batch_size          =      64,
)


# ─────────────────────────── callbacks ───────────────────────────────────────

class PokerStatsCallback(BaseCallback):

    _ACTION_NAMES = ["fold", "check_call", "raise33", "raise66", "raise100", "allin"]

    def __init__(self, csv_path: str, log_freq: int = 1_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path  = csv_path
        self.log_freq  = log_freq
        self._counts   = {i: 0 for i in range(6)}
        self._last_log = 0
        self._csv_file = None
        self._writer   = None
        self._opened   = False       # ¿ya se truncó el CSV en este run?

    def _on_training_start(self) -> None:
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._opened else "w"
        self._csv_file = open(self.csv_path, mode, newline="", encoding="utf-8")
        self._writer   = csv.writer(self._csv_file)
        if not self._opened:
            self._writer.writerow([
                "timestep", "ep_rew_mean", "ep_len_mean",
                *[f"{n}%" for n in self._ACTION_NAMES],
            ])
            self._opened = True

    def _on_step(self) -> bool:
        action = int(self.locals["actions"][0])
        self._counts[action] = self._counts.get(action, 0) + 1

        if self.num_timesteps - self._last_log >= self.log_freq:
            self._flush()
            self._last_log = self.num_timesteps
        return True

    def _flush(self) -> None:
        total = max(sum(self._counts.values()), 1)
        pcts  = [self._counts[i] / total for i in range(6)]

        buf = self.model.ep_info_buffer
        rew_mean = sum(e["r"] for e in buf) / len(buf) if buf else float("nan")
        len_mean = sum(e["l"] for e in buf) / len(buf) if buf else float("nan")

        self._writer.writerow([
            self.num_timesteps,
            round(rew_mean, 4),
            round(len_mean, 2),
            *[round(p, 4) for p in pcts],
        ])
        self._csv_file.flush()

        # Reinicia el contador: la próxima fila mide una ventana nueva.
        self._counts = {i: 0 for i in range(6)}

    def _on_training_end(self) -> None:
        self._flush()
        if self._csv_file:
            self._csv_file.close()


class SelfPlayCallback(BaseCallback):

    def __init__(
        self,
        poker_env: PokerEnv,
        out_dir: str | Path,
        interval: int,
        pool_size: int = 8,
        baselines=(),                 
        baseline_share: float = 0.5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.poker_env      = poker_env
        self.out_dir        = Path(out_dir)
        self.interval       = interval
        self.pool_size      = pool_size
        self.baselines      = list(baselines)
        self.baseline_share = baseline_share
        self._pool: list = []          # RLAgent ya instanciados (yos pasados)
        self._last_save     = 0
        self._n_checkpoints = 0

    def _on_training_start(self) -> None:
        self._checkpoint()
        self._last_save = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.interval:
            self._checkpoint()
            self._last_save = self.num_timesteps
        return True

    def _refresh_pool(self) -> None:
        rl = list(self._pool)
        s  = self.baseline_share
        if not self.baselines or s <= 0 or not rl:
            self.poker_env.set_opponent_pool(rl or list(self.baselines))
            return
        if s >= 1.0:
            self.poker_env.set_opponent_pool(list(self.baselines))
            return

        n_base = max(len(self.baselines), round(s * len(rl) / (1 - s)))
        ranuras = [self.baselines[i % len(self.baselines)] for i in range(n_base)]
        self.poker_env.set_opponent_pool(ranuras + rl)

    def _checkpoint(self) -> None:
        from src.ai.rl_agent import RLAgent

        path = Path(_save(
            self.model,
            self.out_dir / f"selfplay_checkpoint_{self._n_checkpoints:04d}",
        ))
        self._n_checkpoints += 1

        self._pool.append(RLAgent(f"{path}.zip", deterministic=False))
        if len(self._pool) > self.pool_size:
            self._pool.pop(0)          # descarta los más antiguos

        # El entorno muestrea de este pool en cada reset().
        self._refresh_pool()

        if self.verbose:
            perfiles = ", ".join(a.profile.name for a in self.baselines) or "ninguno"
            print(
                f"[self-play] step={self.num_timesteps:,}: "
                f"checkpoint -> {path.name}.zip  |  "
                f"pool de {len(self._pool)} yos + reglas [{perfiles}] "
                f"(~{self.baseline_share:.0%})"
            )


# ─────────────────────────── lógica de entrenamiento ─────────────────────────

def _save(model: PPO, path: str | Path, profiles: list[str] | None = None) -> str:
    path = str(path)
    model.save(path)
    save_meta(
        path,
        starting_stack    = PokerEnv.STARTING_STACK,
        big_blind         = GameController.BIG_BLIND,
        small_blind       = GameController.SMALL_BLIND,
        obs_dim           = PokerEnv.OBS_DIM,
        training_profiles = profiles,
    )
    return path


def _build_model(env: Monitor, r: Recipe, seed: int) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate = r.learning_rate,
        n_steps       = r.n_steps,
        batch_size    = r.batch_size,
        seed          = seed,
        verbose       = 1,
    )


def _print_banner(title: str, **kwargs) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")


# ─────────────────────────── CLI ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena un agente PPO en PokerEnv (NLHE heads-up). El "
                    "entrenamiento normal establece un agente inicial con pocos "
                    "pasos; se crece con `--resume <ckpt> --steps N`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seed", type=int, default=42, metavar="S",
        help="Semilla. Entrena con VARIAS (42, 43, 44...) y reporta media y "
             "desviación: un resultado de una sola semilla no dice nada.",
    )
    p.add_argument(
        "--out", type=str, default="models/rl/", metavar="DIR",
        help="Directorio de salida. Usa uno por semilla (o por versión al "
             "reanudar) para no sobrescribir.",
    )
    p.add_argument(
        "--resume", type=str, default=None, metavar="PATH",
        help="Continúa desde un checkpoint y entrena --steps pasos MÁS. Para "
             "recuperar una caída o seguir entrenando un modelo ya hecho.",
    )
    p.add_argument(
        "--steps", type=int, default=RECIPE.steps, metavar="N",
        help="Pasos de entrenamiento (self-play). Es lo que controla cuánto se "
             "entrena, también al reanudar con --resume.",
    )
    p.add_argument(
        "--baseline-share", type=float, default=0.5, metavar="S",
        help="Fracción de partidas jugadas contra el agente de reglas "
             "(0 = self-play puro; 0.5 = mitad; 1.0 = solo reglas, sin "
             "self-play). Ancla la agresión y evita que la política degenere.",
    )
    p.add_argument(
        "--baseline-profile", type=str, default=DEFAULT_PROFILE.name, metavar="PERFILES",
        help="Perfil(es) de reglas contra los que entrenar, separados por comas, "
             f"o 'all' para todos. Disponibles: {', '.join(sorted(PROFILES))}. "
             "Con varios, las partidas contra reglas se reparten a partes "
             "iguales: entrenar contra un solo estilo enseña a explotar ESE "
             "estilo, no a jugar al póker.",
    )
    return p.parse_args()


def _parse_profiles(spec: str) -> list:
    """'tag,lag' | 'all' -> lista de RuleProfile, sin repetidos y en orden."""
    if spec.strip().lower() == "all":
        return [PROFILES[n] for n in sorted(PROFILES)]

    perfiles, vistos = [], set()
    for trozo in spec.split(","):
        if not trozo.strip():
            continue
        try:
            perfil = get_profile(trozo)
        except KeyError as exc:
            raise SystemExit(str(exc)) from None
        if perfil.name not in vistos:      # duplicar sesgaría el reparto
            vistos.add(perfil.name)
            perfiles.append(perfil)

    if not perfiles:
        raise SystemExit(
            f"--baseline-profile está vacío. Usa uno o varios de: "
            f"{', '.join(sorted(PROFILES))}, o 'all'."
        )
    return perfiles


# ─────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    r    = RECIPE

    out_dir = Path(args.out)
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    from src.ai.rule_based_agent import RuleBasedAgent

    perfiles  = _parse_profiles(args.baseline_profile)
    # Semilla distinta por perfil: si todos comparten RNG, sus faroles caen en
    # las mismas manos y el agente ve menos variedad de la que cree ver.
    baselines = [RuleBasedAgent(p, seed=args.seed + i)
                 for i, p in enumerate(perfiles)]
    etiqueta  = ", ".join(p.name for p in perfiles)

    raw_env = PokerEnv(opponent=baselines[0])
    env     = Monitor(raw_env)

    stats_cb = PokerStatsCallback(str(log_dir / "training_stats.csv"))

    # Un único bucle de self-play (con los agentes de reglas mezclados en el
    # pool según --baseline-share). Al reanudar se carga el modelo; si no, se crea.
    if args.resume:
        _print_banner(
            "ENTRENAMIENTO (reanudar)",
            checkpoint=args.resume,
            steps=f"{args.steps:,}",
            vs_reglas=f"{args.baseline_share:.0%} de las partidas",
            perfiles=etiqueta,
        )
        model = PPO.load(args.resume, env=env)
    else:
        _print_banner(
            "ENTRENAMIENTO",
            steps=f"{args.steps:,}",
            vs_reglas=f"{args.baseline_share:.0%} de las partidas",
            perfiles=etiqueta,
            checkpoint_cada=f"{r.checkpoint_interval:,}",
            seed=args.seed, out=str(out_dir),
        )
        model = _build_model(env, r, args.seed)

    with tempfile.TemporaryDirectory(prefix="rl_selfplay_pool_",
                                     ignore_cleanup_errors=True) as pool_dir:
        sp_cb = SelfPlayCallback(
            raw_env, pool_dir, r.checkpoint_interval, r.pool_size,
            baselines=baselines, baseline_share=args.baseline_share,
            verbose=0,
        )
        model.learn(
            total_timesteps     = args.steps,
            callback            = [stats_cb, sp_cb],
            reset_num_timesteps = not args.resume,   # fresco reinicia; resume continúa
        )

    _save(model, out_dir / "modelo", profiles=[p.name for p in perfiles])

    print(f"\n[train] Modelo final guardado -> {out_dir / 'modelo'}.zip")
    print(f"[train] Entrenamiento completo. "
          f"Métricas por pasos en: {log_dir / 'training_stats.csv'}")


if __name__ == "__main__":
    main()
