from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev
from typing import TextIO

from treys import Card

from src.ai.agent_player import AgentPlayer
from src.ai.base_agent import PokerAgent
from src.engine.controller import (
    GameController, HAND_DONE, STARTING_STACK, Street,
)

_STREET_ORDER = ["PREFLOP", "FLOP", "TURN", "RIVER"]


def _af(raises: int, calls: int) -> float | None:
    """raises/calls. None (no infinito) si no hay calls, para serializar a JSON.

    Compartido por MatchResult (solo postflop) y SessionMatchResult (todas las
    calles): la fórmula es la misma, solo cambia qué contadores se le pasan.
    """
    return None if calls == 0 else raises / calls


@dataclass
class MatchResult:

    name_a: str
    name_b: str
    hands: int
    big_blind: int
    net_chips_a: int = 0                       # fichas netas ganadas por A
    net_chips_b: int = 0                        # fichas netas ganadas por B
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    actions_a: dict[str, int] = field(default_factory=dict)
    actions_b: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

    # Serie completa de resultados por mano (ganancia de A en ciegas grandes).
    # Es la base para la curva de bankroll y para el intervalo de confianza; se
    # guarda entera, no solo su suma.
    deltas_bb: list[float] = field(default_factory=list)

    # Contadores de agresión POSTFLOP, por jugador (calles != PREFLOP).
    raises_postflop_a: int = 0
    calls_postflop_a: int = 0
    raises_postflop_b: int = 0
    calls_postflop_b: int = 0

    # Contadores WTSD, crudos. En heads-up el showdown es propiedad de la MANO,
    # no del jugador, así que hay un único par de contadores por partida.
    n_showdowns: int = 0
    n_saw_flop: int = 0

    # Serie paralela a deltas_bb: True si esa mano llegó a ver el flop (board>=3).
    saw_flop_flags: list[bool] = field(default_factory=list)

    # Manos que terminaron ANTES del flop, por ganador. En HU un fold preflop
    # siempre tiene ganador claro (delta_a != 0), así que aquí no hay empates.
    wins_preflop_a: int = 0
    wins_preflop_b: int = 0

    @property
    def bb_per_100(self) -> float:
        if not self.hands:
            return 0.0
        return (self.net_chips_a / self.big_blind) / self.hands * 100

    @property
    def ci95(self) -> float:
        """Semiancho del intervalo de confianza al 95 % de bb/100.

        1.96 * desviacion_tipica_poblacional(deltas_bb) / sqrt(n) * 100.
        """
        n = len(self.deltas_bb)
        if n == 0:
            return 0.0
        return 1.96 * pstdev(self.deltas_bb) / math.sqrt(n) * 100

    @property
    def significant(self) -> bool:
        """El resultado supera su propio margen de error (a 1.96 sigma)."""
        return abs(self.bb_per_100) > self.ci95

    @property
    def af_a(self) -> float | None:
        """Factor de agresión postflop de A (raises/calls). None si no calls."""
        return _af(self.raises_postflop_a, self.calls_postflop_a)

    @property
    def af_b(self) -> float | None:
        """Factor de agresión postflop de B (raises/calls). None si no calls."""
        return _af(self.raises_postflop_b, self.calls_postflop_b)

    @property
    def wtsd(self) -> float | None:
        """WTSD = showdowns / manos que vieron flop.

        OJO: en heads-up el showdown es una propiedad de la MANO, no del
        jugador. Si hay showdown, llegaron los DOS; por eso es UN solo valor por
        partida, no uno por jugador. None si ninguna mano vio flop (denominador
        0, serializable a JSON).
        """
        if self.n_saw_flop == 0:
            return None
        return self.n_showdowns / self.n_saw_flop

    def _record_hand_metrics(
        self,
        action_log: list[tuple[str, str, int, str, int]],
        result: dict,
        delta_a: int,
    ) -> None:
        """Acumula las métricas nuevas de UNA mano en el agregado de la partida.

        Extraído aparte para poder testear la clasificación (postflop vs
        preflop, denominador de WTSD) sin depender de las cartas de una partida
        real. No toca net_chips_a/wins: de eso sigue encargándose play_match.

        Atribuye cada acción por el ÍNDICE del jugador (5º campo del log), no por
        nombre: en un duelo espejo ambos comparten nombre y contar por nombre
        volcaría todo al jugador A.
        """
        self.deltas_bb.append(delta_a / self.big_blind)

        for _name, action, _amount, street, idx in action_log:
            if street == "PREFLOP":
                continue
            if idx == 0:
                if action == "raise":
                    self.raises_postflop_a += 1
                elif action == "call":
                    self.calls_postflop_a += 1
            else:
                if action == "raise":
                    self.raises_postflop_b += 1
                elif action == "call":
                    self.calls_postflop_b += 1

        saw_flop = len(result["board"]) >= 3
        self.saw_flop_flags.append(saw_flop)
        if saw_flop:
            self.n_saw_flop += 1
        else:
            # Mano terminada antes del flop (siempre por fold en HU): el ganador
            # se lee del signo de delta_a; un fold nunca deja delta 0.
            if delta_a > 0:
                self.wins_preflop_a += 1
            elif delta_a < 0:
                self.wins_preflop_b += 1

        if not result["by_fold"]:
            self.n_showdowns += 1


def _cards(cs) -> str:
    return " ".join(Card.int_to_str(c) for c in cs)


def _board_upto(board: list[int], street: str) -> str:
    n = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}[street]
    return _cards(board[:n]) if n else ""


def _write_hand(
    out: TextIO,
    gc: GameController,
    a: AgentPlayer,
    b: AgentPlayer,
    delta_a: int,
) -> None:
    r = gc.result
    # gc.dealer ya rotó en _finalize; el dealer de ESTA mano es el otro.
    dealer_name = (b if gc.dealer is a else a).name

    out.write(f"\n=== Mano {r['hand_num']} ===   (dealer/SB: {dealer_name})\n")
    out.write(f"  {a.name:<6}: {_cards(r['player_cards'])}\n")
    out.write(f"  {b.name:<6}: {_cards(r['agent_cards'])}\n")

    # Acciones agrupadas por calle, en orden
    by_street: dict[str, list[str]] = {s: [] for s in _STREET_ORDER}
    for name, action, amount, street, _idx in r["actions"]:
        txt = action if amount == 0 else f"{action} {amount}"
        by_street[street].append(f"{name} {txt}")

    for street in _STREET_ORDER:
        if not by_street[street]:
            continue
        board = _board_upto(r["board"], street)
        head = f"  {street:<8}"
        if board:
            head += f"[{board}]  "
        out.write(head + "  |  ".join(by_street[street]) + "\n")

    if r["by_fold"]:
        how = "se retiró el rival"
    else:
        how = f"{r['player_hand']}  vs  {r['agent_hand']}"
    sign = "+" if delta_a >= 0 else ""
    out.write(
        f"  -> gana {r['winner']}  ({how})  |  bote {r['pot']}  |  "
        f"{a.name} {sign}{delta_a}\n"
    )


def play_match(
    agent_a: PokerAgent,
    agent_b: PokerAgent,
    name_a: str,
    name_b: str,
    hands: int = 100,
    seed: int | None = 42,
    log_path: str | Path | None = None,
) -> MatchResult:
    starting_stack = STARTING_STACK
    progress_every = max(hands // 10, 1)
    pa = AgentPlayer(name_a, starting_stack, agent_a)
    pb = AgentPlayer(name_b, starting_stack, agent_b)
    gc = GameController(pa, pb, seed=seed)
    pa.bind(gc)
    pb.bind(gc)

    res = MatchResult(name_a, name_b, hands, GameController.BIG_BLIND)

    log_file = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"{name_a}  vs  {name_b}\n")
        log_file.write(f"manos={hands}  stack={starting_stack}  "
                       f"ciegas={GameController.SMALL_BLIND}/{GameController.BIG_BLIND}  "
                       f"seed={seed}\n")

    t0 = time.perf_counter()
    try:
        for i in range(hands):
            pa.stack = pb.stack = starting_stack
            # El botón alterna: cada agente juega la mitad de manos en cada
            # posición. Sin esto, uno acumularía la ventaja del botón.
            gc.dealer = pa if i % 2 == 0 else pb

            gc.start_hand()
            while gc.step() != HAND_DONE:
                pass

            delta_a = pa.stack - starting_stack
            # net_chips_b permite comprobar la conservacion de fichas (juego de
            # suma cero): net_chips_a + net_chips_b debe ser 0. Una desviacion
            # delata una fuga de fichas en el motor -> resultados invalidos.
            res.net_chips_a += delta_a
            res.net_chips_b += pb.stack - starting_stack
            if delta_a > 0:
                res.wins_a += 1
            elif delta_a < 0:
                res.wins_b += 1
            else:
                res.ties += 1

            for _name, action, _amt, _street, idx in gc.action_log:
                bucket = res.actions_a if idx == 0 else res.actions_b
                bucket[action] = bucket.get(action, 0) + 1

            res._record_hand_metrics(gc.action_log, gc.result, delta_a)

            if log_file:
                _write_hand(log_file, gc, pa, pb, delta_a)

            if (i + 1) % progress_every == 0:
                done = i + 1
                rate = (time.perf_counter() - t0) / done
                eta  = rate * (hands - done)
                print(f"  {done}/{hands} manos  |  "
                      f"{name_a} {res.net_chips_a:+d} fichas  |  "
                      f"ETA {eta/60:.1f} min", flush=True)
    finally:
        res.elapsed_s = time.perf_counter() - t0
        if log_file:
            log_file.write("\n" + summary(res) + "\n")
            log_file.close()

    return res


def _fmt_af(af: float | None) -> str:
    return "n/a" if af is None else f"{af:.2f}"


def _fmt_wtsd(wtsd: float | None) -> str:
    return "n/a" if wtsd is None else f"{wtsd:.1%}"


# ══════════════════════ modo partida al KO (sesiones) ════════════════════════
#
# play_match() resetea los stacks en cada mano: mide bb/100 sobre manos
# independientes. Este modo es lo contrario y NO lo sustituye: los stacks se
# arrastran y se juega hasta que uno arruina al otro, que es lo que pasa en una
# partida real de heads-up. La métrica es "partidas ganadas", no bb/100.


@dataclass
class SessionResult:
    """Resultado de UNA partida al KO."""

    name_a: str
    name_b: str
    winner: str | None          # name_a, name_b, o None si se acabó por límite
    hands: int
    final_stack_a: int
    final_stack_b: int
    # Stack de cada jugador DESPUÉS de cada mano, con el stack inicial en la
    # posición 0. Es la serie para pintar la evolución de una partida concreta,
    # así que se guarda entera (len == hands + 1).
    stacks_a: list[int] = field(default_factory=list)
    stacks_b: list[int] = field(default_factory=list)

    # Contadores de agresión, TODAS las calles (a diferencia de MatchResult,
    # que solo cuenta postflop): aquí la partida se evalúa como paquete
    # completo, no por su fase postflop.
    raises_a: int = 0
    calls_a: int = 0
    raises_b: int = 0
    calls_b: int = 0

    # Contadores WTSD, crudos, mismo criterio que en MatchResult (ver ahí):
    # propiedad de la MANO, no del jugador.
    n_saw_flop: int = 0
    n_showdowns: int = 0

    @property
    def by_limit(self) -> bool:
        """True si la partida no acabó en KO sino al agotar max_hands."""
        return self.winner is None

    def _record_hand_metrics(self, action_log, result: dict) -> None:
        """Acumula agresión (todas las calles) y WTSD de UNA mano.

        Atribuye cada acción por el ÍNDICE del jugador (5º campo del log), no
        por nombre: en un duelo espejo ambos comparten nombre.
        """
        for _name, action, _amount, _street, idx in action_log:
            if idx == 0:
                if action == "raise":
                    self.raises_a += 1
                elif action == "call":
                    self.calls_a += 1
            else:
                if action == "raise":
                    self.raises_b += 1
                elif action == "call":
                    self.calls_b += 1

        if len(result["board"]) >= 3:
            self.n_saw_flop += 1
        if not result["by_fold"]:
            self.n_showdowns += 1


@dataclass
class SessionMatchResult:
    """Marcador de N partidas al KO entre los mismos dos agentes."""

    name_a: str
    name_b: str
    n_sessions: int
    starting_stack: int
    max_hands: int
    seed: int
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0                                  # partidas cortadas por límite
    # Manos que duró cada partida. No se reporta ni se guarda en disco: queda
    # como registro en memoria del que tiran los tests para comprobar que dos
    # tandas con la misma semilla juegan exactamente las mismas partidas.
    durations: list[int] = field(default_factory=list)
    elapsed_s: float = 0.0

    # Agregado de _record_hand_metrics sobre TODAS las manos de TODAS las
    # partidas de la tanda. AF aquí cuenta todas las calles (no solo postflop,
    # a diferencia de MatchResult): la partida se evalúa como paquete completo.
    raises_a: int = 0
    calls_a: int = 0
    raises_b: int = 0
    calls_b: int = 0
    n_saw_flop: int = 0
    n_showdowns: int = 0

    @property
    def af_a(self) -> float | None:
        """Factor de agresión de A, todas las calles (raises/calls)."""
        return _af(self.raises_a, self.calls_a)

    @property
    def af_b(self) -> float | None:
        """Factor de agresión de B, todas las calles (raises/calls)."""
        return _af(self.raises_b, self.calls_b)

    @property
    def wtsd(self) -> float | None:
        """WTSD = showdowns / manos que vieron flop, sobre toda la tanda."""
        if self.n_saw_flop == 0:
            return None
        return self.n_showdowns / self.n_saw_flop


def play_session(
    agent_a: PokerAgent,
    agent_b: PokerAgent,
    name_a: str = "A",
    name_b: str = "B",
    seed: int | None = 42,
    starting_stack: int = STARTING_STACK,
    max_hands: int = 200,
    button_a_first: bool = True,
) -> SessionResult:
    """Juega UNA partida al KO: los stacks se arrastran de mano en mano.

    Termina cuando alguien no puede cubrir la ciega grande, o al llegar a
    max_hands (empate por límite, winner=None).

    El botón lo alterna solo el motor (_finalize lo rota al cerrar cada mano);
    aquí solo se fija quién lo lleva en la PRIMERA mano, que es de donde viene
    el sesgo de posición cuando las partidas son cortas.
    """
    pa = AgentPlayer(name_a, starting_stack, agent_a)
    pb = AgentPlayer(name_b, starting_stack, agent_b)

    gc = GameController(pa, pb, seed=seed)
    
    pa.bind(gc)
    pb.bind(gc)
    gc.dealer = pa if button_a_first else pb

    fichas_totales = pa.stack + pb.stack
    minimo = GameController.BIG_BLIND      # decisión: se corta bajo la ciega grande

    res = SessionResult(name_a, name_b, None, 0, pa.stack, pb.stack,
                        [pa.stack], [pb.stack])

    while res.hands < max_hands and pa.stack >= minimo and pb.stack >= minimo:
        gc.start_hand()
        while gc.step() != HAND_DONE:
            pass
        res.hands += 1
        res.stacks_a.append(pa.stack)
        res.stacks_b.append(pb.stack)
        res._record_hand_metrics(gc.action_log, gc.result)

        if pa.stack + pb.stack != fichas_totales:
            raise RuntimeError(
                f"FUGA DE FICHAS en la mano {res.hands} de la partida "
                f"(semilla {seed}): {name_a}={pa.stack} + {name_b}={pb.stack} "
                f"= {pa.stack + pb.stack}, deberían ser {fichas_totales}. "
                f"El arrastre de stacks tiene un bug: resultado no válido."
            )

    res.final_stack_a, res.final_stack_b = pa.stack, pb.stack
    if pa.stack < minimo and pb.stack < minimo:
        # Solo alcanzable con stacks iniciales diminutos; decide quien tenga más.
        if pa.stack != pb.stack:
            res.winner = name_a if pa.stack > pb.stack else name_b
    elif pb.stack < minimo:
        res.winner = name_a
    elif pa.stack < minimo:
        res.winner = name_b
    # else: se agotó max_hands -> winner sigue a None (empate por límite)
    return res


def play_many_sessions(
    agent_a: PokerAgent,
    agent_b: PokerAgent,
    name_a: str = "A",
    name_b: str = "B",
    n_sessions: int = 100,
    seed: int = 42,
    starting_stack: int = STARTING_STACK,
    max_hands: int = 200,
    verbose: bool = True,
) -> SessionMatchResult:
    """Juega n_sessions partidas independientes y devuelve el marcador.

    Por eso n_sessions se redondea hacia arriba al siguiente par.
    """
    if n_sessions < 1:
        raise ValueError(f"n_sessions debe ser >= 1, recibido {n_sessions}")
    if name_a == name_b:
        # El marcador se atribuye por nombre; con nombres iguales toda partida
        # ganada se le contaría a A. Mejor fallar que publicar un 100 %.
        raise ValueError(
            f"Los dos agentes se llaman '{name_a}'. En modo partidas los "
            f"nombres tienen que ser distintos para poder atribuir el marcador."
        )
    n_pares = (n_sessions + 1) // 2
    n_sessions = n_pares * 2

    # Semillas decorreladas y reproducibles a partir de la semilla base.
    rng = random.Random(seed)
    semillas = [rng.randrange(2 ** 31) for _ in range(n_pares)]

    res = SessionMatchResult(name_a, name_b, n_sessions, starting_stack,
                             max_hands, seed)
    t0 = time.perf_counter()
    aviso_cada = max(n_pares // 10, 1)

    for i, s in enumerate(semillas):
        # Segunda vuelta = espejo: B ocupa el asiento 0, así que recibe las
        # cartas y la posición que tuvo A en la primera.
        for a_en_asiento_0 in (True, False):
            args = ((agent_a, agent_b, name_a, name_b) if a_en_asiento_0
                    else (agent_b, agent_a, name_b, name_a))
            sr = play_session(*args, seed=s, starting_stack=starting_stack,
                              max_hands=max_hands, button_a_first=True)

            res.durations.append(sr.hands)
            # sr trae su propio name_a/name_b: en la segunda vuelta de cada
            # par (espejo) el asiento 0 lo ocupa el agente B real, así que hay
            # que mapear los contadores por nombre, no por asiento.
            if sr.name_a == name_a:
                res.raises_a += sr.raises_a
                res.calls_a  += sr.calls_a
                res.raises_b += sr.raises_b
                res.calls_b  += sr.calls_b
            else:
                res.raises_a += sr.raises_b
                res.calls_a  += sr.calls_b
                res.raises_b += sr.raises_a
                res.calls_b  += sr.calls_a
            res.n_saw_flop  += sr.n_saw_flop
            res.n_showdowns += sr.n_showdowns

            if sr.winner is None:
                res.draws += 1
            elif sr.winner == name_a:
                res.wins_a += 1
            elif sr.winner == name_b:
                res.wins_b += 1
            else:
                raise RuntimeError(
                    f"Ganador '{sr.winner}' no coincide con '{name_a}' ni "
                    f"con '{name_b}'."
                )

        if verbose and (i + 1) % aviso_cada == 0:
            hechas = (i + 1) * 2
            rate = (time.perf_counter() - t0) / hechas
            print(f"  {hechas}/{n_sessions} partidas  |  "
                  f"{name_a} {res.wins_a} - {res.wins_b} {name_b}  |  "
                  f"ETA {rate * (n_sessions - hechas) / 60:.1f} min", flush=True)

    res.elapsed_s = time.perf_counter() - t0

    total = res.wins_a + res.wins_b + res.draws
    if total != res.n_sessions:
        raise RuntimeError(
            f"Marcador incoherente: {res.wins_a}+{res.wins_b}+{res.draws} "
            f"= {total} != {res.n_sessions} partidas jugadas."
        )
    return res


def session_summary(res: SessionMatchResult) -> str:
    corte = f"{res.draws} ({res.draws / res.n_sessions:.0%})" if res.draws else "0"
    lines = [
        "=" * 62,
        f"  {res.name_a}  vs  {res.name_b}      ({res.n_sessions} partidas al KO)",
        "=" * 62,
        f"  Marcador               : {res.name_a} {res.wins_a} - "
        f"{res.wins_b} {res.name_b}   (empates {res.draws})",
        f"  Cortadas por límite    : {corte}  (max_hands={res.max_hands})",
        f"  Stack inicial          : {res.starting_stack} "
        f"({res.starting_stack / GameController.BIG_BLIND:.0f} ciegas grandes)",
        f"  AF (todas las calles)  : {res.name_a} {_fmt_af(res.af_a)} | "
        f"{res.name_b} {_fmt_af(res.af_b)}",
        f"  WTSD (por mano)        : {_fmt_wtsd(res.wtsd)}  "
        f"({res.n_showdowns}/{res.n_saw_flop} manos con flop)",
        f"  Tiempo                 : {res.elapsed_s / 60:.1f} min",
        "=" * 62,
    ]
    return "\n".join(lines)


def summary(res: MatchResult) -> str:
    ns = "" if res.significant else "   n.s."
    lines = [
        "=" * 62,
        f"  {res.name_a}  vs  {res.name_b}      ({res.hands} manos)",
        "=" * 62,
        f"  Fichas netas {res.name_a:<10}: {res.net_chips_a:+d}",
        f"  bb/100                 : {res.bb_per_100:+.2f} +/- {res.ci95:.2f}{ns}   "
        f"(> 0 significa que gana {res.name_a})",
        f"  Manos ganadas          : {res.name_a} {res.wins_a} | "
        f"{res.name_b} {res.wins_b} | empates {res.ties}",
        f"  AF postflop            : {res.name_a} {_fmt_af(res.af_a)} | "
        f"{res.name_b} {_fmt_af(res.af_b)}",
        f"  WTSD (por mano)        : {_fmt_wtsd(res.wtsd)}  "
        f"({res.n_showdowns}/{res.n_saw_flop} manos con flop)",
        f"  Tiempo                 : {res.elapsed_s/60:.1f} min "
        f"({res.elapsed_s/max(res.hands,1):.2f} s/mano)",
        "",
        "  Acciones:",
    ]
    for name, acts in ((res.name_a, res.actions_a), (res.name_b, res.actions_b)):
        total = sum(acts.values()) or 1
        dist = "  ".join(f"{k} {v/total:.0%}" for k, v in sorted(acts.items()))
        lines.append(f"    {name:<10}: {dist}")
    return "\n".join(lines)
