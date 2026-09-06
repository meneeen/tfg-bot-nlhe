from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any

from treys import Card as TreysCard

from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:11434/api/chat"

_ACTION_MAP: dict[str, ActionType] = {
    "fold"  : ActionType.FOLD,
    "check" : ActionType.CHECK,
    "call"  : ActionType.CALL,
    "raise" : ActionType.RAISE,
    "all_in": ActionType.ALL_IN,
}

_KIND_PARSE   = "parse"
_KIND_INVALID = "invalid"
_KIND_NETWORK = "network"
_RETRYABLE    = (_KIND_PARSE, _KIND_INVALID)


# ═════════════════════════════ LLMAgent ══════════════════════════════════════

class LLMAgent(PokerAgent):
    def __init__(
        self,
        model: str = "llama3.1",
        include_reasoning: bool = True,
        temperature: float = 0.7,
        fallback_agent: PokerAgent | None = None,
        timeout: int = 240,
        keep_alive: str = "30m",
    ) -> None:
        self.model             = model
        self.include_reasoning = include_reasoning
        self.temperature       = temperature
        self.fallback_agent    = fallback_agent
        self.timeout           = timeout
        self.keep_alive        = keep_alive

        self.last_reasoning: str = ""

        self.last_server_model: str = ""

        self.n_calls           = 0    # peticiones lanzadas (reintentos incluidos)
        self.n_output_tokens   = 0    # tokens generados (eval_count de Ollama)
        self.n_parse_failures  = 0    # respuestas de las que no se pudo sacar JSON
        self.n_invalid_actions = 0    # JSON válido pero con acción desconocida
        self.n_retries         = 0    # decisiones que necesitaron un reintento
        self.n_fallbacks       = 0    # decisiones que acabaron en fallback
        self._total_latency    = 0.0  # segundos acumulados en llamadas


    def decide_action(self, state: GameState) -> AgentAction:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user",   "content": self._build_prompt(state)},
        ]

        action, kind, reason = self._attempt(messages, state)
        if action is not None:
            return action

        if kind in _RETRYABLE:
            self.n_retries += 1
            logger.info("LLMAgent: reintento tras fallo de %s (%s)", kind, reason)
            messages = messages + [
                {"role": "user", "content": self._build_retry_message(reason)},
            ]
            action, kind, reason = self._attempt(messages, state)
            if action is not None:
                return action

        return self._fallback(state, reason=reason)

    def stats(self) -> dict[str, Any]:
        calls     = self.n_calls
        malformed = self.n_parse_failures + self.n_invalid_actions
        return {
            "model"             : self.model,
            "server_model"      : self.last_server_model or "(sin respuesta aún)",
            "temperature"       : self.temperature,
            "n_calls"           : calls,
            "avg_output_tokens" : (self.n_output_tokens / calls) if calls else 0.0,
            "n_parse_failures"  : self.n_parse_failures,
            "n_invalid_actions" : self.n_invalid_actions,
            "n_retries"         : self.n_retries,
            "n_fallbacks"       : self.n_fallbacks,
            "malformed_rate"    : (malformed / calls) if calls else 0.0,
            "avg_latency_s"     : (self._total_latency / calls) if calls else 0.0,
        }

    def _attempt(
        self, messages: list[dict[str, str]], state: GameState
    ) -> tuple[AgentAction | None, str, str]:
        try:
            raw = self._timed_call(messages)
        except Exception as exc:
            logger.error(
                "LLMAgent: fallo de red al contactar Ollama (%s): %s",
                type(exc).__name__, exc,
            )
            return None, _KIND_NETWORK, f"network error: {type(exc).__name__}"

        return self._parse_and_validate(raw, state)

    def _timed_call(self, messages: list[dict[str, str]]) -> str:
        """Envuelve _call_ollama para contabilizar llamadas y latencia."""
        self.n_calls += 1
        t0 = time.perf_counter()
        try:
            return self._call_ollama(messages)
        finally:
            self._total_latency += time.perf_counter() - t0

    def _schema(self) -> str:
        if self.include_reasoning:
            return (
                '{"action": "fold|check|call|raise|all_in", '
                '"amount": <integer or null>, '
                '"reasoning": "<one-sentence explanation>"}'
            )
        return (
            '{"action": "fold|check|call|raise|all_in", '
            '"amount": <integer or null>}'
        )

    def _build_system_prompt(self) -> str:
        return "\n".join([
            "You are an expert No-Limit Texas Hold'em heads-up poker player.",
            "Analyze the game state and choose the single best action.",
            "",
            "CRITICAL: respond with ONLY a valid JSON object — no other text,",
            "no markdown, no explanation outside the JSON.",
            f"Required format: {self._schema()}",
            "",
            "Action semantics:",
            "  fold   — surrender the hand.",
            "  check  — pass without paying (ONLY valid when amount_to_call == 0).",
            "  call   — match the current bet; set amount to null.",
            "  raise  — bet or raise; 'amount' is the TOTAL raise-to this street",
            "           (not the increment). Must be in [min_raise_to, max_raise_to].",
            "  all_in — go all-in; set amount to null.",
            "",
            "Set 'amount' to null for fold, check, call, and all_in.",
        ])

    def _build_retry_message(self, reason: str) -> str:
        return "\n".join([
            f"Your previous response was invalid ({reason}).",
            "Respond with ONLY a valid JSON object — no prose, no markdown.",
            f"Required format: {self._schema()}",
        ])

    def _build_prompt(self, state: GameState) -> str:
        hole  = _cards_to_str(state.hole_cards)
        board = _cards_to_str(state.community_cards) if state.community_cards else "(none yet)"

        if state.to_call > 0:
            pot_odds     = state.to_call / (state.pot + state.to_call)
            pot_odds_str = (
                f"{pot_odds:.1%} "
                f"(call {state.to_call} into total pot of {state.pot + state.to_call})"
            )
        else:
            pot_odds_str = "N/A (no bet to call, you can check)"

        pos_label = (
            "BTN / Small Blind — acts first preflop, last postflop"
            if state.position.value == "btn"
            else "BB  / Big Blind  — acts last preflop, first postflop"
        )

        history_lines: list[str] = [
            "  {}: {}{}  [{}]".format(
                r.player_name,
                r.action_type.value,
                f" {r.amount}" if r.amount > 0 else "",
                r.street.name,
            )
            for r in state.action_history
        ] or ["  (none yet)"]

        return "\n".join([
            "=== POKER DECISION ===",
            f"Your hole cards   : {hole}",
            f"Community cards   : {board}",
            f"Street            : {state.street.name}",
            f"Position          : {pos_label}",
            "",
            f"Pot               : {state.pot}",
            f"Your stack        : {state.my_stack}",
            f"Opponent's stack  : {state.opp_stack}",
            "",
            f"Amount to call    : {state.to_call}",
            f"Pot odds          : {pot_odds_str}",
            f"Min raise to      : {state.min_raise_to}",
            f"Max raise (all-in): {state.max_raise_to}",
            f"Blinds            : {state.small_blind}/{state.big_blind}",
            "",
            "Hand history (this hand):",
            *history_lines,
            "",
            "What is your action?",
        ])

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model"     : self.model,
            "messages"  : messages,
            "stream"    : False,
            # Modo JSON nativo: Ollama restringe la generación a JSON válido.
            "format"    : "json",
            "options"   : {"temperature": self.temperature},
            # Evita que Ollama descargue el modelo a los 5 min (defecto) y
            # obligue a pagar de nuevo la carga en frío a media tanda.
            "keep_alive": self.keep_alive,
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            _DEFAULT_URL,
            data    = data,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            
        self.last_server_model = body.get("model", "")
  
        self.n_output_tokens += int(body.get("eval_count", 0) or 0)

        return body["message"]["content"]

    # ─────────────────────────── parseo ──────────────────────────────────────

    def _parse_and_validate(
        self, raw: str, state: GameState
    ) -> tuple[AgentAction | None, str, str]:

        json_str = _extract_json(raw)
        if json_str is None:
            self.n_parse_failures += 1
            logger.warning(
                "LLMAgent: no se encontró JSON en la respuesta (primeros 300 chars): %r",
                raw[:300],
            )
            return None, _KIND_PARSE, "no JSON in response"

        try:
            data: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self.n_parse_failures += 1
            logger.warning(
                "LLMAgent: JSONDecodeError — %s | fragmento: %r", exc, json_str[:300]
            )
            return None, _KIND_PARSE, "JSON decode error"

        reasoning = data.get("reasoning", "")
        self.last_reasoning = reasoning if isinstance(reasoning, str) else str(reasoning)

        action_str  = str(data.get("action", "")).strip().lower()
        action_type = _ACTION_MAP.get(action_str)
        if action_type is None:
            self.n_invalid_actions += 1
            logger.warning("LLMAgent: acción desconocida %r en la respuesta", action_str)
            return None, _KIND_INVALID, f"unknown action '{action_str}'"

        raw_amount = data.get("amount")
        try:
            amount = int(raw_amount) if raw_amount is not None else 0
        except (ValueError, TypeError):
            amount = 0

        return _sanitize(AgentAction(action_type, amount), state), "", ""

    def _fallback(self, state: GameState, reason: str = "") -> AgentAction:
        self.n_fallbacks += 1
        logger.debug("LLMAgent: fallback — %s", reason or "sin motivo")

        if self.fallback_agent is not None:
            name = type(self.fallback_agent).__name__
            self.last_reasoning = f"[fallback -> {name}: {reason}]"
            return _sanitize(self.fallback_agent.decide_action(state), state)

        self.last_reasoning = f"[fallback: {reason}]" if reason else "[fallback]"
        if state.to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.FOLD)
    

def _cards_to_str(cards: tuple[int, ...]) -> str:
    return " ".join(TreysCard.int_to_str(c) for c in cards)


def _extract_json(text: str) -> str | None:

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

    return None


def _sanitize(action: AgentAction, state: GameState) -> AgentAction:
    t = action.action_type

    if t == ActionType.FOLD and state.to_call == 0:
        return AgentAction(ActionType.CHECK)

    if t == ActionType.CHECK and state.to_call > 0:
        return AgentAction(ActionType.CALL)

    if t == ActionType.ALL_IN and state.my_stack <= state.to_call:
        # stack exactamente igual o menor que la deuda: es un call, no un raise
        return AgentAction(ActionType.CALL)

    if t == ActionType.RAISE:
        return _clamp_raise(state, action.amount)

    return action


def _clamp_raise(state: GameState, amount: int) -> AgentAction:
    """Ajusta el importe al rango legal ``[min_raise_to, max_raise_to]``."""
    if state.max_raise_to <= state.min_raise_to:
        # Solo cabe ir all-in; no hay rango válido para un raise estándar
        return AgentAction(ActionType.ALL_IN)
    amount = max(amount, state.min_raise_to)
    if amount >= state.max_raise_to:
        return AgentAction(ActionType.ALL_IN)
    return AgentAction(ActionType.RAISE, amount)
