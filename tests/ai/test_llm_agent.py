"""
Tests para LLMAgent — la llamada HTTP a Ollama está mockeada.

No se requiere que Ollama esté corriendo para ejecutar estos tests.

Cobertura:
  1. _extract_json               — extracción de JSON de texto libre.
  2. Respuestas válidas          — acción correcta por cada tipo de respuesta.
  3. Parseo robusto              — markdown fences, texto alrededor, campos extra.
  4. Fallback                    — JSON inválido, acción desconocida, error de red.
  5. Sanitización                — LLM devuelve acción ilegal → se corrige.
  6. include_reasoning           — afecta al system prompt y al guardado de reasoning.
  7. Prompt                      — el prompt contiene la información esperada.

Ejecutar con:
    pytest tests/ai/test_llm_agent.py -v
"""
from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest
from treys import Card

from src.ai.base_agent import AgentAction, ActionType, GameState, Position
from src.ai.llm_agent import LLMAgent, _extract_json
from src.engine.controller import Street


# ─────────────────────────────── helpers ─────────────────────────────────────

def make_state(**overrides) -> GameState:
    """GameState con valores por defecto razonables."""
    defaults: dict = dict(
        hole_cards         = (Card.new("As"), Card.new("Kh")),
        community_cards    = (),
        pot                = 100,
        my_stack           = 980,
        opp_stack          = 980,
        my_bet_this_street = 20,
        to_call            = 20,
        min_raise_to       = 40,
        max_raise_to       = 1000,
        position           = Position.BTN,
        street             = Street.PREFLOP,
        action_history     = (),
        big_blind          = 20,
        small_blind        = 10,
    )
    defaults.update(overrides)
    return GameState(**defaults)


def llm_json(action: str, amount=None, reasoning: str = "") -> str:
    """Construye el string que _call_ollama devolvería para respuesta limpia."""
    d: dict = {"action": action, "amount": amount}
    if reasoning:
        d["reasoning"] = reasoning
    return json.dumps(d)


# ═══════════════════════ 1. _extract_json ════════════════════════════════════

class TestExtractJson:
    """Tests unitarios de la función de extracción (no necesita mock)."""

    def test_clean_json_object(self):
        raw = '{"action": "fold", "amount": null}'
        assert _extract_json(raw) == raw

    def test_text_before_json(self):
        raw = 'Sure, here is my pick: {"action": "call", "amount": null}'
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result)["action"] == "call"

    def test_text_after_json(self):
        raw = '{"action": "raise", "amount": 80} That is my decision.'
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result)["amount"] == 80

    def test_text_before_and_after(self):
        raw = 'Let me analyze... {"action": "check", "amount": null} End.'
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result)["action"] == "check"

    def test_markdown_fence_with_json_tag(self):
        raw = '```json\n{"action": "fold", "amount": null}\n```'
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result)["action"] == "fold"

    def test_markdown_fence_without_tag(self):
        raw = '```\n{"action": "call", "amount": null}\n```'
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result)["action"] == "call"

    def test_no_json_returns_none(self):
        assert _extract_json("There is no JSON here at all.") is None

    def test_empty_string_returns_none(self):
        assert _extract_json("") is None

    def test_json_with_reasoning_field(self):
        raw = '{"action": "raise", "amount": 120, "reasoning": "strong hand"}'
        result = _extract_json(raw)
        assert result is not None
        data = json.loads(result)
        assert data["reasoning"] == "strong hand"


# ═══════════════════════ 2. Respuestas válidas ════════════════════════════════

class TestValidResponses:
    """El agente procesa respuestas bien formadas del LLM correctamente."""

    def test_fold(self):
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("fold")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_call(self):
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("call")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CALL

    def test_check(self):
        state = make_state(to_call=0)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("check")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CHECK

    def test_raise_valid_amount(self):
        state = make_state(to_call=0, min_raise_to=40, max_raise_to=1000)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("raise", 80)):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.RAISE
        assert action.amount == 80

    def test_all_in(self):
        state = make_state(to_call=20, my_stack=500)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("all_in")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.ALL_IN

    def test_reasoning_stored(self):
        agent = LLMAgent()
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama",
                          return_value=llm_json("fold", reasoning="pair is weak")):
            agent.decide_action(state)
        assert agent.last_reasoning == "pair is weak"

    def test_reasoning_empty_when_absent_from_json(self):
        agent = LLMAgent()
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("fold")):
            agent.decide_action(state)
        assert agent.last_reasoning == ""


# ═══════════════════════ 3. Parseo robusto ════════════════════════════════════

class TestRobustParsing:
    """El agente tolera formatos de respuesta comunes de los LLM."""

    def test_markdown_fence_json(self):
        raw = '```json\n{"action": "fold", "amount": null, "reasoning": "weak"}\n```'
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_text_surrounding_json(self):
        raw = 'After analysis: {"action": "call", "amount": null} — final answer.'
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CALL

    def test_amount_as_string_is_coerced(self):
        """El LLM puede serializar el número como string; debemos aceptarlo."""
        raw = '{"action": "raise", "amount": "80", "reasoning": "value bet"}'
        state = make_state(to_call=0, min_raise_to=40, max_raise_to=1000)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.RAISE
        assert action.amount == 80

    def test_extra_fields_in_json_ignored(self):
        raw = '{"action": "check", "amount": null, "confidence": 0.9, "ev": "+1.2"}'
        state = make_state(to_call=0)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CHECK

    def test_null_amount_treated_as_zero(self):
        raw = '{"action": "call", "amount": null}'
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CALL
        assert action.amount == 0


# ═══════════════════════ 4. Fallback ══════════════════════════════════════════

class TestFallback:
    """Ante fallos el agente devuelve una acción segura y loguea el error."""

    def test_invalid_json_with_bet_pending(self):
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value="this is not json"):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_invalid_json_free_check(self):
        state = make_state(to_call=0)
        with patch.object(LLMAgent, "_call_ollama", return_value="not json"):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CHECK

    def test_unknown_action_name(self):
        """'bet' no está en el enum → fallback."""
        raw = '{"action": "bet", "amount": 50}'
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_empty_action_field(self):
        raw = '{"action": "", "amount": null}'
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_network_error_falls_back(self):
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("connection refused")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_timeout_falls_back(self):
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=TimeoutError("timed out")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_empty_response_falls_back(self):
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=""):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_fallback_sets_last_reasoning(self):
        agent = LLMAgent()
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value="not json"):
            agent.decide_action(state)
        assert agent.last_reasoning.startswith("[fallback:")


# ═══════════════════════ 5. Sanitización ══════════════════════════════════════

class TestSanitization:
    """El agente corrige acciones semánticamente ilegales devueltas por el LLM."""

    def test_check_when_must_call_becomes_call(self):
        state = make_state(to_call=30)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("check")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CALL

    def test_fold_when_free_becomes_check(self):
        state = make_state(to_call=0)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("fold")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CHECK

    def test_raise_below_min_clamped_to_min(self):
        """El LLM propone 10, el mínimo es 40 → se clampea a 40."""
        state = make_state(to_call=0, min_raise_to=40, max_raise_to=1000)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("raise", 10)):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.RAISE
        assert action.amount >= state.min_raise_to

    def test_raise_above_max_becomes_all_in(self):
        """El LLM propone 9999, el máximo es 200 → ALL_IN."""
        state = make_state(to_call=0, min_raise_to=40, max_raise_to=200, my_stack=200)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("raise", 9999)):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.ALL_IN

    def test_raise_when_max_below_min_becomes_all_in(self):
        """Stack corto: max_raise_to < min_raise_to → solo cabe ALL_IN."""
        state = make_state(
            to_call=0, min_raise_to=40, max_raise_to=30, my_stack=30
        )
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("raise", 40)):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.ALL_IN

    def test_all_in_without_extra_chips_becomes_call(self):
        """stack == to_call: ir all-in solo cubre la deuda, no la supera → CALL."""
        state = make_state(to_call=100, my_stack=100)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("all_in")):
            action = LLMAgent().decide_action(state)
        assert action.action_type == ActionType.CALL

    def test_raise_amount_never_negative(self):
        state = make_state(to_call=0, min_raise_to=40, max_raise_to=1000)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("raise", -50)):
            action = LLMAgent().decide_action(state)
        # −50 → max(−50, 40) = 40 → válido
        assert action.amount >= 0


# ═══════════════════════ 6. include_reasoning ═════════════════════════════════

class TestIncludeReasoning:

    def test_system_prompt_contains_reasoning_when_enabled(self):
        agent = LLMAgent(include_reasoning=True)
        assert "reasoning" in agent._build_system_prompt()

    def test_system_prompt_omits_reasoning_when_disabled(self):
        agent = LLMAgent(include_reasoning=False)
        assert "reasoning" not in agent._build_system_prompt()

    def test_reasoning_saved_even_if_not_requested(self):
        """Si el LLM devuelve 'reasoning' aunque no se le pidió, lo almacenamos."""
        agent = LLMAgent(include_reasoning=False)
        raw = '{"action": "fold", "amount": null, "reasoning": "surprise field"}'
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=raw):
            agent.decide_action(state)
        assert agent.last_reasoning == "surprise field"

    def test_last_reasoning_resets_each_call(self):
        """El reasoning de una llamada no contamina la siguiente."""
        agent = LLMAgent()
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama",
                          return_value=llm_json("fold", reasoning="first")):
            agent.decide_action(state)
        assert agent.last_reasoning == "first"

        with patch.object(LLMAgent, "_call_ollama",
                          return_value=llm_json("fold")):   # sin reasoning
            agent.decide_action(state)
        assert agent.last_reasoning == ""


# ═══════════════════════ 7. Construcción del prompt ═══════════════════════════

class TestPromptConstruction:
    """El prompt contiene toda la información relevante del GameState."""

    AGENT = LLMAgent()

    def test_prompt_contains_hole_cards(self):
        state = make_state(hole_cards=(Card.new("As"), Card.new("Kh")))
        prompt = self.AGENT._build_prompt(state)
        assert "As" in prompt or "Kh" in prompt

    def test_prompt_contains_pot_amount(self):
        state = make_state(pot=250)
        prompt = self.AGENT._build_prompt(state)
        assert "250" in prompt

    def test_prompt_contains_stacks(self):
        state = make_state(my_stack=750, opp_stack=1250)
        prompt = self.AGENT._build_prompt(state)
        assert "750" in prompt
        assert "1250" in prompt

    def test_prompt_contains_pot_odds_when_facing_bet(self):
        # to_call=50, pot=100 → pot_odds=50/150≈33.3%
        state = make_state(pot=100, to_call=50)
        prompt = self.AGENT._build_prompt(state)
        assert "33.3%" in prompt

    def test_prompt_says_no_pot_odds_when_free(self):
        state = make_state(to_call=0)
        prompt = self.AGENT._build_prompt(state)
        assert "N/A" in prompt

    def test_prompt_contains_street_name(self):
        state = make_state(street=Street.RIVER)
        prompt = self.AGENT._build_prompt(state)
        assert "RIVER" in prompt

    def test_prompt_contains_min_and_max_raise(self):
        state = make_state(min_raise_to=80, max_raise_to=550)
        prompt = self.AGENT._build_prompt(state)
        assert "80" in prompt
        assert "550" in prompt

    def test_prompt_contains_community_cards_when_present(self):
        board = (Card.new("Ac"), Card.new("Kd"), Card.new("Qh"))
        state = make_state(community_cards=board, street=Street.FLOP)
        prompt = self.AGENT._build_prompt(state)
        # Al menos una de las cartas del tablero aparece en el prompt
        assert any(c in prompt for c in ["Ac", "Kd", "Qh"])

    def test_prompt_says_none_yet_when_no_board(self):
        state = make_state(community_cards=(), street=Street.PREFLOP)
        prompt = self.AGENT._build_prompt(state)
        assert "none yet" in prompt.lower()


# ═══════════════════════ 8. Payload de Ollama ═════════════════════════════════

class TestOllamaPayload:
    """El payload usa el modo JSON nativo y la temperatura configurable."""

    def _capture_payload(self, agent: LLMAgent) -> dict:
        """Ejecuta una decisión interceptando urlopen y devuelve el payload."""
        captured: dict = {}

        class _FakeResp:
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): return False
            def read(self_inner):
                body = {"message": {"content": llm_json("call")}}
                return json.dumps(body).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            captured.update(json.loads(req.data.decode("utf-8")))
            return _FakeResp()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            agent.decide_action(make_state(to_call=20))
        return captured

    def test_format_json_is_requested(self):
        payload = self._capture_payload(LLMAgent())
        assert payload["format"] == "json", (
            "Falta el modo JSON nativo de Ollama en el payload"
        )

    def test_default_temperature_is_07(self):
        payload = self._capture_payload(LLMAgent())
        assert payload["options"]["temperature"] == 0.7

    def test_temperature_is_configurable(self):
        payload = self._capture_payload(LLMAgent(temperature=0.0))
        assert payload["options"]["temperature"] == 0.0

    def test_messages_carry_system_and_user(self):
        payload = self._capture_payload(LLMAgent())
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]

    def test_stream_is_disabled(self):
        payload = self._capture_payload(LLMAgent())
        assert payload["stream"] is False


# ═══════════════════════ 9. Reintento ═════════════════════════════════════════

class TestRetry:
    """Un único reintento ante parseo fallido o acción inválida."""

    def test_bad_json_then_valid_succeeds_on_retry(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=["no soy json", llm_json("call")]) as m:
            action = agent.decide_action(make_state(to_call=20))
        assert action.action_type == ActionType.CALL
        assert m.call_count == 2
        assert agent.n_retries == 1
        assert agent.n_fallbacks == 0       # el reintento salvó la decisión

    def test_invalid_action_then_valid_succeeds_on_retry(self):
        agent = LLMAgent()
        bad = '{"action": "teleport", "amount": null}'
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=[bad, llm_json("fold")]):
            action = agent.decide_action(make_state(to_call=20))
        assert action.action_type == ActionType.FOLD
        assert agent.n_retries == 1
        assert agent.n_invalid_actions == 1

    def test_retry_message_repeats_the_schema(self):
        agent = LLMAgent()
        seen: list = []

        def spy(self_inner, messages):
            seen.append(messages)
            return "sigue sin ser json" if len(seen) == 1 else llm_json("call")

        with patch.object(LLMAgent, "_call_ollama", spy):
            agent.decide_action(make_state(to_call=20))

        retry_messages = seen[1]
        assert len(retry_messages) == 3, "El reintento debe añadir un mensaje de usuario"
        extra = retry_messages[-1]
        assert extra["role"] == "user"
        assert "invalid" in extra["content"].lower()
        assert "action" in extra["content"], "El reintento no repite el esquema"

    def test_only_one_retry_then_fallback(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          return_value="nunca es json") as m:
            action = agent.decide_action(make_state(to_call=20))
        assert m.call_count == 2, "Debe reintentar UNA vez, no entrar en bucle"
        assert agent.n_retries == 1
        assert agent.n_fallbacks == 1
        assert action.action_type == ActionType.FOLD    # fallback sin agente

    def test_network_error_does_not_retry(self):
        """Repetirle el esquema al modelo no arregla una red caída."""
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")) as m:
            agent.decide_action(make_state(to_call=20))
        assert m.call_count == 1, "Un fallo de red no debe reintentarse"
        assert agent.n_retries == 0
        assert agent.n_fallbacks == 1


# ═══════════════════════ 10. fallback_agent ═══════════════════════════════════

class _StubFallback:
    """Agente de reserva de prueba: siempre sube al mínimo legal."""

    def __init__(self):
        self.calls = 0

    def decide_action(self, state: GameState) -> AgentAction:
        self.calls += 1
        return AgentAction(ActionType.RAISE, state.min_raise_to)


class TestFallbackAgent:

    def test_without_fallback_agent_folds_facing_a_bet(self):
        """Comportamiento por defecto: tira la mano ante un timeout."""
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")):
            action = agent.decide_action(make_state(to_call=20))
        assert action.action_type == ActionType.FOLD

    def test_fallback_agent_is_used_instead_of_folding(self):
        stub  = _StubFallback()
        agent = LLMAgent(fallback_agent=stub)
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")):
            action = agent.decide_action(make_state(to_call=20))
        assert stub.calls == 1
        assert action.action_type == ActionType.RAISE, (
            "Con fallback_agent no debe tirarse una mano potencialmente ganadora"
        )

    def test_fallback_agent_used_after_failed_retry(self):
        stub  = _StubFallback()
        agent = LLMAgent(fallback_agent=stub)
        with patch.object(LLMAgent, "_call_ollama", return_value="basura"):
            action = agent.decide_action(make_state(to_call=20))
        assert agent.n_retries == 1
        assert stub.calls == 1
        assert action.action_type == ActionType.RAISE

    def test_fallback_agent_output_is_sanitized(self):
        """Ni el agente de reserva puede colar una acción ilegal."""

        class _IllegalFallback:
            def decide_action(self, state):
                return AgentAction(ActionType.RAISE, 999_999)   # supera el máximo

        agent = LLMAgent(fallback_agent=_IllegalFallback())
        state = make_state(to_call=20, max_raise_to=500)
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")):
            action = agent.decide_action(state)
        assert action.action_type == ActionType.ALL_IN     # clampeado

    def test_fallback_agent_reflected_in_last_reasoning(self):
        agent = LLMAgent(fallback_agent=_StubFallback())
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")):
            agent.decide_action(make_state(to_call=20))
        assert "_StubFallback" in agent.last_reasoning


# ═══════════════════════ 11. Instrumentación ══════════════════════════════════

class TestStats:

    def test_counters_start_at_zero(self):
        s = LLMAgent().stats()
        assert s["n_calls"] == 0
        assert s["n_parse_failures"] == 0
        assert s["n_invalid_actions"] == 0
        assert s["n_retries"] == 0
        assert s["n_fallbacks"] == 0
        assert s["malformed_rate"] == 0.0
        assert s["avg_latency_s"] == 0.0

    def test_n_calls_counts_every_request_including_retries(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=["basura", llm_json("call")]):
            agent.decide_action(make_state(to_call=20))
        assert agent.n_calls == 2      # intento + reintento

    def test_successful_call_increments_nothing_but_calls(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("call")):
            agent.decide_action(make_state(to_call=20))
        s = agent.stats()
        assert s["n_calls"] == 1
        assert s["n_parse_failures"] == 0
        assert s["n_invalid_actions"] == 0
        assert s["n_retries"] == 0
        assert s["n_fallbacks"] == 0
        assert s["malformed_rate"] == 0.0

    def test_parse_failures_counted_per_call(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama", return_value="nunca json"):
            agent.decide_action(make_state(to_call=20))
        # dos llamadas (intento + reintento), ambas ilegibles
        assert agent.n_parse_failures == 2
        assert agent.stats()["malformed_rate"] == 1.0

    def test_invalid_actions_counted(self):
        agent = LLMAgent()
        bad = '{"action": "dance", "amount": null}'
        with patch.object(LLMAgent, "_call_ollama", return_value=bad):
            agent.decide_action(make_state(to_call=20))
        assert agent.n_invalid_actions == 2
        assert agent.n_parse_failures == 0

    def test_network_errors_do_not_count_as_malformed(self):
        """Una red caída no es una respuesta malformada del modelo."""
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")):
            agent.decide_action(make_state(to_call=20))
        s = agent.stats()
        assert s["n_calls"] == 1           # la llamada se contabiliza…
        assert s["malformed_rate"] == 0.0  # …pero no como malformada
        assert s["n_fallbacks"] == 1

    def test_latency_is_recorded(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("call")):
            agent.decide_action(make_state(to_call=20))
        assert agent.stats()["avg_latency_s"] > 0.0

    def test_latency_recorded_even_on_network_error(self):
        agent = LLMAgent()
        with patch.object(LLMAgent, "_call_ollama",
                          side_effect=urllib.error.URLError("down")):
            agent.decide_action(make_state(to_call=20))
        assert agent.stats()["avg_latency_s"] > 0.0

    def test_counters_accumulate_across_decisions(self):
        agent = LLMAgent()
        state = make_state(to_call=20)
        with patch.object(LLMAgent, "_call_ollama", return_value=llm_json("call")):
            for _ in range(5):
                agent.decide_action(state)
        assert agent.n_calls == 5

    def test_stats_reports_model_and_temperature(self):
        s = LLMAgent(model="mistral", temperature=0.2).stats()
        assert s["model"] == "mistral"
        assert s["temperature"] == 0.2
