# TFG — Póker NLHE Heads-Up vs IA

Juego de póker *No-Limit Hold'em* (heads-up, 1 contra 1) con interfaz gráfica en
pygame y una arquitectura de agentes intercambiables: reglas, LLM local y
aprendizaje por refuerzo, todos detrás de la misma interfaz.

## Estado actual
- ✅ **Motor** completo: ciegas, preflop/flop/turn/river, apuestas, all-in con
  devolución de apuesta no igualada, regla de subida mínima (incremento real de
  NLHE, no la ciega grande), all-in incompleto que no reabre la acción,
  showdown y reparto. Reparto reproducible por semilla.
- ✅ **Interfaz común de agentes** (`PokerAgent` / `GameState`): cualquier
  estrategia se enchufa al motor sin tocarlo.
- ✅ **Agente de reglas** (`RuleBasedAgent`): Chen preflop, equity y pot odds
  postflop, con farol y semi-farol. Es el baseline de comparación.
- ✅ **Perfiles de estilo** (`rule_profiles.py`): una sola lógica de decisión
  parametrizada por un vector de umbrales, con seis arquetipos (`nit`,
  `baseline`, `tag`, `lag`, `station`, `maniac`). Un baseline único no mide
  fuerza de juego, solo lo bien que el rival explota ESE baseline; la batería
  cubre los ejes tight↔loose y passive↔aggressive.
- ✅ **Agente LLM** (`LLMAgent`): consulta un modelo local vía Ollama, con modo
  JSON nativo, reintento, agente de reserva e instrumentación (latencia, tasa
  de respuestas malformadas).
- ✅ **Agente RL** (`RLAgent` + `PokerEnv`): entorno Gymnasium y entrenamiento
  PPO con Stable-Baselines3, sparring contra reglas y luego self-play contra un
  pool de versiones anteriores.
- ✅ **Arnés de evaluación** (`duelo.py`), con dos modos que miden cosas
  distintas y no se sustituyen:
  - `--modo manos` (default): N manos independientes reseteando los stacks.
    Mide **bb/100** con intervalo de confianza. Es la métrica de precisión: mide
    cuánto se gana por mano, pero no simula una partida real.
  - `--modo partidas`: partidas **al KO**, con los stacks arrastrándose de mano
    en mano hasta que uno arruina al otro. Mide **partidas ganadas**, que es lo
    que decide un heads-up real, y acompaña el marcador con el **AF** de cada
    agente y el **WTSD** de la tanda. Ojo: aquí el AF cuenta TODAS las calles
    (preflop incluido), no solo postflop como en `--modo manos`; una partida al
    KO se juzga como paquete completo, y el all-in preflop es parte del juego,
    no ruido a descartar.

  > **Aviso sobre el modo partidas.** El stack inicial son 20 ciegas grandes.
  > Entre agentes agresivos (RL contra RL) la partida mediana dura **1 mano**:
  > se resuelve en un all-in, y el marcador acaba midiendo quién gana ese volado
  > más que quién juega mejor. Con `rb-baseline-hybrid` contra `rb-baseline-pure`
  > el marcador sale 45,5 % pese a que en bb/100 el primero gana por +72,9. El
  > informe avisa cuando la duración mediana baja de 2 manos. Stacks más
  > profundos exigirían reentrenar: `build_obs` normaliza pote y SPR por la
  > constante `STARTING_STACK`, y `RLAgent` rechaza modelos entrenados con otra.

## Estructura
```
tfg-poker/
├── src/
│   ├── engine/
│   │   ├── card.py            # baraja y evaluador (wrapper de treys)
│   │   ├── player.py          # Player / HumanPlayer
│   │   └── controller.py      # GameController: máquina de estados del juego
│   ├── ai/
│   │   ├── base_agent.py      # PokerAgent, GameState, AgentAction
│   │   ├── agent_player.py    # adaptador PokerAgent -> Player del motor
│   │   ├── hand_strength.py   # Chen preflop, equity postflop, proyectos
│   │   ├── rule_based_agent.py # motor de reglas (lógica, sin números)
│   │   ├── rule_profiles.py   # perfiles de estilo (números, sin lógica)
│   │   ├── llm_agent.py       # LLM local vía Ollama
│   │   ├── rl_agent.py        # carga un modelo PPO entrenado
│   │   └── rl/
│   │       ├── poker_env.py   # entorno Gymnasium
│   │       └── train.py       # entrenamiento PPO (fase 1 + self-play)
│   └── gui/
│       └── main_window.py     # interfaz pygame
├── tests/                     # 227 tests (motor, agentes, entorno RL)
├── play_vs_rb.py              # jugar contra el agente de reglas
├── play_vs_llm.py             # jugar contra el LLM
├── play_vs_rl.py              # jugar contra el agente RL entrenado
├── rl_vs_rl.py                # el RL contra sí mismo        -> log legible
├── rl_vs_llm.py               # el RL contra el LLM          -> log legible
└── requirements.txt
```

## Instalación
```bash
python -m venv venv
venv\Scripts\activate           # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Para el agente LLM hace falta además [Ollama](https://ollama.com) corriendo en
local y un modelo descargado (`ollama pull llama3.1`). Los modelos ocupan varios
GB: la variable de entorno `OLLAMA_MODELS` permite guardarlos en otro disco.

## Formas de ejecutarlo
```bash
# Jugar tú, en la interfaz gráfica
python play_vs_rb.py            # contra el agente de reglas (no necesita nada mas)
python play_vs_rb.py --list     # ver los perfiles de reglas disponibles
python play_vs_rb.py -p maniac  # elegir el estilo del rival de reglas
python play_vs_llm.py           # contra el LLM (precalienta Ollama antes de abrir)
python play_vs_rl.py            # contra el agente RL (requiere modelo entrenado)
python play_vs_rl.py --model models/rb-baseline-hybrid/rl_2m/modelo.zip

# Agente contra agente, con log legible mano a mano en logs/
python rl_vs_rl.py              # el RL contra sí mismo
python rl_vs_llm.py             # el RL contra el LLM

# Duelos medidos (bb/100 con intervalo de confianza) -> datos/duelos.json
python duelo.py reglas:tag reglas:station -n 50000
python duelo.py rl:models/rb-baseline-hybrid/rl_2m/modelo.zip reglas:maniac -n 50000

# Partidas al KO (marcador de partidas ganadas) -> datos/partidas.json
python duelo.py reglas:tag reglas:nit --modo partidas --sessions 500

# Entrenamiento y tests
python -m src.ai.rl.train                              # sparring: perfil baseline
python -m src.ai.rl.train --baseline-profile tag,lag   # contra varios perfiles
python -m src.ai.rl.train --baseline-profile all       # contra los seis
python -m pytest tests/                                # ejecutar los tests
```

Cada tanda de `--modo partidas` se añade a `datos/partidas.json` con los campos
`n_tanda`, `a`, `b`, `partidas`, `seed`, `ganadas_a`, `ganadas_b`, `empates`,
`af_a`, `af_b` y `wtsd`. Solo datos crudos: el % de victorias no se guarda
porque sale de `ganadas_a / partidas`, y un derivado guardado solo puede acabar
contradiciendo al dato que lo genera. `af_a`/`af_b` valen `null` si ese agente
no pagó ni una vez (división por cero), y `wtsd` vale `null` si ninguna mano
llegó al flop. El WTSD es **uno solo** por tanda, no uno por jugador: en
heads-up el showdown es propiedad de la mano — si lo hay, llegaron los dos.

Contra qué se entrena queda grabado en `modelo.meta.json` (`training_profiles`),
para poder distinguir dos modelos que solo se diferencian en su sparring.

## Cómo se juega (controles)
- **RETIRARSE**: te retiras (fold).
- **PASAR / IGUALAR**: pasa si no hay apuesta; iguala si la hay.
- **− / +** y **SUBIR**: ajusta el importe y sube.
- **ALL-IN**: apuesta todas tus fichas.
- Pulsa **SIGUIENTE MANO** para repartir la siguiente.

## Añadir un agente nuevo
Basta con subclasificar `PokerAgent` e implementar un método:

```python
from src.ai.base_agent import PokerAgent, GameState, AgentAction, ActionType

class MiAgente(PokerAgent):
    def decide_action(self, state: GameState) -> AgentAction:
        if state.to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.CALL)
```

El `GameState` trae cartas, bote, stacks, posición, historial de la mano y los
importes legales de subida. Para enfrentarlo a un humano en la GUI:

```python
from src.gui.main_window import main
main(MiAgente())
```
