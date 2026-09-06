import sys
import threading

import pygame
from src.engine.controller import GameController, STARTING_STACK
from src.engine.player import Action
from src.engine.card import card_to_str

# ───────────────────────── constantes ─────────────────────────

WIDTH, HEIGHT = 1024, 720
FPS           = 60
STEP_DELAY_MS = 650          # ritmo de juego automático (IA / reparto)

# ── Layout vertical ──────────────────────────────────────────
# Todas las coordenadas Y en un único sitio: cambiar una sola aquí evita los
# solapamientos que aparecían al ajustarlas dispersas por el código.
# El header debe cubrir la cresta de la elipse (y su rail dorado), o asoma un
# sliver de madera entre la barra y el fieltro: por eso HEADER_H > TABLE top.
HEADER_H       = 68
TABLE_RECT     = pygame.Rect(30, 64, WIDTH - 60, 556)   # 64 .. 620
AGENT_PANEL_CY = 108        # centro panel IA (74 .. 142): libre del rail dorado
AGENT_CARDS_Y  = 148        # alto 76  -> 148 .. 224
POT_CY         = 250        # píldora del bote (alto 38 -> 231 .. 269)
STREET_CY      = 284        # etiqueta de calle (alto 22 -> 273 .. 295)
BOARD_Y        = 302        # alto 98  -> 302 .. 400

# Toast de última acción: caja negra a la IZQUIERDA del fieltro, con pop-up.
# Se centra en x=176 y como mucho mide 234 px de ancho -> acaba en x=293,
# holgado respecto al borde izquierdo del board (x=308).
MSG_CX         = 172
MSG_CY         = 352        # alineado con el centro vertical del board
MSG_MAX_TEXT_W = 208        # ancho máximo del texto antes de partir en 2 líneas
MSG_MIN_W      = 196        # ancho mínimo: evita cajas enclenques con textos cortos
MSG_POP_MS     = 260        # duración de la animación de entrada
HUMAN_PANEL_CY = 464        # centro panel humano (430 .. 498)
HUMAN_CARDS_Y  = 506        # alto 102 -> 506 .. 608  (por encima de la barra)
ACTION_BAR_Y   = HEIGHT - 96   # 624 .. 720
BTN_Y          = HEIGHT - 72   # 648 .. 698
RESULT_CY      = 260        # banner de resultado (228 .. 292): sustituye bote+calle

# Tamaños de carta
CARD_W, CARD_H       = 72, 98     # board
HOLE_W, HOLE_H       = 76, 102    # cartas del humano
AGENT_CW, AGENT_CH   = 58, 76     # cartas de la IA

# Paleta
FELT_DARK   = (16,  68,  46)
FELT_LIGHT  = (26,  94,  64)
FELT_EDGE   = (10,  44,  30)
GOLD        = (212, 175, 55)
GOLD_DIM    = (150, 122, 40)
GOLD_BRIGHT = (255, 215, 80)
CREAM       = (242, 235, 215)
WHITE       = (248, 248, 246)
BLACK       = (20,  20,  24)
RED         = (190, 35,  45)
CARD_BACK   = (120, 22,  30)
CARD_BACK2  = (90,  16,  22)
CARD_BACK3  = (145, 30,  38)
SHADOW      = (0,   0,   0)
BTN         = (38,  38,  44)
BTN_HOVER   = (58,  58,  66)
BTN_DISABLE = (30,  30,  34)
TEXT_DIM    = (180, 180, 175)
WOOD        = (60,  38,  14)
WOOD_DARK   = (40,  24,   8)
WOOD_LIGHT  = (85,  54,  20)
HEADER_BG   = (10,  10,  14)
AMBER       = (255, 190, 50)

SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLOR  = {"s": BLACK, "c": BLACK, "h": RED, "d": RED}
RANK_DISP   = {"T": "10"}    # el resto se muestra tal cual

STREET_LABEL = {0: "PRE-FLOP", 3: "FLOP", 4: "TURN", 5: "RIVER"}


def _ease_out_back(t: float, s: float = 1.9) -> float:
    """Easing con ligero rebote: sobrepasa 1.0 y vuelve. Da sensación de 'pop'."""
    t -= 1.0
    return t * t * ((s + 1) * t + s) + 1.0


class PokerGUI:
    # Constante del motor. No es configurable: un RLAgent solo sabe jugar el
    # stack con el que se entrenó, así que la GUI juega siempre el estándar.
    STARTING_STACK = STARTING_STACK

    def __init__(self, opponent=None, agent_name: str = "IA"):
        pygame.init()
        pygame.display.set_caption(f"TFG — Póker NLHE Heads-Up vs {agent_name}")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()

        # Fuentes serif para el aire de casino…
        self.f_title = pygame.font.SysFont("georgia,timesnewroman,serif", 28, bold=True)
        self.f_big   = pygame.font.SysFont("georgia,timesnewroman,serif", 23, bold=True)
        self.f_mid   = pygame.font.SysFont("georgia,timesnewroman,serif", 19)
        self.f_small = pygame.font.SysFont("georgia,timesnewroman,serif", 16)
        self.f_tiny  = pygame.font.SysFont("georgia,timesnewroman,serif", 13)
        self.f_street= pygame.font.SysFont("georgia,timesnewroman,serif", 14, bold=True)
        # …y DejaVu SOLO donde hay glifos de palo (las serif los pintan como □).
        self.f_card    = pygame.font.SysFont("dejavusans,arial,sans-serif", 26, bold=True)
        self.f_csuit   = pygame.font.SysFont("dejavusans,arial,sans-serif", 30, bold=True)
        self.f_suit_sm = pygame.font.SysFont("dejavusans,arial,sans-serif", 15, bold=True)
        self.f_suit_lg = pygame.font.SysFont("dejavusans,arial,sans-serif", 22, bold=True)

        # Juego
        from src.engine.player import HumanPlayer
        from src.ai.agent_player import AgentPlayer
        if opponent is None:
            from src.ai.rule_based_agent import RuleBasedAgent
            opponent = RuleBasedAgent()
        self.human = HumanPlayer("Tú", self.STARTING_STACK)
        self.agent = AgentPlayer(agent_name, self.STARTING_STACK, opponent)
        self.gc = GameController(self.human, self.agent)
        self.agent.bind(self.gc)

        self.last_step = 0
        self.prev_awaiting = False
        self.raise_to = 0
        self.session_over = False
        self._next_btn_rect = None
        # Estado del toast de última acción (texto mostrado + inicio de su pop-up)
        self._msg_text = ""
        self._msg_t0   = 0
        # Hilo del paso de la IA: evita congelar la ventana con agentes lentos
        # (un LLMAgent tarda segundos por decisión). Ver _auto_advance().
        self._worker: threading.Thread | None = None
        self._think_started = 0

        self.gc.start_hand()
        self.last_step = pygame.time.get_ticks()

    # ─────────────────── dibujo de primitivas ───────────────────

    def _text(self, surf, text, font, color, center=None, topleft=None, shadow=True):
        if shadow:
            sh = font.render(text, True, SHADOW)
            if center:    surf.blit(sh, sh.get_rect(center=(center[0]+1, center[1]+1)))
            elif topleft: surf.blit(sh, (topleft[0]+1, topleft[1]+1))
        img = font.render(text, True, color)
        if center:    surf.blit(img, img.get_rect(center=center))
        elif topleft: surf.blit(img, topleft)
        return img.get_rect(center=center) if center else img.get_rect(topleft=topleft)

    def _draw_diamond(self, cx, cy, size, color):
        pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
        pygame.draw.polygon(self.screen, color, pts)

    def _draw_header(self):
        """Barra superior oscura con título y número de mano."""
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, WIDTH, HEADER_H))
        pygame.draw.line(self.screen, GOLD,     (0, HEADER_H - 1), (WIDTH, HEADER_H - 1), 2)
        pygame.draw.line(self.screen, GOLD_DIM, (0, HEADER_H - 3), (WIDTH, HEADER_H - 3), 1)
        for x in (18, WIDTH - 18):
            self._draw_diamond(x, HEADER_H // 2, 9, GOLD_DIM)

        cy = HEADER_H // 2
        title = "PÓKER NLHE HEADS-UP"
        tw = self.f_title.size(title)[0]
        self._text(self.screen, title, self.f_title, GOLD, center=(WIDTH // 2, cy))
        # Los palos se pintan con DejaVu; con la serif saldrían como cajas.
        self._text(self.screen, "♠", self.f_suit_lg, GOLD_DIM,
                   center=(WIDTH // 2 - tw // 2 - 24, cy), shadow=False)
        self._text(self.screen, "♥", self.f_suit_lg, GOLD_DIM,
                   center=(WIDTH // 2 + tw // 2 + 24, cy), shadow=False)

        self._text(self.screen, f"MANO #{self.gc.hand_num}", self.f_small, GOLD_DIM,
                   center=(WIDTH - 88, cy))

    def _draw_felt(self):
        self.screen.fill(WOOD_DARK)
        t = TABLE_RECT
        pygame.draw.ellipse(self.screen, WOOD,       t.inflate(28, 28))
        pygame.draw.ellipse(self.screen, WOOD_LIGHT, t.inflate(20, 20), 5)
        pygame.draw.ellipse(self.screen, WOOD_DARK,  t.inflate(14, 14), 3)
        pygame.draw.ellipse(self.screen, GOLD_DIM,   t.inflate(6, 6))
        pygame.draw.ellipse(self.screen, GOLD,       t.inflate(4, 4), 3)
        pygame.draw.ellipse(self.screen, FELT_DARK,  t)
        inner = t.inflate(-30, -30)
        pygame.draw.ellipse(self.screen, FELT_LIGHT, inner)
        pygame.draw.ellipse(self.screen, FELT_EDGE,  inner.inflate(-18, -18), 1)
        self._draw_felt_suits(t)

    def _draw_felt_suits(self, table):
        self._text(self.screen, "♦", self.f_suit_lg, FELT_EDGE,
                   center=(table.right - 86, table.centery), shadow=False)

    def _draw_card(self, card, x, y, w=CARD_W, h=CARD_H, face_up=True):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (0, 0, 0), rect.move(3, 4), border_radius=8)

        if not face_up:
            pygame.draw.rect(self.screen, CARD_BACK, rect, border_radius=8)
            inner = rect.inflate(-10, -10)
            clip = self.screen.get_clip()
            self.screen.set_clip(inner)
            for d in range(0, inner.width + inner.height, 8):
                x1 = inner.left + min(d, inner.width)
                y1 = inner.top  + max(0, d - inner.width)
                x2 = inner.left + max(0, d - inner.height)
                y2 = inner.top  + min(d, inner.height)
                pygame.draw.line(self.screen, CARD_BACK2, (x1, y1), (x2, y2), 1)
                x1b = inner.right - min(d, inner.width)
                pygame.draw.line(self.screen, CARD_BACK3,
                                 (x1b, y1), (inner.right - max(0, d - inner.height), y2), 1)
            self.screen.set_clip(clip)
            pygame.draw.rect(self.screen, CARD_BACK2, inner, 2, border_radius=6)
            pygame.draw.rect(self.screen, GOLD_DIM, rect, 2, border_radius=8)
            return rect

        pygame.draw.rect(self.screen, WHITE, rect, border_radius=8)
        pygame.draw.rect(self.screen, (205, 205, 200), rect, 1, border_radius=8)

        s = card_to_str(card)
        rank, suit = RANK_DISP.get(s[0], s[0]), s[1]
        col, sym   = SUIT_COLOR[suit], SUIT_SYMBOL[suit]

        # Rango arriba-izquierda + palo pequeño justo debajo (ambos legibles)
        self._text(self.screen, rank, self.f_card, col,
                   topleft=(x + 7, y + 3), shadow=False)
        self._text(self.screen, sym, self.f_suit_sm, col,
                   topleft=(x + 9, y + 32), shadow=False)
        # Palo grande centrado
        self._text(self.screen, sym, self.f_csuit, col,
                   center=(x + w // 2 + 6, y + h // 2 + 10), shadow=False)
        return rect

    def _draw_card_row(self, cards, cx, y, face_up=True, gap=12, w=CARD_W, h=CARD_H):
        n = len(cards)
        if n == 0:
            return
        startx = cx - (n * w + (n - 1) * gap) // 2
        for i, c in enumerate(cards):
            self._draw_card(c, startx + i * (w + gap), y, w=w, h=h, face_up=face_up)

    def _draw_placeholder_board(self, cx, y, count=5):
        startx = cx - (count * CARD_W + (count - 1) * 12) // 2
        for i in range(count):
            r = pygame.Rect(startx + i * (CARD_W + 12), y, CARD_W, CARD_H)
            pygame.draw.rect(self.screen, FELT_EDGE, r, border_radius=8)
            pygame.draw.rect(self.screen, FELT_DARK, r, 2, border_radius=8)

    def _draw_pot(self, pot):
        """Píldora legible con el bote (sustituye a la ficha diminuta)."""
        label = f"BOTE   {pot}"
        tw    = self.f_big.size(label)[0]
        w, h  = tw + 76, 38
        rect  = pygame.Rect(0, 0, w, h)
        rect.center = (WIDTH // 2, POT_CY)

        pygame.draw.rect(self.screen, (0, 0, 0), rect.move(2, 3), border_radius=h // 2)
        pygame.draw.rect(self.screen, (12, 46, 32), rect, border_radius=h // 2)
        pygame.draw.rect(self.screen, GOLD, rect, 2, border_radius=h // 2)

        # Ficha decorativa dentro de la píldora, a la izquierda
        chip = (rect.left + 24, rect.centery)
        pygame.draw.circle(self.screen, GOLD,          chip, 13)
        pygame.draw.circle(self.screen, GOLD_DIM,      chip, 13, 2)
        pygame.draw.circle(self.screen, (180, 140, 30), chip, 8, 1)
        pygame.draw.circle(self.screen, GOLD_BRIGHT, (chip[0] - 4, chip[1] - 4), 3)

        self._text(self.screen, label, self.f_big, GOLD,
                   center=(rect.centerx + 16, rect.centery), shadow=False)

    @staticmethod
    def _wrap(text, font, max_w):
        """Parte el texto en líneas que quepan en `max_w` píxeles."""
        lines, cur = [], ""
        for word in text.split():
            probe = f"{cur} {word}".strip()
            if not cur or font.size(probe)[0] <= max_w:
                cur = probe
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _draw_action_toast(self):
        text = self.gc.last_action_str
        if not text or self.gc.hand_over:
            return

        now = pygame.time.get_ticks()
        if text != self._msg_text:          # texto nuevo -> reinicia la animación
            self._msg_text = text
            self._msg_t0   = now

        t     = min(1.0, (now - self._msg_t0) / MSG_POP_MS)
        ease  = _ease_out_back(t)
        scale = 0.82 + 0.18 * ease           # 0.82 -> ~1.05 -> 1.0 (rebote)
        alpha = int(255 * min(1.0, t * 3))   # fade-in rápido
        dx    = int(-20 * (1.0 - ease))      # entra deslizándose

        font  = self.f_big
        lines = self._wrap(text, font, MSG_MAX_TEXT_W)
        lh    = font.get_height()
        tw    = max(font.size(ln)[0] for ln in lines)
        w     = max(tw + 52, MSG_MIN_W)
        h     = len(lines) * lh + 26

        # Se dibuja en una superficie aparte para poder escalarla y darle alpha.
        surf = pygame.Surface((w + 10, h + 10), pygame.SRCALPHA)
        box  = pygame.Rect(5, 5, w, h)
        pygame.draw.rect(surf, (0, 0, 0, 235), box, border_radius=12)
        pygame.draw.rect(surf, GOLD,     box, 2, border_radius=12)
        pygame.draw.rect(surf, GOLD_DIM, box.inflate(-7, -7), 1, border_radius=9)
        # Barra de acento ámbar en el lateral izquierdo
        pygame.draw.rect(surf, AMBER,
                         pygame.Rect(box.left + 9, box.top + 11, 5, box.height - 22),
                         border_radius=3)
        for i, ln in enumerate(lines):
            img = font.render(ln, True, AMBER)
            surf.blit(img, img.get_rect(
                center=(box.centerx + 8, box.top + 13 + i * lh + lh // 2)))

        surf = pygame.transform.smoothscale(
            surf, (max(1, int(surf.get_width() * scale)),
                   max(1, int(surf.get_height() * scale))))
        surf.set_alpha(alpha)
        self.screen.blit(surf, surf.get_rect(center=(MSG_CX + dx, MSG_CY)))

    def _draw_street_label(self):
        lbl = STREET_LABEL.get(len(self.gc.board))
        if not lbl:
            return
        w, h = 104, 22
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (WIDTH // 2, STREET_CY)
        pygame.draw.rect(self.screen, (8, 38, 24), rect, border_radius=6)
        pygame.draw.rect(self.screen, GOLD_DIM, rect, 1, border_radius=6)
        self._text(self.screen, lbl, self.f_street, GOLD_DIM,
                   center=rect.center, shadow=False)

    # ─────────────────────── panel jugador ──────────────────────

    @staticmethod
    def _fit_text(text, max_w, fonts):
        """(fuente, texto) que cabe en max_w probando `fonts` de mayor a menor.

        Si ni la más pequeña cabe, recorta con puntos suspensivos: un nombre
        cortado sigue siendo legible; uno que se sale del panel, no.
        """
        for f in fonts:
            if f.size(text)[0] <= max_w:
                return f, text

        f, corto = fonts[-1], text
        while corto and f.size(corto + "…")[0] > max_w:
            corto = corto[:-1]
        return f, (corto + "…" if corto else text)

    def _draw_player_panel(self, name, stack, is_dealer, cy, highlight):
        w, h = 250, 68
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (WIDTH // 2, cy)
        pygame.draw.rect(self.screen, (0, 0, 0), rect.move(3, 4), border_radius=12)
        pygame.draw.rect(self.screen, (54, 46, 18) if highlight else (28, 28, 36),
                         rect, border_radius=12)
        border_col = GOLD if highlight else GOLD_DIM
        pygame.draw.rect(self.screen, border_col, rect, 2, border_radius=12)

        av     = (rect.left + 36, rect.centery)
        av_bg  = (180, 140, 30) if highlight else (55, 55, 70)
        pygame.draw.circle(self.screen, av_bg,      av, 22)
        pygame.draw.circle(self.screen, border_col, av, 22, 2)
        self._text(self.screen, name[0].upper(), self.f_big,
                   BLACK if highlight else CREAM, center=av, shadow=False)

        # El nombre ya no es siempre "IA": puede ser "RL (self-play/rl_2m)". Se
        # baja de fuente antes que recortar, para que el modelo siga siendo
        # identificable de un vistazo.
        f_name, shown = self._fit_text(
            name, rect.right - 10 - (rect.left + 68),
            (self.f_big, self.f_mid, self.f_small, self.f_tiny),
        )
        self._text(self.screen, shown, f_name, GOLD if highlight else CREAM,
                   topleft=(rect.left + 68, rect.top + 10))
        self._text(self.screen, f"{stack} fichas", self.f_small,
                   AMBER if highlight else GOLD,
                   topleft=(rect.left + 68, rect.bottom - 25))

        if is_dealer:
            dc = (rect.right + 24, rect.centery)
            pygame.draw.circle(self.screen, CREAM, dc, 15)
            pygame.draw.circle(self.screen, GOLD,  dc, 15, 2)
            self._text(self.screen, "D", self.f_small, BLACK, center=dc, shadow=False)

    # ───────────────────────── botones ──────────────────────────

    def _action_buttons(self):
        gc      = self.gc
        to_call = gc.human_to_call
        buttons = [
            {"id": "fold", "label": "RETIRARSE",
             "rect": pygame.Rect(36, BTN_Y, 150, 50), "enabled": True},
        ]
        if to_call == 0:
            lbl = "PASAR"
        elif to_call >= self.human.stack:
            lbl = f"IGUALAR {self.human.stack} (ALL-IN)"
        else:
            lbl = f"IGUALAR {to_call}"
        buttons.append({"id": "call", "label": lbl,
                        "rect": pygame.Rect(200, BTN_Y, 230, 50), "enabled": True})

        if self.human.stack > to_call:
            buttons += [
                {"id": "dec",    "label": "−",
                 "rect": pygame.Rect(448, BTN_Y, 46, 50),  "enabled": True},
                {"id": "amount", "label": f"{self.raise_to}",
                 "rect": pygame.Rect(498, BTN_Y, 100, 50), "enabled": False},
                {"id": "inc",    "label": "+",
                 "rect": pygame.Rect(602, BTN_Y, 46, 50),  "enabled": True},
                {"id": "raise",  "label": "SUBIR",
                 "rect": pygame.Rect(662, BTN_Y, 130, 50), "enabled": True},
                {"id": "allin",  "label": "ALL-IN",
                 "rect": pygame.Rect(806, BTN_Y, 150, 50), "enabled": True},
            ]
        return buttons

    def _draw_button(self, b, mouse):
        rect       = b["rect"]
        hover      = rect.collidepoint(mouse) and b["enabled"]
        is_special = b["id"] in ("raise", "allin")

        if not b["enabled"]:
            color = BTN_DISABLE
        elif hover:
            color = BTN_HOVER
        else:
            color = (48, 36, 10) if is_special else BTN
        pygame.draw.rect(self.screen, color, rect, border_radius=10)

        edge = GOLD_BRIGHT if hover else (GOLD if is_special else GOLD_DIM)
        pygame.draw.rect(self.screen, edge, rect, 2, border_radius=10)

        txt_col = GOLD if is_special else (CREAM if b["enabled"] else TEXT_DIM)
        self._text(self.screen, b["label"], self.f_mid, txt_col, center=rect.center)

    def _draw_action_bg(self):
        s = pygame.Surface((WIDTH, HEIGHT - ACTION_BAR_Y), pygame.SRCALPHA)
        s.fill((0, 0, 0, 165))
        self.screen.blit(s, (0, ACTION_BAR_Y))
        pygame.draw.line(self.screen, GOLD_DIM,
                         (0, ACTION_BAR_Y), (WIDTH, ACTION_BAR_Y), 1)

    def _draw_action_hint(self):
        """Línea de ayuda con los importes legales, encima de los botones."""
        gc   = self.gc
        bits = [f"Para igualar: {gc.human_to_call}"] if gc.human_to_call else ["Puedes pasar"]
        if self.human.stack > gc.human_to_call:
            bits.append(f"Subida mín: {min(gc.human_min_raise, gc.human_all_in)}")
            bits.append(f"Máx (all-in): {gc.human_all_in}")
        self._text(self.screen, "   ·   ".join(bits), self.f_tiny, TEXT_DIM,
                   center=(WIDTH // 2, ACTION_BAR_Y + 14), shadow=False)

    def _draw_next_button(self, mouse):
        label = "NUEVA PARTIDA" if self.session_over else "SIGUIENTE MANO"
        rect  = pygame.Rect(0, 0, 280, 50)
        rect.center = (WIDTH // 2, BTN_Y + 25)
        hover = rect.collidepoint(mouse)
        pygame.draw.rect(self.screen, BTN_HOVER if hover else (45, 35, 8),
                         rect, border_radius=12)
        pygame.draw.rect(self.screen, GOLD if hover else GOLD_DIM, rect, 2, border_radius=12)
        self._text(self.screen, label, self.f_big, GOLD if hover else CREAM,
                   center=rect.center)
        return rect

    # ─────────────────────── render general ─────────────────────

    def render(self):
        gc    = self.gc
        mouse = pygame.mouse.get_pos()

        self._draw_felt()
        self._draw_header()

        agent_turn = (not gc.hand_over) and (gc.to_act is self.agent)
        human_turn = gc.awaiting_human

        # ── IA (arriba) ──
        self._draw_player_panel(self.agent.name, self.agent.stack,
                                gc.dealer is self.agent,
                                AGENT_PANEL_CY, highlight=agent_turn)
        reveal = gc.hand_over and gc.result and gc.result["reveal_agent"]
        self._draw_card_row(self.agent.hole_cards, WIDTH // 2, AGENT_CARDS_Y,
                            face_up=reveal, w=AGENT_CW, h=AGENT_CH, gap=10)

        # ── Centro: bote + calle, o banner de resultado al terminar ──
        if gc.hand_over and gc.result:
            self._draw_result_banner()
        else:
            self._draw_pot(gc.pot)
            self._draw_street_label()

        # ── Tablero ──
        if gc.board:
            self._draw_card_row(gc.board, WIDTH // 2, BOARD_Y)
        else:
            self._draw_placeholder_board(WIDTH // 2, BOARD_Y)

        # ── Toast de última acción (izquierda, con pop-up) ──
        self._draw_action_toast()

        # ── Humano (abajo): panel y cartas siempre visibles ──
        self._draw_player_panel(self.human.name, self.human.stack,
                                gc.dealer is self.human,
                                HUMAN_PANEL_CY, highlight=human_turn)
        self._draw_card_row(self.human.hole_cards, WIDTH // 2, HUMAN_CARDS_Y,
                            face_up=True, w=HOLE_W, h=HOLE_H, gap=14)

        # ── Barra de acción ──
        self._draw_action_bg()
        if gc.hand_over:
            self._next_btn_rect = self._draw_next_button(mouse)
        elif human_turn:
            self._draw_action_hint()
            for b in self._action_buttons():
                self._draw_button(b, mouse)
        else:
            self._text(self.screen, self._thinking_label(), self.f_mid, TEXT_DIM,
                       center=(WIDTH // 2, BTN_Y + 25))

        pygame.display.flip()

    def _draw_result_banner(self):
        r = self.gc.result
        if r["winner"] == "Empate":
            head, head_col = "Empate — bote dividido", CREAM
        else:
            how  = " (rival se retiró)" if r["by_fold"] else ""
            head = f"Gana {r['winner']}{how}   ·   Bote {r['pot']}"
            head_col = GOLD_BRIGHT

        sub = (None if r["by_fold"]
               else f"Tú: {r['player_hand']}    |    "
                    f"{self.agent.name}: {r['agent_hand']}")

        w = max(self.f_big.size(head)[0], self.f_small.size(sub)[0] if sub else 0) + 56
        h = 64 if sub else 44
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (WIDTH // 2, RESULT_CY)

        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 205))
        self.screen.blit(s, rect.topleft)
        pygame.draw.rect(self.screen, GOLD, rect, 2, border_radius=12)

        if sub:
            self._text(self.screen, head, self.f_big, head_col,
                       center=(rect.centerx, rect.top + 21))
            self._text(self.screen, sub, self.f_small, CREAM,
                       center=(rect.centerx, rect.bottom - 20))
        else:
            self._text(self.screen, head, self.f_big, head_col, center=rect.center)

        if self.session_over:
            who = self.human.name if self.human.stack > 0 else self.agent.name
            self._text(self.screen, f"¡{who} gana la partida!", self.f_mid, AMBER,
                       center=(WIDTH // 2, rect.bottom + 20))

    # ───────────────────────── lógica ──────────────────────────

    def _on_human_turn_start(self):
        # El mínimo legal puede superar el all-in (stack corto): manda el all-in.
        self.raise_to = min(self.gc.human_min_raise, self.gc.human_all_in)

    def _clamp_raise(self):
        lo = min(self.gc.human_min_raise, self.gc.human_all_in)
        hi = self.gc.human_all_in
        self.raise_to = max(lo, min(self.raise_to, hi))

    def _handle_click(self, pos):
        gc = self.gc
        # Mientras la IA piensa en su hilo, nadie más toca el controlador.
        if self._worker is not None and self._worker.is_alive():
            return
        if gc.hand_over:
            if self._next_btn_rect and self._next_btn_rect.collidepoint(pos):
                self._next_hand()
            return
        if not gc.awaiting_human:
            return
        for b in self._action_buttons():
            if not b["enabled"] or not b["rect"].collidepoint(pos):
                continue
            bid = b["id"]
            if bid == "fold":
                gc.submit_human_action(Action.FOLD)
            elif bid == "call":
                act = Action.CHECK if gc.human_to_call == 0 else Action.CALL
                gc.submit_human_action(act, gc.human_to_call)
            elif bid == "dec":
                self.raise_to -= GameController.BIG_BLIND; self._clamp_raise()
            elif bid == "inc":
                self.raise_to += GameController.BIG_BLIND; self._clamp_raise()
            elif bid == "raise":
                gc.submit_human_action(Action.RAISE, self.raise_to)
            elif bid == "allin":
                gc.submit_human_action(Action.RAISE, gc.human_all_in)
            break

    def _next_hand(self):
        if self.session_over:
            self.human.stack = self.STARTING_STACK
            self.agent.stack = self.STARTING_STACK
            self.gc.dealer   = self.human
            self.session_over = False
        self.gc.start_hand()
        self.last_step = pygame.time.get_ticks()

    def _auto_advance(self):
        gc = self.gc

        # ¿Hay un paso de la IA en curso?
        if self._worker is not None:
            if self._worker.is_alive():
                return                       # sigue pensando; no tocar gc
            self._worker = None              # terminó: recoger y seguir
            self.last_step = pygame.time.get_ticks()
            if gc.hand_over and gc.session_over():
                self.session_over = True
            return

        if gc.hand_over or gc.awaiting_human:
            return

        now = pygame.time.get_ticks()
        if now - self.last_step >= STEP_DELAY_MS:
            self._think_started = now
            self._worker = threading.Thread(target=gc.step, daemon=True)
            self._worker.start()

    def _thinking_label(self) -> str:
        """Texto del turno del rival, con segundos si tarda (LLM lento)."""
        base = f"{self.agent.name} está pensando…"
        if self._worker is None or self._think_started == 0:
            return base
        secs = (pygame.time.get_ticks() - self._think_started) // 1000
        if secs < 2:
            return base
        return f"{base}  ({secs} s)"

    # ─────────────────────── bucle principal ────────────────────

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if self.gc.awaiting_human and not self.prev_awaiting:
                self._on_human_turn_start()
            self.prev_awaiting = self.gc.awaiting_human

            self._auto_advance()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main(opponent=None, agent_name: str = "IA"):
    PokerGUI(opponent, agent_name).run()


if __name__ == "__main__":
    main()
