"""OpenCV HUD rendering for SignBridge."""
from __future__ import annotations

import cv2

NAVY = (22, 24, 39)
PANEL = (31, 36, 54)
CYAN = (64, 220, 205)
WHITE = (238, 243, 249)
MUTED = (165, 178, 195)
AMBER = (70, 190, 250)


def draw_text(image, text: str, origin: tuple[int, int], scale: float = .56,
              color=WHITE, thickness: int = 1) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(frame, *, category: str, confidence: float, phrase: str, sentence: str,
           history: list[str], camera_status: str, recognition_status: str, speech_status: str,
           fps: float, show_help: bool, stability: tuple[int, int] = (0, 0),
           width: int = 1120, height: int = 780):
    """Compose a resizable, dark-themed video/HUD view."""
    canvas = cv2.resize(frame, (width, height - 245), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (width, 56), NAVY, -1)
    draw_text(canvas, "SignBridge", (24, 37), .9, CYAN, 2)
    draw_text(canvas, camera_status, (width - 340, 34), .52, CYAN)
    cv2.rectangle(canvas, (0, height - 245), (width, height), NAVY, -1)
    cv2.rectangle(canvas, (18, height - 225), (width - 18, height - 136), PANEL, -1)
    draw_text(canvas, "MODE / GESTURE", (35, height - 196), .43, MUTED)
    accepted = stability[0] >= stability[1] and stability[1] > 0
    draw_text(canvas, f"{category}  {confidence:.0%}" if category else "No hand detected", (35, height - 166), .58,
              CYAN if accepted else WHITE)
    draw_text(canvas, f"Phrase: {phrase or '—'}", (width // 2, height - 166), .62, CYAN)
    draw_text(canvas, f"FPS {fps:.0f}", (width - 150, height - 166), .53, MUTED)
    cv2.rectangle(canvas, (18, height - 126), (width - 18, height - 58), PANEL, -1)
    draw_text(canvas, "CURRENT SENTENCE", (35, height - 103), .40, MUTED)
    draw_text(canvas, sentence or "Your sentence will appear here", (35, height - 75), .61, WHITE)
    recent = "  •  ".join(history[-5:]) or "No phrases yet"
    draw_text(canvas, f"HISTORY: {recent}", (24, height - 39), .48, MUTED)
    draw_text(canvas, f"{recognition_status}  |  {speech_status}", (24, height - 16), .38, CYAN)
    draw_text(canvas, "[C] Clear   [Backspace] Undo   [S] Speak   [L] Landmarks   [H] Help   [Q/Esc] Quit",
              (24, height - 1), .42, WHITE)
    if show_help:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (width // 2 - 320, 84), (width // 2 + 320, 286), NAVY, -1)
        cv2.addWeighted(overlay, .92, canvas, .08, 0, canvas)
        draw_text(canvas, "SIGNBRIDGE  |  DEMONSTRATION ONLY", (width // 2 - 290, 120), .65, CYAN, 2)
        draw_text(canvas, "Built-in: HELLO, STOP, YES, NO, PEACE, QUESTION, I LOVE YOU", (width // 2 - 290, 158), .45)
        draw_text(canvas, "GOODBYE: deliberate open-palm side-to-side wave.", (width // 2 - 290, 188), .45, CYAN)
        draw_text(canvas, "Custom phrases need your collected samples and trained model.", (width // 2 - 290, 218), .43)
        draw_text(canvas, "Accepted phrases speak automatically; S speaks the full sentence.", (width // 2 - 290, 246), .43)
        draw_text(canvas, "Demo only; not a complete sign-language translator. Press H to hide.", (width // 2 - 290, 274), .42, AMBER)
    return canvas
