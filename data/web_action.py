"""Web action space definitions and encoding utilities.

Action vector layout (ACTION_DIM = 20 floats):
  [0]      action_type_id   int in [0, NUM_ACTION_TYPES)
  [1]      x_norm           float in [0, 1]
  [2]      y_norm           float in [0, 1]
  [3]      scroll_x         float (negative = left, positive = right)
  [4]      scroll_y         float (negative = up, positive = down)
  [5..19]  text_chars       float in [0, 1], char_byte / 255.0, zero-padded
"""

import numpy as np

ACTION_TYPES = ['noop', 'click', 'type', 'scroll', 'navigate', 'hover', 'drag', 'key_press']
ACTION_TYPE_TO_ID = {a: i for i, a in enumerate(ACTION_TYPES)}
NUM_ACTION_TYPES = len(ACTION_TYPES)
MAX_TEXT_LEN = 15
ACTION_DIM = 5 + MAX_TEXT_LEN  # 20


def encode_action(
    action_type: str,
    x_norm: float = 0.0,
    y_norm: float = 0.0,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
    text: str = "",
) -> np.ndarray:
    """Encode a web action into a fixed-size float32 vector."""
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    vec[0] = ACTION_TYPE_TO_ID.get(action_type, 0)
    vec[1] = float(x_norm)
    vec[2] = float(y_norm)
    vec[3] = float(scroll_x)
    vec[4] = float(scroll_y)
    for i, ch in enumerate(text[:MAX_TEXT_LEN]):
        vec[5 + i] = ord(ch) / 255.0
    return vec


def decode_action(vec: np.ndarray) -> dict:
    """Decode an action vector back to a human-readable dict (for debugging)."""
    type_id = int(round(vec[0]))
    type_id = max(0, min(type_id, NUM_ACTION_TYPES - 1))
    text_chars = [chr(int(round(vec[5 + i] * 255.0))) for i in range(MAX_TEXT_LEN) if vec[5 + i] > 0]
    return {
        "action_type": ACTION_TYPES[type_id],
        "x_norm": float(vec[1]),
        "y_norm": float(vec[2]),
        "scroll_x": float(vec[3]),
        "scroll_y": float(vec[4]),
        "text": "".join(text_chars),
    }
