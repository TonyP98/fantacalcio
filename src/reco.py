from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RecConfig:
    buy_alpha: float = 0.15
    hold_alpha: float = -0.05
    w_price: float = 0.5
    w_grid: float = 0.5


def value_alpha_from_fair(fair_value: Optional[float], price_500: Optional[float]) -> float:
    if not fair_value or not price_500 or price_500 <= 0:
        return 0.0
    return (float(fair_value) - float(price_500)) / float(price_500)


def classify_from_alpha(alpha: float, cfg: RecConfig) -> str:
    if alpha >= cfg.buy_alpha:
        return "BUY"
    if alpha >= cfg.hold_alpha:
        return "HOLD"
    return "AVOID"


def combine_gk(alpha_price: float, grid_signal: float, cfg: RecConfig) -> float:
    return cfg.w_price * alpha_price + cfg.w_grid * grid_signal


def recommend_player(
    player: dict,
    *,
    fair_value: Optional[float],
    price_500: Optional[float],
    role: str,
    team: str,
    gk_signal: Optional[float] = None,
    cfg: Optional[RecConfig] = None,
) -> tuple[str, float]:
    cfg = cfg or RecConfig()
    alpha = value_alpha_from_fair(fair_value, price_500)

    if role.upper() != "P":
        label = classify_from_alpha(alpha, cfg)
        return label, alpha

    gk_sig = gk_signal if gk_signal is not None else 0.0
    score = combine_gk(alpha, gk_sig, cfg)
    label = classify_from_alpha(score, cfg)
    return label, score
