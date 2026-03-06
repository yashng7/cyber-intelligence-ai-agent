"""Deterministic math tools — no LLM arithmetic."""

import structlog

logger = structlog.get_logger()


def compute_cagr(beginning_value: float, ending_value: float, years: float) -> float:
    if beginning_value <= 0 or ending_value <= 0 or years <= 0:
        raise ValueError("All values must be positive")
    result = (ending_value / beginning_value) ** (1 / years) - 1
    logger.info("CAGR", beginning=beginning_value, ending=ending_value, years=years, result=round(result, 4))
    return round(result, 4)


def compute_difference(a: float, b: float) -> float:
    result = a - b
    logger.info("Difference", a=a, b=b, result=round(result, 2))
    return round(result, 2)


def compute_percentage_change(old_value: float, new_value: float) -> float:
    if old_value == 0:
        raise ValueError("Old value cannot be zero")
    result = ((new_value - old_value) / old_value) * 100
    logger.info("Pct change", old=old_value, new=new_value, result=round(result, 2))
    return round(result, 2)