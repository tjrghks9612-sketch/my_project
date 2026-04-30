"""Message helpers."""

from __future__ import annotations

from components.cards import info_card, warning_card
from style.copy_ko import LARGE_DATA_NOTICE, NOT_CAUSAL


def info_message(message: str) -> None:
    info_card(message, "안내")


def warning_message(message: str) -> None:
    warning_card(message, "확인 필요")


def caution_message(message: str) -> None:
    warning_card(message, "주의")


def not_causal_notice() -> None:
    info_card(NOT_CAUSAL, "해석 주의")


def large_data_notice() -> None:
    info_card(LARGE_DATA_NOTICE, "대용량 데이터 안내")


def candidate_notice() -> None:
    not_causal_notice()
