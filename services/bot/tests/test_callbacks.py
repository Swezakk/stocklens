"""Тесты CallbackData-фабрик: pack/unpack round-trip (DESIGN §11 / bot-ux-polish).

Проверяем, что каждая фабрика сериализует и десериализует данные без потерь,
а AlertKind StrEnum восстанавливается как нужный тип.
"""

from bot.callbacks import (
    DeleteSubCb,
    MenuAction,
    MenuCb,
    WizCancelCb,
    WizKindCb,
    WizManualCb,
    WizTickerCb,
)
from stocklens_core.enums import AlertKind


def test_menu_cb_roundtrip_portfolio() -> None:
    packed = MenuCb(action=MenuAction.PORTFOLIO).pack()
    parsed = MenuCb.unpack(packed)
    assert parsed.action == MenuAction.PORTFOLIO


def test_menu_cb_roundtrip_all_actions() -> None:
    for action in MenuAction:
        packed = MenuCb(action=action).pack()
        assert MenuCb.unpack(packed).action == action


def test_delete_sub_cb_roundtrip_preserves_id() -> None:
    packed = DeleteSubCb(sub_id=42).pack()
    parsed = DeleteSubCb.unpack(packed)
    assert parsed.sub_id == 42


def test_wiz_kind_cb_roundtrip_price_level() -> None:
    packed = WizKindCb(kind=AlertKind.PRICE_LEVEL).pack()
    parsed = WizKindCb.unpack(packed)
    assert parsed.kind is AlertKind.PRICE_LEVEL
    assert isinstance(parsed.kind, AlertKind)


def test_wiz_kind_cb_roundtrip_all_alert_kinds() -> None:
    for kind in AlertKind:
        packed = WizKindCb(kind=kind).pack()
        assert WizKindCb.unpack(packed).kind == kind


def test_wiz_ticker_cb_roundtrip_preserves_ticker() -> None:
    packed = WizTickerCb(ticker="SBER").pack()
    parsed = WizTickerCb.unpack(packed)
    assert parsed.ticker == "SBER"


def test_wiz_manual_cb_roundtrip() -> None:
    packed = WizManualCb().pack()
    WizManualCb.unpack(packed)


def test_wiz_cancel_cb_roundtrip() -> None:
    packed = WizCancelCb().pack()
    WizCancelCb.unpack(packed)


def test_delete_sub_cb_prefix_differs_from_menu_cb() -> None:
    packed_del = DeleteSubCb(sub_id=1).pack()
    packed_menu = MenuCb(action=MenuAction.PORTFOLIO).pack()
    assert packed_del.split(":")[0] != packed_menu.split(":")[0]
