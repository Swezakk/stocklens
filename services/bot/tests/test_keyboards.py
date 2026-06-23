"""Тесты структуры inline-клавиатур: тексты кнопок, callback_data payload, раскладка.

Чистые функции keyboards.py — тестируются без рантайма Telegram.
"""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from bot.api_client.dto import SubscriptionOut
from bot.callbacks import (
    DeleteSubCb,
    MenuAction,
    MenuCb,
    WizCancelCb,
    WizKindCb,
    WizManualCb,
    WizTickerCb,
)
from bot.keyboards import (
    _MAX_TICKER_BUTTONS,
    main_menu,
    subscriptions_kb,
    wizard_kind_kb,
    wizard_ticker_kb,
)
from bot.subscriptions import _SUBSCRIBABLE_KINDS
from stocklens_core.enums import AlertKind


def _flat_buttons(markup: InlineKeyboardMarkup) -> list[InlineKeyboardButton]:
    return [btn for row in markup.inline_keyboard for btn in row]


def _assert_markup(value: object) -> InlineKeyboardMarkup:
    assert isinstance(value, InlineKeyboardMarkup)
    return value


def _callback_data(btn: InlineKeyboardButton) -> str:
    assert btn.callback_data is not None
    return btn.callback_data


def test_main_menu_has_portfolio_digest_subs_buttons() -> None:
    markup = _assert_markup(main_menu())
    texts = [btn.text for btn in _flat_buttons(markup)]
    assert any("Портфель" in t for t in texts)
    assert any("Дайджест" in t for t in texts)
    assert any("Подписки" in t for t in texts)


def test_main_menu_portfolio_callback_data_is_menu_action() -> None:
    markup = _assert_markup(main_menu())
    btn = next(b for b in _flat_buttons(markup) if "Портфель" in b.text)
    parsed = MenuCb.unpack(_callback_data(btn))
    assert parsed.action == MenuAction.PORTFOLIO


def test_main_menu_has_three_buttons() -> None:
    markup = _assert_markup(main_menu())
    assert len(_flat_buttons(markup)) == 3


def test_subscriptions_kb_has_delete_button_per_sub() -> None:
    subs = [
        SubscriptionOut(id=1, chat_id=7, kind=AlertKind.PRICE_LEVEL, params={"ticker": "SBER"}),
        SubscriptionOut(id=2, chat_id=7, kind=AlertKind.SENTIMENT_SPIKE, params={}),
    ]
    markup = _assert_markup(subscriptions_kb(subs))
    btns = _flat_buttons(markup)
    delete_btns = [b for b in btns if b.text.startswith("❌")]
    assert len(delete_btns) == 2


def test_subscriptions_kb_delete_button_has_correct_sub_id() -> None:
    subs = [SubscriptionOut(id=99, chat_id=7, kind=AlertKind.PRICE_LEVEL, params={})]
    markup = _assert_markup(subscriptions_kb(subs))
    del_btn = next(b for b in _flat_buttons(markup) if b.text.startswith("❌"))
    parsed = DeleteSubCb.unpack(_callback_data(del_btn))
    assert parsed.sub_id == 99


def test_subscriptions_kb_has_add_button() -> None:
    markup = _assert_markup(subscriptions_kb([]))
    btns = _flat_buttons(markup)
    assert any("Добавить" in b.text for b in btns)


def test_subscriptions_kb_add_button_triggers_subscribe_action() -> None:
    markup = _assert_markup(subscriptions_kb([]))
    add_btn = next(b for b in _flat_buttons(markup) if "Добавить" in b.text)
    parsed = MenuCb.unpack(_callback_data(add_btn))
    assert parsed.action == MenuAction.SUBSCRIBE


def test_wizard_kind_kb_has_only_subscribable_kinds() -> None:
    markup = _assert_markup(wizard_kind_kb())
    btns = _flat_buttons(markup)
    kind_btns = [b for b in btns if b.text != "❌ Отмена"]
    assert len(kind_btns) == len(_SUBSCRIBABLE_KINDS)
    for btn in kind_btns:
        parsed = WizKindCb.unpack(_callback_data(btn))
        assert parsed.kind in _SUBSCRIBABLE_KINDS


def test_wizard_kind_kb_does_not_include_volatility_regime() -> None:
    markup = _assert_markup(wizard_kind_kb())
    btns = _flat_buttons(markup)
    for btn in btns:
        cd = btn.callback_data
        if cd is not None and cd.startswith("wkind:"):
            parsed = WizKindCb.unpack(cd)
            assert parsed.kind is not AlertKind.VOLATILITY_REGIME


def test_wizard_kind_kb_has_cancel_button() -> None:
    markup = _assert_markup(wizard_kind_kb())
    btns = _flat_buttons(markup)
    cancel_btns = [b for b in btns if b.text == "❌ Отмена"]
    assert len(cancel_btns) == 1
    WizCancelCb.unpack(_callback_data(cancel_btns[0]))


def test_wizard_ticker_kb_shows_tickers_from_list() -> None:
    markup = _assert_markup(wizard_ticker_kb(["SBER", "GAZP", "LKOH"]))
    btns = _flat_buttons(markup)
    ticker_btns = [
        b for b in btns if b.callback_data is not None and b.callback_data.startswith("wticker:")
    ]
    tickers_in_kb = {WizTickerCb.unpack(_callback_data(b)).ticker for b in ticker_btns}
    assert {"SBER", "GAZP", "LKOH"} == tickers_in_kb


def test_wizard_ticker_kb_caps_at_max_buttons() -> None:
    many = [f"T{i:02d}" for i in range(20)]
    markup = _assert_markup(wizard_ticker_kb(many))
    btns = _flat_buttons(markup)
    ticker_btns = [
        b for b in btns if b.callback_data is not None and b.callback_data.startswith("wticker:")
    ]
    assert len(ticker_btns) == _MAX_TICKER_BUTTONS


def test_wizard_ticker_kb_has_manual_and_cancel_buttons() -> None:
    markup = _assert_markup(wizard_ticker_kb([]))
    btns = _flat_buttons(markup)
    texts = [b.text for b in btns]
    assert any("Ввести вручную" in t for t in texts)
    assert any("Отмена" in t for t in texts)


def test_wizard_ticker_kb_manual_button_payload() -> None:
    markup = _assert_markup(wizard_ticker_kb([]))
    btns = _flat_buttons(markup)
    manual_btn = next(b for b in btns if "Ввести вручную" in b.text)
    WizManualCb.unpack(_callback_data(manual_btn))
