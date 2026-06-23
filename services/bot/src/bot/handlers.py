"""Хендлеры команд и callback-запросов бота (aiogram 3.x Router) — тонкая склейка.

Хендлер извлекает из aiogram-объектов примитивы (chat_id, текст, callback_data),
делегирует бизнес-логику в ``responses`` / ``wizard``, отправляет результат.
API-клиент инъектируется в polling как kwarg ``api_client`` (DI через Dispatcher/start_polling).
"""

import html
from datetime import datetime
from zoneinfo import ZoneInfo

import structlog
from aiogram import Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from stocklens_core.enums import AlertKind

from bot import formatting, responses
from bot.api_client.client import ApiClient
from bot.api_client.dto import SubscriptionIn
from bot.api_client.errors import ApiError
from bot.callbacks import (
    DeleteSubCb,
    MenuAction,
    MenuCb,
    WizCancelCb,
    WizKindCb,
    WizManualCb,
    WizTickerCb,
)
from bot.keyboards import main_menu, subscriptions_kb, wizard_kind_kb, wizard_ticker_kb
from bot.states import SubscribeWizard
from bot.wizard import WizardError, build_subscription, validate_level, validate_ticker

router = Router(name="commands")

_MOSCOW_TZ = ZoneInfo("Europe/Moscow")
_log = structlog.get_logger()

_ERR_WIZARD_CANCELLED = "Создание подписки отменено."

_FSM_KEY_KIND = "kind"
_FSM_KEY_TICKER = "ticker"


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """/start — приветствие с кнопками быстрых действий."""
    await message.answer(responses.start_response(), reply_markup=main_menu())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """/help — структурированная справка с кнопками быстрых действий."""
    await message.answer(responses.help_response(), reply_markup=main_menu())


@router.message(Command("portfolio"))
async def cmd_portfolio(message: Message, api_client: ApiClient) -> None:
    """/portfolio — сводка портфеля."""
    await message.answer(await responses.portfolio_response(api_client))


@router.message(Command("digest"))
async def cmd_digest(message: Message, api_client: ApiClient) -> None:
    """/digest — дайджест по портфелю."""
    today = datetime.now(tz=_MOSCOW_TZ).date()
    await message.answer(await responses.digest_response(api_client, today))


@router.message(Command("subscribe"))
async def cmd_subscribe(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    api_client: ApiClient,
) -> None:
    """/subscribe: с аргументами — текстовый путь (создать сразу), без — FSM-мастер."""
    if command.args:
        await message.answer(
            await responses.subscribe_response(api_client, message.chat.id, command.args),
            reply_markup=main_menu(),
        )
        return
    await state.set_state(SubscribeWizard.choosing_kind)
    await message.answer(
        "🔔 <b>Новая подписка</b>\n\nВыберите вид алерта:",
        reply_markup=wizard_kind_kb(),
    )


@router.message(Command("unsubscribe"))
async def cmd_unsubscribe(
    message: Message,
    command: CommandObject,
    api_client: ApiClient,
) -> None:
    """/unsubscribe: с id — удалить сразу, без аргументов — список с кнопками ❌/➕."""
    if command.args:
        await message.answer(await responses.unsubscribe_response(api_client, command.args))
        return
    chat_id = message.chat.id
    try:
        subs = await api_client.list_subscriptions(chat_id)
    except ApiError as exc:
        _log.warning("api_error", detail=exc.user_message)
        await message.answer(exc.user_message)
        return
    text = formatting.format_subscriptions(subs)
    await message.answer(text, reply_markup=subscriptions_kb(subs))


@router.callback_query(MenuCb.filter())
async def cb_menu(
    query: CallbackQuery,
    callback_data: MenuCb,
    state: FSMContext,
    api_client: ApiClient,
) -> None:
    """Quick-action кнопки из /start и /help: портфель, дайджест, подписки, мастер."""
    await query.answer()
    action = callback_data.action
    if action == MenuAction.PORTFOLIO:
        text = await responses.portfolio_response(api_client)
        await _safe_send(query, text)
    elif action == MenuAction.DIGEST:
        today = datetime.now(tz=_MOSCOW_TZ).date()
        text = await responses.digest_response(api_client, today)
        await _safe_send(query, text)
    elif action == MenuAction.SUBS:
        if isinstance(query.message, Message):
            chat_id = query.message.chat.id
        else:
            chat_id = query.from_user.id
        try:
            subs = await api_client.list_subscriptions(chat_id)
        except ApiError as exc:
            _log.warning("api_error", detail=exc.user_message)
            await _safe_send(query, exc.user_message)
            return
        text = formatting.format_subscriptions(subs)
        await _safe_send(query, text, reply_markup=subscriptions_kb(subs))
    elif action == MenuAction.SUBSCRIBE:
        await state.set_state(SubscribeWizard.choosing_kind)
        await _safe_send(
            query,
            "🔔 <b>Новая подписка</b>\n\nВыберите вид алерта:",
            reply_markup=wizard_kind_kb(),
        )


@router.callback_query(DeleteSubCb.filter())
async def cb_delete_sub(
    query: CallbackQuery,
    callback_data: DeleteSubCb,
    api_client: ApiClient,
) -> None:
    """Удалить подписку по id и перерисовать список."""
    await query.answer()
    chat_id = query.message.chat.id if isinstance(query.message, Message) else query.from_user.id
    try:
        await api_client.delete_subscription(callback_data.sub_id)
    except ApiError as exc:
        _log.warning("api_error", detail=exc.user_message)
        await _safe_send(query, exc.user_message)
        return
    try:
        subs = await api_client.list_subscriptions(chat_id)
    except ApiError as exc:
        _log.warning("api_error", detail=exc.user_message)
        await _safe_send(query, exc.user_message)
        return
    text = formatting.format_subscriptions(subs)
    if isinstance(query.message, Message):
        await query.message.edit_text(text, reply_markup=subscriptions_kb(subs))


@router.callback_query(WizCancelCb.filter())
async def cb_wiz_cancel(query: CallbackQuery, state: FSMContext) -> None:
    """Отмена мастера /subscribe на любом шаге."""
    await query.answer()
    await state.clear()
    await _safe_send(query, _ERR_WIZARD_CANCELLED, reply_markup=main_menu())


@router.callback_query(WizKindCb.filter(), SubscribeWizard.choosing_kind)
async def cb_wiz_kind(
    query: CallbackQuery,
    callback_data: WizKindCb,
    state: FSMContext,
    api_client: ApiClient,
) -> None:
    """Шаг 1 мастера: пользователь выбрал вид алерта."""
    await query.answer()
    await state.update_data({_FSM_KEY_KIND: callback_data.kind.value})
    tickers = await _portfolio_tickers(api_client)
    await state.set_state(SubscribeWizard.choosing_ticker)
    await _safe_send(query, "Выберите тикер:", reply_markup=wizard_ticker_kb(tickers))


@router.callback_query(WizTickerCb.filter(), SubscribeWizard.choosing_ticker)
async def cb_wiz_ticker(
    query: CallbackQuery,
    callback_data: WizTickerCb,
    state: FSMContext,
    api_client: ApiClient,
) -> None:
    """Шаг 2 мастера: пользователь выбрал тикер из пикера."""
    await query.answer()
    data = await state.get_data()
    kind = AlertKind(data[_FSM_KEY_KIND])
    ticker = callback_data.ticker
    await state.update_data({_FSM_KEY_TICKER: ticker})
    if kind is AlertKind.PRICE_LEVEL:
        await state.set_state(SubscribeWizard.entering_level)
        await _safe_send(
            query,
            f"Тикер: <code>{html.escape(ticker)}</code>\n\n"
            "Введите уровень цены числом (например: <code>250</code>):",
        )
    else:
        await _create_and_confirm(query, state, api_client, kind, ticker, level=None)


@router.callback_query(WizManualCb.filter(), SubscribeWizard.choosing_ticker)
async def cb_wiz_manual(query: CallbackQuery, state: FSMContext) -> None:
    """Шаг 2 мастера: пользователь выбрал ручной ввод тикера."""
    await query.answer()
    await state.set_state(SubscribeWizard.entering_ticker)
    await _safe_send(query, "Введите тикер вручную (например: <code>SBER</code>):")


@router.callback_query()
async def cb_fallback(query: CallbackQuery) -> None:
    """Устаревшая кнопка (FSM-state сброшен): ответить, чтобы не висели «часики» Telegram."""
    await query.answer("Кнопка устарела. Откройте /subscribe или /unsubscribe заново.")


@router.message(SubscribeWizard.entering_ticker)
async def msg_wiz_ticker(message: Message, state: FSMContext, api_client: ApiClient) -> None:
    """Шаг 2 мастера (ручной ввод): обработать введённый тикер."""
    raw = message.text or ""
    result = validate_ticker(raw)
    if isinstance(result, WizardError):
        await message.answer(result.message)
        return
    ticker = result
    data = await state.get_data()
    kind = AlertKind(data[_FSM_KEY_KIND])
    await state.update_data({_FSM_KEY_TICKER: ticker})
    if kind is AlertKind.PRICE_LEVEL:
        await state.set_state(SubscribeWizard.entering_level)
        await message.answer(
            f"Тикер: <code>{html.escape(ticker)}</code>\n\n"
            "Введите уровень цены числом (например: <code>250</code>):"
        )
    else:
        await _create_and_confirm_from_message(message, state, api_client, kind, ticker, level=None)


@router.message(SubscribeWizard.entering_level)
async def msg_wiz_level(message: Message, state: FSMContext, api_client: ApiClient) -> None:
    """Шаг 3 мастера (только price_level): обработать введённый уровень цены."""
    raw = message.text or ""
    level_result = validate_level(raw)
    if isinstance(level_result, WizardError):
        await message.answer(level_result.message)
        return
    data = await state.get_data()
    kind = AlertKind(data[_FSM_KEY_KIND])
    ticker = str(data[_FSM_KEY_TICKER])
    await _create_and_confirm_from_message(
        message, state, api_client, kind, ticker, level=level_result
    )


async def _create_and_confirm(
    query: CallbackQuery,
    state: FSMContext,
    api_client: ApiClient,
    kind: AlertKind,
    ticker: str,
    level: float | None,
) -> None:
    """Создать подписку через API и показать подтверждение (колбэк-контекст)."""
    chat_id = query.message.chat.id if isinstance(query.message, Message) else query.from_user.id
    result = build_subscription(chat_id, kind, ticker, level)
    if isinstance(result, WizardError):
        await state.clear()
        await _safe_send(query, result.message, reply_markup=main_menu())
        return
    await state.clear()
    await _post_subscription(query, api_client, result)


async def _create_and_confirm_from_message(
    message: Message,
    state: FSMContext,
    api_client: ApiClient,
    kind: AlertKind,
    ticker: str,
    level: float | None,
) -> None:
    """Создать подписку через API и показать подтверждение (message-контекст)."""
    result = build_subscription(message.chat.id, kind, ticker, level)
    if isinstance(result, WizardError):
        await state.clear()
        await message.answer(result.message, reply_markup=main_menu())
        return
    await state.clear()
    try:
        created = await api_client.create_subscription(result)
    except ApiError as exc:
        _log.warning("api_error", detail=exc.user_message)
        await message.answer(exc.user_message, reply_markup=main_menu())
        return
    await message.answer(formatting.format_subscription_created(created), reply_markup=main_menu())


async def _post_subscription(
    query: CallbackQuery,
    api_client: ApiClient,
    subscription: SubscriptionIn,
) -> None:
    """Отправить POST к API и показать результат (колбэк-контекст)."""
    try:
        created = await api_client.create_subscription(subscription)
    except ApiError as exc:
        _log.warning("api_error", detail=exc.user_message)
        await _safe_send(query, exc.user_message, reply_markup=main_menu())
        return
    await _safe_send(
        query,
        formatting.format_subscription_created(created),
        reply_markup=main_menu(),
    )


async def _portfolio_tickers(api_client: ApiClient) -> list[str]:
    """Получить тикеры из портфеля для пикера мастера (пустой список при сбое API)."""
    try:
        summary = await api_client.get_portfolio_summary()
        return [pos.ticker for pos in summary.positions]
    except ApiError:
        return []


async def _safe_send(
    query: CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Редактировать текущее сообщение если доступно, иначе отправить новое."""
    if isinstance(query.message, Message):
        await query.message.edit_text(text, reply_markup=reply_markup)
    elif query.from_user is not None and query.bot is not None:
        await query.bot.send_message(query.from_user.id, text, reply_markup=reply_markup)
