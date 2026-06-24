"""Форматирование сообщений бота — чистые функции, HTML parse mode (DESIGN §11).

Слой без aiogram: на вход — mirror-DTO ответов API, на выход — HTML-строка для
``message.answer`` (parse_mode=HTML). Динамический текст (заголовки новостей, тикеры)
экранируется ``html.escape``, чтобы не сломать разметку. Unit-тестируется без рантайма бота.
"""

import html
from collections.abc import Sequence
from decimal import Decimal
from typing import assert_never

from stocklens_core.enums import AlertKind, Currency

from bot.api_client.dto import NewsOut, PendingAlert, PortfolioSummaryOut, SubscriptionOut
from bot.digest_model import DigestData, UpcomingDividend

START_TEXT = (
    "<b>📊 StockLens</b> — аналитика рынка MOEX.\n\n"
    "Команды:\n"
    "/portfolio — сводка портфеля: P&amp;L и доходность против IMOEX\n"
    "/digest — дайджест: портфель, ближайшие дивиденды, негативные новости\n"
    "/subscribe — подписаться на алерт\n"
    "/unsubscribe — мои подписки\n"
    "/help — справка\n\n"
    "Алерты работают: sweep каждые 30 минут, дайджест в 08:30 МСК."
)

HELP_TEXT = (
    "<b>📊 StockLens — справка</b>\n\n"
    "<b>Что умеет бот:</b>\n"
    "• /portfolio — стоимость портфеля, P&amp;L, доходность vs IMOEX, Sharpe, просадка\n"
    "• /digest — утренний дайджест: IMOEX, портфель, ближайшие дивидендные отсечки, "
    "негативные новости\n"
    "• /subscribe — подписка на алерт (мастер выбора)\n"
    "• /unsubscribe — список подписок с кнопками удаления\n\n"
    "<b>🔔 Виды алертов:</b>\n"
    "• <b>📉 Уровень цены</b> — уведомление, когда цена бумаги пересекает заданный уровень.\n"
    "  Пример: <code>SBER, уровень 250</code>\n"
    "• <b>⚠️ Всплеск негатива</b> — резкий рост доли негативных новостей по бумаге.\n"
    "  Пример: <code>GAZP</code>\n"
    "• <b>💰 Дивидендная отсечка</b> — уведомление за несколько дней до даты отсечки.\n"
    "  Пример: <code>LKOH</code>\n\n"
    "<b>Как работают подписки:</b>\n"
    "Алерты проверяются каждые 30 минут. Дайджест отправляется автоматически в 08:30 МСК.\n"
    "Управлять подписками можно кнопками ниже или командами /subscribe и /unsubscribe."
)

SUBSCRIBE_USAGE = (
    "🔔 <b>Подписка на алерт:</b>\n"
    "<code>/subscribe price_level ТИКЕР УРОВЕНЬ</code> — цена пересекла уровень\n"
    "<code>/subscribe sentiment_spike ТИКЕР</code> — всплеск негативных новостей\n"
    "<code>/subscribe dividend_upcoming ТИКЕР</code> — близкая дивидендная отсечка\n\n"
    "Пример: <code>/subscribe price_level SBER 250</code>\n\n"
    "Или используйте мастер /subscribe без аргументов."
)

UNSUBSCRIBE_USAGE = "Отписка: <code>/unsubscribe ID</code> — id берётся из списка выше."

#: RU-метки видов алертов для отображения подписок.
_ALERT_LABELS: dict[AlertKind, str] = {
    AlertKind.PRICE_LEVEL: "уровень цены",
    AlertKind.SENTIMENT_SPIKE: "всплеск негатива",
    AlertKind.DIVIDEND_UPCOMING: "дивидендная отсечка",
    AlertKind.VOLATILITY_REGIME: "режим волатильности",
}

#: Заглушки пустых состояний (RU-копи).
_NO_POSITIONS = "Портфель пуст: добавьте позиции в дашборде."
_NO_SUBSCRIPTIONS = "У вас нет активных подписок. Добавьте через /subscribe."
_DIGEST_NO_DIVIDENDS = "<i>ближайших отсечек нет</i>"
_DIGEST_NO_NEWS = "<i>негативных новостей нет</i>"

#: Предел длины заголовка новости в дайджесте (символов) — чтобы сообщение не разрослось.
_NEWS_TITLE_LIMIT = 90


def _money(value: Decimal) -> str:
    """Денежная сумма с разрядным пробелом и знаком рубля (без знака +)."""
    return f"{value:,.2f}".replace(",", " ") + " ₽"


#: Символы валют для сумм, которые могут быть не в рублях (дивиденды MOEX бывают USD/EUR/CNY).
_CURRENCY_SYMBOLS: dict[Currency, str] = {
    Currency.RUB: "₽",
    Currency.USD: "$",
    Currency.EUR: "€",
    Currency.CNY: "¥",
}


def _money_in(value: Decimal, currency: Currency) -> str:
    """Денежная сумма с символом указанной валюты (дивиденды бывают не в рублях)."""
    symbol = _CURRENCY_SYMBOLS.get(currency, currency.value)
    return f"{value:,.2f}".replace(",", " ") + f" {symbol}"


def _signed_money(value: Decimal) -> str:
    """Денежная сумма со знаком (+/−) — для P&L."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):,.2f}".replace(",", " ") + " ₽"


def _signed_pct(value: float) -> str:
    """Процент со знаком (+/−) и двумя знаками после запятой."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f}%"


def _optional_money(value: Decimal | None) -> str:
    """Денежная сумма или «—», если значение отсутствует (нет рыночной цены)."""
    return _money(value) if value is not None else "—"


def format_portfolio(summary: PortfolioSummaryOut) -> str:
    """Собрать HTML-сводку портфеля: стоимость, P&L, доходность против IMOEX, позиции."""
    head = (
        "<b>📊 Портфель</b>\n"
        f"Стоимость: {_money(summary.total_value)} (вложено {_money(summary.total_cost)})\n"
        f"Нереализованный P&amp;L: {_signed_money(summary.total_unrealized_pnl)}\n"
        f"Доходность: {_signed_pct(summary.portfolio_return_pct)} "
        f"(IMOEX {_signed_pct(summary.imoex_return_pct)})\n"
        f"Sharpe {summary.sharpe:.2f} · просадка {summary.max_drawdown:.1%}\n"
        f"Период: {summary.period_from:%d.%m.%Y} – {summary.period_to:%d.%m.%Y}"
    )
    if not summary.positions:
        return f"{head}\n\n{_NO_POSITIONS}"

    rows = [
        f"• <code>{html.escape(pos.ticker)}</code> — {pos.quantity} шт · "
        f"{_optional_money(pos.current_value)} · P&amp;L {_signed_money(pos.unrealized_pnl)}"
        if pos.unrealized_pnl is not None
        else f"• <code>{html.escape(pos.ticker)}</code> — {pos.quantity} шт · нет рыночной цены"
        for pos in summary.positions
    ]
    return f"{head}\n\n<b>Позиции</b>\n" + "\n".join(rows)


def format_subscriptions(subscriptions: Sequence[SubscriptionOut]) -> str:
    """Собрать HTML-список подписок с id (для /unsubscribe и /subscribe без аргументов)."""
    if not subscriptions:
        return _NO_SUBSCRIPTIONS
    rows = []
    for sub in subscriptions:
        label = _ALERT_LABELS.get(sub.kind, sub.kind.value)
        params = _format_params(sub.params)
        suffix = f" ({params})" if params else ""
        rows.append(f"<code>{sub.id}</code> · {html.escape(label)}{html.escape(suffix)}")
    return "<b>🔔 Ваши подписки:</b>\n" + "\n".join(rows)


def _format_params(params: dict[str, object]) -> str:
    """Компактно отрисовать параметры подписки (тикер/уровень) для списка."""
    parts = []
    ticker = params.get("ticker")
    if isinstance(ticker, str) and ticker:
        parts.append(ticker)
    level = params.get("level")
    if isinstance(level, int | float):
        parts.append(str(level))
    return ", ".join(parts)


def format_alert(alert: PendingAlert) -> str:
    """Сформировать HTML-сообщение для отправки по сработавшему алерту.

    Каждый вид алерта форматируется по собственному шаблону. Покрыты все виды AlertKind —
    новый вид без ветки поймает mypy (exhaustiveness), а не рантайм.
    """
    ticker = html.escape(alert.ticker)
    if alert.kind == AlertKind.PRICE_LEVEL:
        level = _money(alert.level) if alert.level is not None else "—"
        close = _money(alert.close) if alert.close is not None else "—"
        return (
            f"<b>Уровень цены: {ticker}</b>\n"
            f"Цена пересекла уровень {level}\n"
            f"Последний close: {close}"
        )
    if alert.kind == AlertKind.SENTIMENT_SPIKE:
        title = html.escape(alert.article_title or ticker)
        url = alert.article_url or ""
        link = f'<a href="{html.escape(url)}">{title}</a>' if url else title
        return f"<b>Негативные новости: {ticker}</b>\n{link}"
    if alert.kind == AlertKind.DIVIDEND_UPCOMING:
        ex = alert.ex_date.strftime("%d.%m.%Y") if alert.ex_date is not None else "—"
        if alert.dividend_value is not None and alert.dividend_currency is not None:
            value = _money_in(alert.dividend_value, alert.dividend_currency)
        else:
            value = "—"
        return f"<b>Дивидендная отсечка: {ticker}</b>\nДата отсечки: {ex} · Дивиденд: {value}"
    if alert.kind == AlertKind.VOLATILITY_REGIME:
        vol = f"{alert.volatility * 100:.1f}%" if alert.volatility is not None else "—"
        thr = f"{alert.threshold * 100:.1f}%" if alert.threshold is not None else "—"
        return (
            f"<b>Режим волатильности: {ticker}</b>\n"
            f"Прогноз 5-дневной волатильности {vol} превысил порог {thr}"
        )
    assert_never(alert.kind)


def format_digest(data: DigestData) -> str:
    """Собрать HTML-дайджест: заголовок + IMOEX + портфель + отсечки + негативные новости."""
    blocks = [
        "📋 <b>Дайджест по портфелю</b>",
        _format_imoex(data),
        format_portfolio(data.summary),
        "<b>💰 Дивидендные отсечки</b>\n" + _format_dividends(data.dividends),
        "<b>📰 Негативные новости</b>\n" + _format_news(data.negative_news),
    ]
    return "\n\n".join(blocks)


def _format_dividends(dividends: Sequence[UpcomingDividend]) -> str:
    """Список ближайших дивидендных отсечек по портфелю или заглушка."""
    if not dividends:
        return _DIGEST_NO_DIVIDENDS
    return "\n".join(
        f"<code>{html.escape(item.ticker)}</code> — {item.ex_date:%d.%m.%Y} · "
        f"{_money_in(item.value, item.currency)}"
        for item in dividends
    )


def _format_news(news: Sequence[NewsOut]) -> str:
    """Список негативных новостей (заголовок + тикеры) или заглушка."""
    if not news:
        return _DIGEST_NO_NEWS
    return "\n".join(
        f"• {html.escape(_truncate(article.title))}"
        + (f" [{html.escape(', '.join(article.tickers))}]" if article.tickers else "")
        for article in news
    )


def _truncate(text: str) -> str:
    """Обрезать заголовок до предела с многоточием, чтобы дайджест не разрастался."""
    if len(text) <= _NEWS_TITLE_LIMIT:
        return text
    return text[: _NEWS_TITLE_LIMIT - 1].rstrip() + "…"


def format_subscription_created(subscription: SubscriptionOut) -> str:
    """Подтверждение создания подписки (RU-копи, HTML)."""
    label = _ALERT_LABELS.get(subscription.kind, subscription.kind.value)
    params = _format_params(subscription.params)
    suffix = f" ({params})" if params else ""
    return (
        f"✅ Подписка создана: {html.escape(label)}{html.escape(suffix)} · "
        f"id <code>{subscription.id}</code>."
    )


def format_unsubscribed(sub_id: int) -> str:
    """Подтверждение удаления подписки (RU-копи)."""
    return f"✅ Подписка <code>{sub_id}</code> удалена."


def _format_imoex(data: DigestData) -> str:
    """Строка IMOEX за вчера: close и изменение относительно позапрошлого дня (spec §357)."""
    if data.imoex_yesterday is None:
        return "<b>📈 IMOEX</b>\nДанные индекса недоступны"
    close_str = f"{data.imoex_yesterday.close:,.2f}".replace(",", " ")
    date_str = f"{data.imoex_yesterday.trade_date:%d.%m.%Y}"
    if data.imoex_prior is not None:
        change_pct = (
            float(data.imoex_yesterday.close - data.imoex_prior.close)
            / float(data.imoex_prior.close)
            * 100
        )
        arrow = "📈" if change_pct >= 0 else "📉"
        return (
            f"<b>{arrow} IMOEX</b> · {date_str}\nЗакрытие {close_str} · {_signed_pct(change_pct)}"
        )
    return f"<b>📈 IMOEX</b> · {date_str}\nЗакрытие {close_str}"
