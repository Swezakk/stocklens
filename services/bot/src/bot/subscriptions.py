"""Разбор аргументов команд /subscribe и /unsubscribe (чистые функции, DESIGN §11).

Хендлеры тонкие: разбор и валидацию формы аргументов делает этот слой (unit-тестируемо),
а правила алертов и хранение — API (единственный источник истины). VOLATILITY_REGIME
подписывается по тикеру; порог (квантиль/окно) задаётся сервером по умолчанию (ml-spec §9).
"""

from dataclasses import dataclass, field

from stocklens_core.enums import AlertKind

#: Виды алертов, доступные для подписки (детерминированные).
_SUBSCRIBABLE_KINDS = frozenset(
    {
        AlertKind.PRICE_LEVEL,
        AlertKind.SENTIMENT_SPIKE,
        AlertKind.DIVIDEND_UPCOMING,
        AlertKind.VOLATILITY_REGIME,
    }
)

#: Ключи параметров подписки (без хардкода строк в логике).
_PARAM_TICKER = "ticker"
_PARAM_LEVEL = "level"

#: Минимум токенов: price_level (kind + тикер + уровень) и опциональный тикер (kind + тикер).
_PRICE_LEVEL_TOKENS = 3
_KIND_AND_TICKER_TOKENS = 2

_ERR_UNKNOWN_KIND = (
    "Неизвестный вид алерта. Доступно: price_level, sentiment_spike, "
    "dividend_upcoming, volatility_regime."
)
_ERR_PRICE_LEVEL_ARGS = "Для price_level укажите тикер и уровень: /subscribe price_level SBER 250"
_ERR_LEVEL_NOT_NUMBER = "Уровень должен быть числом: /subscribe price_level SBER 250"
_ERR_UNSUBSCRIBE_ID = "Укажите числовой id подписки: /unsubscribe 3"


@dataclass(frozen=True)
class ParsedSubscribe:
    """Разобранная подписка: вид алерта и параметры для SubscriptionIn."""

    kind: AlertKind
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseError:
    """Ошибка разбора аргументов с готовым RU-сообщением для пользователя."""

    message: str


def parse_subscribe(args: str) -> ParsedSubscribe | ParseError:
    """Разобрать аргументы /subscribe («kind [тикер] [уровень]») в подписку или ошибку."""
    tokens = args.split()
    try:
        kind = AlertKind(tokens[0].lower())
    except ValueError:
        return ParseError(_ERR_UNKNOWN_KIND)

    if kind not in _SUBSCRIBABLE_KINDS:
        return ParseError(_ERR_UNKNOWN_KIND)
    if kind is AlertKind.PRICE_LEVEL:
        return _parse_price_level(tokens)
    return _parse_ticker_required(kind, tokens)


def _parse_price_level(tokens: list[str]) -> ParsedSubscribe | ParseError:
    """price_level: обязательны тикер и числовой уровень."""
    if len(tokens) < _PRICE_LEVEL_TOKENS:
        return ParseError(_ERR_PRICE_LEVEL_ARGS)
    try:
        level = float(tokens[2])
    except ValueError:
        return ParseError(_ERR_LEVEL_NOT_NUMBER)
    return ParsedSubscribe(
        kind=AlertKind.PRICE_LEVEL,
        params={_PARAM_TICKER: tokens[1].upper(), _PARAM_LEVEL: level},
    )


def _parse_ticker_required(kind: AlertKind, tokens: list[str]) -> ParsedSubscribe | ParseError:
    """sentiment_spike / dividend_upcoming: тикер обязателен (подписка — на конкретную бумагу)."""
    if len(tokens) < _KIND_AND_TICKER_TOKENS:
        return ParseError(f"Для {kind.value} укажите тикер: /subscribe {kind.value} SBER")
    return ParsedSubscribe(kind=kind, params={_PARAM_TICKER: tokens[1].upper()})


def parse_unsubscribe(args: str) -> int | ParseError:
    """Разобрать аргумент /unsubscribe (id подписки) в int или ошибку."""
    tokens = args.split()
    if not tokens:
        return ParseError(_ERR_UNSUBSCRIBE_ID)
    try:
        return int(tokens[0])
    except ValueError:
        return ParseError(_ERR_UNSUBSCRIBE_ID)
