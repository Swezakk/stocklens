"""Сервис управления Telegram-подписками на алерты."""

from stocklens_core.enums import AlertKind
from stocklens_core.models.portfolio import BotSubscription

from api.core.exceptions import InvalidAlertParamsError, SubscriptionNotFoundError
from api.repositories.protocols import BotSubscriptionRepository, SecurityRepository
from api.schemas.bot import DIVIDEND_LEAD_DAYS_DEFAULT, SubscriptionIn, SubscriptionOut

_PARAM_TICKER = "ticker"
_PARAM_LEVEL = "level"
_PARAM_LEAD_DAYS = "lead_days"
_PARAM_QUANTILE = "quantile"
_PARAM_LOOKBACK = "lookback"

_LEAD_DAYS_MIN = 1
_LEAD_DAYS_MAX = 30
_QUANTILE_MIN = 0.5
_QUANTILE_MAX = 0.99
_LOOKBACK_MIN = 60
_LOOKBACK_MAX = 1000


class BotSubscriptionService:
    """Управляет Telegram-подписками пользователей.

    Валидирует параметры по типу алерта и разрешает ticker в securities.
    Все три активных типа (price_level, sentiment_spike, dividend_upcoming) требуют
    обязательный параметр 'ticker', который должен быть известен в БД.
    """

    def __init__(
        self,
        repo: BotSubscriptionRepository,
        security_repo: SecurityRepository,
    ) -> None:
        self._repo = repo
        self._security_repo = security_repo

    async def list_by_chat(self, chat_id: int) -> list[SubscriptionOut]:
        """Вернуть все подписки для указанного chat_id."""
        subscriptions = await self._repo.list_by_chat(chat_id)
        return [_to_out(s) for s in subscriptions]

    async def create(self, data: SubscriptionIn) -> SubscriptionOut:
        """Создать подписку.

        Raises:
            InvalidAlertParamsError: если параметры не соответствуют типу алерта
                или ticker не найден в БД.
        """
        ticker = _require_ticker(data.kind, data.params)
        security = await self._security_repo.get_by_ticker(ticker)
        if security is None:
            raise InvalidAlertParamsError(
                f"Тикер {ticker!r} не найден в БД. "
                f"Проверьте тикер или дождитесь синхронизации данных."
            )
        _validate_kind_specific_params(data.kind, data.params)

        subscription = await self._repo.create(
            chat_id=data.chat_id,
            kind=data.kind,
            params=data.params,
        )
        return _to_out(subscription)

    async def delete(self, sub_id: int) -> None:
        """Удалить подписку.

        Raises:
            SubscriptionNotFoundError: если подписка не найдена.
        """
        deleted = await self._repo.delete(sub_id)
        if not deleted:
            raise SubscriptionNotFoundError(sub_id)


def _require_ticker(kind: AlertKind, params: dict[str, object]) -> str:
    """Извлечь обязательный строковый параметр 'ticker'.

    Raises:
        InvalidAlertParamsError: если параметр отсутствует или не является строкой.
    """
    raw = params.get(_PARAM_TICKER)
    if raw is None:
        raise InvalidAlertParamsError(
            f"Для подписки {kind} обязателен параметр 'ticker' (тикер ценной бумаги)"
        )
    if not isinstance(raw, str):
        raise InvalidAlertParamsError(
            f"Параметр 'ticker' должен быть строкой, получено: {type(raw).__name__}"
        )
    return raw


def _validate_kind_specific_params(kind: AlertKind, params: dict[str, object]) -> None:
    """Проверить дополнительные параметры, специфичные для типа алерта.

    Raises:
        InvalidAlertParamsError: если параметры нарушают контракт типа алерта.
    """
    if kind == AlertKind.PRICE_LEVEL:
        _validate_price_level_params(params)
    elif kind == AlertKind.DIVIDEND_UPCOMING:
        _validate_dividend_upcoming_params(params)
    elif kind == AlertKind.VOLATILITY_REGIME:
        _validate_volatility_regime_params(params)


def _validate_price_level_params(params: dict[str, object]) -> None:
    """Проверить параметры price_level: level должен быть положительным числом.

    Raises:
        InvalidAlertParamsError: если 'level' отсутствует, не числовой или <= 0.
    """
    raw = params.get(_PARAM_LEVEL)
    if raw is None:
        raise InvalidAlertParamsError(
            "Для подписки price_level обязателен параметр 'level' "
            "(числовой целевой уровень цены, > 0)"
        )
    if not isinstance(raw, int | float):
        raise InvalidAlertParamsError(
            f"Параметр 'level' должен быть числом, получено: {type(raw).__name__}"
        )
    if raw <= 0:
        raise InvalidAlertParamsError(
            f"Параметр 'level' должен быть положительным числом, получено: {raw}"
        )


def _validate_dividend_upcoming_params(params: dict[str, object]) -> None:
    """Проверить параметры dividend_upcoming: lead_days в диапазоне [1..30].

    Raises:
        InvalidAlertParamsError: если 'lead_days' не числовой или вне диапазона.
    """
    raw = params.get(_PARAM_LEAD_DAYS, DIVIDEND_LEAD_DAYS_DEFAULT)
    if not isinstance(raw, int):
        raise InvalidAlertParamsError(
            f"Параметр 'lead_days' должен быть целым числом, получено: {type(raw).__name__}"
        )
    if not (_LEAD_DAYS_MIN <= raw <= _LEAD_DAYS_MAX):
        raise InvalidAlertParamsError(
            f"Параметр 'lead_days' должен быть в диапазоне "
            f"[{_LEAD_DAYS_MIN}..{_LEAD_DAYS_MAX}], получено: {raw}"
        )


def _validate_volatility_regime_params(params: dict[str, object]) -> None:
    """Проверить параметры volatility_regime: опциональные quantile и lookback.

    Raises:
        InvalidAlertParamsError: если quantile или lookback присутствуют, но невалидны.
    """
    raw_quantile = params.get(_PARAM_QUANTILE)
    if raw_quantile is not None:
        if not isinstance(raw_quantile, int | float):
            raise InvalidAlertParamsError(
                f"Параметр 'quantile' должен быть числом, получено: {type(raw_quantile).__name__}"
            )
        q = float(raw_quantile)
        if not (_QUANTILE_MIN <= q <= _QUANTILE_MAX):
            raise InvalidAlertParamsError(
                f"Параметр 'quantile' должен быть в диапазоне "
                f"[{_QUANTILE_MIN}..{_QUANTILE_MAX}], получено: {raw_quantile}"
            )

    raw_lookback = params.get(_PARAM_LOOKBACK)
    if raw_lookback is not None:
        if not isinstance(raw_lookback, int):
            raise InvalidAlertParamsError(
                "Параметр 'lookback' должен быть целым числом, "
                f"получено: {type(raw_lookback).__name__}"
            )
        if not (_LOOKBACK_MIN <= raw_lookback <= _LOOKBACK_MAX):
            raise InvalidAlertParamsError(
                f"Параметр 'lookback' должен быть в диапазоне "
                f"[{_LOOKBACK_MIN}..{_LOOKBACK_MAX}], получено: {raw_lookback}"
            )


def _to_out(subscription: BotSubscription) -> SubscriptionOut:
    """Преобразовать ORM-объект в DTO."""
    return SubscriptionOut.model_validate(subscription)
