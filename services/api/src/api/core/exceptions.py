"""Иерархия доменных исключений API.

Все 4xx-ошибки наследуют ApiError. Тексты — на русском, с сущностью и причиной.
SchemaNotReadyError используется только при старте (не HTTP-ошибка).
"""


class ApiError(Exception):
    """Базовое исключение API. Преобразуется в Problem Details response."""

    status: int
    title: str
    problem_type: str

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class SecurityNotFoundError(ApiError):
    """Ценная бумага с указанным тикером не найдена в БД."""

    status = 404
    title = "Бумага не найдена"
    problem_type = "https://stocklens.local/problems/security-not-found"

    def __init__(self, ticker: str) -> None:
        super().__init__(f"Бумага с тикером {ticker!r} не найдена")
        self.ticker = ticker


class InvalidSortFieldError(ApiError):
    """Запрошена сортировка по недопустимому полю."""

    status = 400
    title = "Недопустимое поле сортировки"
    problem_type = "https://stocklens.local/problems/invalid-sort-field"

    def __init__(self, field: str, allowed: list[str]) -> None:
        super().__init__(f"Сортировка по полю {field!r} недопустима. Допустимые поля: {allowed}")
        self.field = field
        self.allowed = allowed


class InsufficientDataError(ApiError):
    """Недостаточно данных для выполнения запроса."""

    status = 422
    title = "Недостаточно данных"
    problem_type = "https://stocklens.local/problems/insufficient-data"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class PositionNotFoundError(ApiError):
    """Позиция по указанному тикеру не найдена в портфеле."""

    status = 404
    title = "Позиция не найдена"
    problem_type = "https://stocklens.local/problems/position-not-found"

    def __init__(self, ticker: str) -> None:
        super().__init__(f"Позиция по тикеру {ticker!r} не найдена")
        self.ticker = ticker


class SubscriptionNotFoundError(ApiError):
    """Подписка с указанным идентификатором не найдена."""

    status = 404
    title = "Подписка не найдена"
    problem_type = "https://stocklens.local/problems/subscription-not-found"

    def __init__(self, sub_id: int) -> None:
        super().__init__(f"Подписка {sub_id} не найдена")
        self.sub_id = sub_id


class WatchlistItemNotFoundError(ApiError):
    """Тикер отсутствует в списке наблюдения."""

    status = 404
    title = "Тикер не найден в вотчлисте"
    problem_type = "https://stocklens.local/problems/watchlist-item-not-found"

    def __init__(self, ticker: str) -> None:
        super().__init__(f"Тикер {ticker!r} отсутствует в списке наблюдения")
        self.ticker = ticker


class WatchlistItemExistsError(ApiError):
    """Тикер уже добавлен в список наблюдения."""

    status = 409
    title = "Тикер уже в вотчлисте"
    problem_type = "https://stocklens.local/problems/watchlist-item-exists"

    def __init__(self, ticker: str) -> None:
        super().__init__(f"Тикер {ticker!r} уже в списке наблюдения")
        self.ticker = ticker


class InvalidStrategyParamsError(ApiError):
    """Недопустимые или отсутствующие параметры стратегии оптимизации."""

    status = 422
    title = "Недопустимые параметры стратегии"
    problem_type = "https://stocklens.local/problems/invalid-strategy-params"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class SchemaNotReadyError(Exception):
    """БД не готова к работе: схема не применена после N попыток."""

    def __init__(self, attempts: int) -> None:
        super().__init__(f"Схема БД недоступна после {attempts} попыток")
        self.attempts = attempts
