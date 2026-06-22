"""Доменные ошибки async-клиента бота к StockLens API (DESIGN.md §11).

Бот раскладывает каждый сетевой вызов на три ветки (как дашборд): сеть недоступна /
ошибка сервера / отвергнутая аутентификация. ``user_message`` — готовая RU-строка, которую
бот отправляет пользователю в чат при сбое (вместо сырого traceback).
"""


class ApiError(Exception):
    """Базовая ошибка взаимодействия бота с API.

    user_message — русский текст для отправки в чат; str(exc) — то же сообщение,
    чтобы лог и сообщение пользователю были согласованы.
    """

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__(user_message)


_UNAVAILABLE_MESSAGE = "Сервис данных недоступен: не удалось связаться с API. Повторите позже."
_AUTH_MESSAGE = "Не удалось авторизоваться в API: проверьте учётные данные владельца."


class ApiUnavailableError(ApiError):
    """API недоступен: сетевая ошибка или таймаут запроса."""

    def __init__(self, user_message: str = _UNAVAILABLE_MESSAGE) -> None:
        super().__init__(user_message)


class ApiServerError(ApiError):
    """API вернул 4xx/5xx (после обработки 401): запрос к данным не выполнен."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(
            f"Сервис данных вернул ошибку {status}: запрос не выполнен. Повторите позже."
        )


class AuthError(ApiError):
    """Ошибка аутентификации: API отверг учётные данные владельца (401)."""

    def __init__(self, user_message: str = _AUTH_MESSAGE) -> None:
        super().__init__(user_message)
