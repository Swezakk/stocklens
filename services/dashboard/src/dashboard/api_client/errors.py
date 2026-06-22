"""Доменные ошибки HTTP-клиента дашборда с RU-текстами для пользователя (DESIGN.md §5, §7).

Каждая страница обрабатывает три ветки сетевого вызова (успех / ошибка сервера /
сеть недоступна) через эти типы; пустых экранов без объяснения нет. user_message —
готовая к показу русская строка с сущностью и причиной (правило обработки ошибок).
"""


class ApiError(Exception):
    """Базовая ошибка взаимодействия дашборда с API.

    user_message — русский текст для показа пользователю; str(exc) — то же сообщение,
    чтобы лог и UI были согласованы.
    """

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__(user_message)


_UNAVAILABLE_MESSAGE = "Сервис данных недоступен: не удалось связаться с API. Повторите позже."
_AUTH_MESSAGE = "Сессия истекла или пароль неверен: войдите заново."


class ApiUnavailableError(ApiError):
    """API недоступен: сетевая ошибка или таймаут запроса."""

    def __init__(self, user_message: str = _UNAVAILABLE_MESSAGE) -> None:
        super().__init__(user_message)


class ApiServerError(ApiError):
    """API вернул ошибку 5xx: сбой на стороне сервера данных."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(
            f"Сервис данных вернул ошибку {status}: запрос не выполнен. Повторите позже."
        )


class AuthError(ApiError):
    """Ошибка аутентификации: неверный пароль или истёкшая сессия (сброс на гейт)."""

    def __init__(self, user_message: str = _AUTH_MESSAGE) -> None:
        super().__init__(user_message)
