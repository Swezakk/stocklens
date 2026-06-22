"""Парольный гейт и жизненный цикл JWT-токена дашборда (DESIGN.md §7, §8).

Единый секрет: гейт дашборда = логин в API. Дашборд знает ``API_URL`` и
``AUTH_OWNER_USERNAME``; пароль вводит пользователь и удерживается в ``st.session_state``
для проактивного refresh. Отдельного ``DASHBOARD_PASSWORD`` нет.

Жизненный цикл (DESIGN §7):
- Проактивный refresh: токен перевыпускается, когда ``now > expiry - margin`` (margin ≥ 60s
  = leeway API), поэтому реактивных 401 почти не бывает.
- Single-flight: новый токен пишется в state ДО возврата, поэтому параллельные вызовы в
  одном rerun переиспользуют его и не упираются в лимит логина API (5/60s на IP).
- Реактивный 401: ``force_refresh`` — хук ``ApiClient.on_unauthorized`` (форс-перевыпуск
  перед единственным ретраем data-вызова).

``TokenManager`` отделён от Streamlit: state — любой объект с протоколом ``MutableState``
(``st.session_state`` подходит), а часы инъектируются (``Callable[[], float]``) — проверка
истечения unit-тестируема без рантайма (DESIGN §8). Гейт ``require_auth`` — единственная
часть, завязанная на ``st.*``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import time
from typing import Protocol

import httpx
import streamlit as st
from pydantic import BaseModel

from dashboard.api_client.client import ApiClient
from dashboard.api_client.errors import ApiUnavailableError, AuthError
from dashboard.api_client.fetch import get_client
from dashboard.settings import get_settings

#: Путь выдачи токена под префиксом версии (form-encoded OAuth2 Password Flow).
_TOKEN_PATH = "/auth/token"

#: HTTP 401: неверные учётные данные владельца (контракт POST /auth/token).
_HTTP_UNAUTHORIZED = 401

#: Граница ошибок: всё с 400 — сбой выдачи токена (как в api_client/client.py).
_HTTP_BAD_REQUEST = 400

#: Текст ошибки гейта при отклонённом пароле (DESIGN §7: «Неверный пароль»).
_WRONG_PASSWORD_MESSAGE = "Неверный пароль"

#: Ключи состояния сессии (без строковых литералов в логике — DESIGN §7).
_STATE_PASSWORD = "password"
_STATE_TOKEN = "token"
_STATE_TOKEN_EXPIRY = "token_expiry"

#: RU-копи гейта.
_GATE_TITLE = "Вход в StockLens"
_GATE_PROMPT = "Введите пароль владельца, чтобы открыть дашборд."
_GATE_PASSWORD_LABEL = "Пароль"
_GATE_SUBMIT_LABEL = "Войти"
_GATE_EMPTY_PASSWORD = "Введите пароль."
_GATE_FORM_KEY = "auth_gate"


class MutableState(Protocol):
    """Минимальный протокол изменяемого хранилища состояния (под ``st.session_state``).

    Сознательно узкий: ``get`` / ``__setitem__`` / ``__contains__`` — ровно то, что нужно
    TokenManager. Сигнатуры позиционно-параметрические, чтобы структурно удовлетворялись и
    ``dict``, и Streamlit-прокси (их ``get`` объявлен с позиционными аргументами).
    """

    def get(self, key: str, /) -> object: ...

    def __setitem__(self, key: str, value: object, /) -> None: ...

    def __contains__(self, key: object, /) -> bool: ...


class TokenResponse(BaseModel):
    """Зеркало ответа POST /auth/token (TokenOut API): токен и время жизни."""

    access_token: str
    token_type: str
    expires_in: int


@dataclass(frozen=True)
class AuthConfig:
    """Неизменяемая конфигурация аутентификации (группирует параметры конструктора)."""

    api_base_url: str
    api_prefix: str
    auth_username: str
    token_refresh_margin_seconds: int
    request_timeout_seconds: float


def build_auth_config() -> AuthConfig:
    """Собрать AuthConfig из настроек дашборда (pydantic-settings, без чтения env по коду)."""
    settings = get_settings()
    return AuthConfig(
        api_base_url=settings.api_base_url,
        api_prefix=settings.api_prefix,
        auth_username=settings.auth_username,
        token_refresh_margin_seconds=settings.token_refresh_margin_seconds,
        request_timeout_seconds=settings.request_timeout_seconds,
    )


class TokenManager:
    """Управление JWT: кэш, проактивный refresh, single-flight, реактивный 401.

    Параметры:
    - ``config`` — хост/префикс API, имя владельца, margin, таймаут;
    - ``state`` — изменяемое хранилище (``st.session_state`` или dict в тестах) с паролем
      под ключом ``password`` и кэшем токена под ``token`` / ``token_expiry``;
    - ``clock`` — источник текущего epoch-времени (по умолчанию ``time.time``); инъекция
      делает проверку истечения unit-тестируемой (DESIGN §8).
    """

    def __init__(
        self,
        config: AuthConfig,
        state: MutableState,
        clock: Callable[[], float] = time,
    ) -> None:
        self._config = config
        self._state = state
        self._clock = clock

    def get_token(self) -> str:
        """Вернуть валидный Bearer-токен, перевыпустив его при отсутствии или близости к истечению.

        Single-flight: ``mint`` пишет токен в state до возврата, поэтому следующий вызов в
        том же rerun читает свежий кэш, а не минтит повторно.
        """
        cached = self._read_cached_token()
        if cached is not None and not self._is_expiring(cached.expiry):
            return cached.token
        return self.mint()

    def force_refresh(self) -> None:
        """Немедленно перевыпустить токен (хук ``ApiClient.on_unauthorized`` при 401)."""
        self.mint()

    def mint(self) -> str:
        """Запросить новый токен у API и сохранить его с временем истечения в state.

        Раскладка по трём веткам: 401 → ``AuthError("Неверный пароль")``; сетевая ошибка /
        таймаут → ``ApiUnavailableError`` (гейт не показывает сырой traceback, DESIGN §7).
        """
        password = self._read_password()
        url = f"{self._config.api_base_url}{self._config.api_prefix}{_TOKEN_PATH}"
        form = {"username": self._config.auth_username, "password": password}

        try:
            response = httpx.post(url, data=form, timeout=self._config.request_timeout_seconds)
        except httpx.RequestError as exc:
            raise ApiUnavailableError() from exc

        if response.status_code == _HTTP_UNAUTHORIZED:
            raise AuthError(_WRONG_PASSWORD_MESSAGE)
        if response.status_code >= _HTTP_BAD_REQUEST:
            raise ApiUnavailableError()

        token = TokenResponse.model_validate(response.json())
        expiry = self._clock() + token.expires_in
        self._state[_STATE_TOKEN] = token.access_token
        self._state[_STATE_TOKEN_EXPIRY] = expiry
        return token.access_token

    def _read_cached_token(self) -> "_CachedToken | None":
        """Прочитать токен и его истечение из state с narrowing типов (без Any-индексации)."""
        raw_token = self._state.get(_STATE_TOKEN)
        raw_expiry = self._state.get(_STATE_TOKEN_EXPIRY)
        if isinstance(raw_token, str) and isinstance(raw_expiry, int | float):
            return _CachedToken(token=raw_token, expiry=float(raw_expiry))
        return None

    def _read_password(self) -> str:
        """Достать удержанный пароль из state; его отсутствие = слетевшая сессия → на гейт."""
        raw_password = self._state.get(_STATE_PASSWORD)
        if isinstance(raw_password, str) and raw_password:
            return raw_password
        raise AuthError(_WRONG_PASSWORD_MESSAGE)

    def _is_expiring(self, expiry: float) -> bool:
        """True, если до истечения осталось меньше margin (нужен проактивный refresh)."""
        return self._clock() > expiry - self._config.token_refresh_margin_seconds


@dataclass(frozen=True)
class _CachedToken:
    """Снимок кэшированного токена из state: значение и epoch истечения."""

    token: str
    expiry: float


def build_token_provider(manager: TokenManager) -> Callable[[], str]:
    """Связанный поставщик токена для ``ApiClient.token_provider`` (DESIGN §6, §7)."""
    return manager.get_token


def build_on_unauthorized(manager: TokenManager) -> Callable[[], None]:
    """Связанный хук форс-обновления для ``ApiClient.on_unauthorized`` (реактивный 401)."""
    return manager.force_refresh


def get_token_manager() -> TokenManager:
    """TokenManager, привязанный к ``st.session_state`` (для гейта и провайдеров клиента)."""
    return TokenManager(config=build_auth_config(), state=st.session_state)


def get_api_client() -> ApiClient:
    """Единая точка получения готового ApiClient на странице (DESIGN §6, §7, §8).

    Связывает провайдер токена и хук 401 с TokenManager на ``st.session_state`` и отдаёт
    тот же singleton из ``st.cache_resource`` (``get_client``), поэтому rerun не плодит
    пулы соединений. Здесь зависимость auth → fetch (а не наоборот): ``fetch.get_client``
    сознательно не знает про auth, чтобы кэш-слой не зависел от реализации аутентификации.
    """
    manager = get_token_manager()
    return get_client(
        build_token_provider(manager),
        build_on_unauthorized(manager),
    )


def require_auth() -> None:
    """Гейт аутентификации: до успешного логина страницы и навигация не строятся (DESIGN §7).

    Если токен уже в сессии — пропускает. Иначе рендерит форму с полем пароля; на сабмит
    пытается выпустить токен, при успехе — rerun (страницы строятся уже с доступом), при
    ошибке — RU-сообщение. ``st.stop`` завершает rerun, пока вход не выполнен.
    """
    if _STATE_TOKEN in st.session_state:
        return

    manager = get_token_manager()
    st.title(_GATE_TITLE)
    st.caption(_GATE_PROMPT)

    with st.form(_GATE_FORM_KEY):
        password = st.text_input(_GATE_PASSWORD_LABEL, type="password")
        submitted = st.form_submit_button(_GATE_SUBMIT_LABEL)

    if submitted:
        _attempt_login(manager, password)

    st.stop()


def _attempt_login(manager: TokenManager, password: str) -> None:
    """Сохранить пароль, выпустить токен и перезапустить скрипт; ошибки — RU-копи гейта."""
    if not password:
        st.error(_GATE_EMPTY_PASSWORD)
        return

    st.session_state[_STATE_PASSWORD] = password
    try:
        manager.mint()
    except (AuthError, ApiUnavailableError) as exc:
        st.error(exc.user_message)
        return
    st.rerun()
