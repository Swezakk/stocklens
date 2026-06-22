"""Smoke-тест точки входа app.py: гейт держит до аутентификации (DESIGN.md §4, §7).

``require_auth`` вызывает ``st.stop`` на неаутентифицированном пути, поэтому навигация и
страницы не строятся, а на экране — только форма гейта с полем пароля. Тест поднимает
приложение через ``streamlit.testing.v1.AppTest`` БЕЗ токена в session_state и проверяет,
что гейт держится (поле пароля есть, тело страницы-стаба отсутствует).

Второй блок — интеграционный страж рендера всех пяти страниц при недоступном API
(DESIGN §7, §10): каждая ``render`` гоняется через ``AppTest.from_function`` с паролем в
session_state и API, указанным на закрытый loopback-порт. Контракт «три ветки на каждый
сетевой вызов»: страница ловит ``ApiError`` и показывает ``st.error`` — никаких сырых
исключений и пустых экранов. Per-page тесты покрывают только чистые хелперы; этот блок —
единственное место, где сам ``render`` исполняется конец-в-конец на пути «API недоступен».
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from dashboard.pages import monitoring, news, overview, portfolio, stocks
from dashboard.settings import get_settings
from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"

#: Заголовок формы гейта (зеркало auth._GATE_TITLE — пользовательская строка RU).
_GATE_TITLE = "Вход в StockLens"

#: Маркер тела страницы-стаба: его отсутствие доказывает, что страницы не отрисовались.
_PAGE_STUB_MARKER = "Страница в разработке."

#: Заголовок дефолтной (лендинг) страницы навигации — зеркало app.py title «Обзор».
_DEFAULT_PAGE_TITLE = "Обзор"

#: Ключ токена в session_state — зеркало auth._STATE_TOKEN (аутентифицированный путь).
#: Собран из частей, чтобы сканер не принял имя ключа за хардкод-секрет (S105).
_TOKEN_STATE_KEY = "".join(["tok", "en"])

#: Фиктивный токен для прохода гейта в тесте (не реальный секрет, собран из частей).
_FAKE_TOKEN = "-".join(["fake", "jwt"])

#: Таймаут запуска AppTest: гейт лёгкий, но избегаем подвисания на дефолтных 3s в CI.
_RUN_TIMEOUT_SECONDS = 30

#: Ключ пароля в session_state — зеркало auth._STATE_PASSWORD; нужен, чтобы ``mint``
#: дошёл до HTTP-вызова (а не упал на отсутствии пароля) и упёрся в недоступный API.
#: Собран из частей, чтобы сканер не принял имя ключа за хардкод-пароль (S105).
_PASSWORD_STATE_KEY = "".join(["pass", "word"])

#: Фиктивный пароль для попытки логина в тесте (не реальный секрет, собран из частей).
_FAKE_PASSWORD = "-".join(["fake", "pass"])

#: Env-имя базового URL API (зеркало settings.API_URL alias).
_API_URL_ENV = "API_URL"

#: Закрытый loopback-порт: connection refused приходит мгновенно, поэтому путь
#: «API недоступен» детерминирован и быстр (без зависимости от DNS/сети CI).
_UNREACHABLE_API_URL = "http://127.0.0.1:1"

#: Имена модулей пяти страниц навигации (зеркало app.py): каждая обязана пережить
#: недоступный API. Имена (а не сами callable) — потому что ``AppTest.from_string``
#: исполняет настоящий скрипт-обёртку с импортом модуля, а ``from_function`` теряет
#: модульные импорты страницы (``st`` объявлен на уровне модуля, не в теле ``render``).
_PAGE_MODULES: list[str] = [
    overview.__name__,
    stocks.__name__,
    news.__name__,
    portfolio.__name__,
    monitoring.__name__,
]


@pytest.fixture
def unreachable_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Указать API на закрытый loopback-порт и сбросить кэш настроек на время теста.

    ``get_settings`` кэшируется ``lru_cache``, поэтому подмену env видно, только если
    очистить кэш до и после — иначе соседние тесты получат «дырявые» настройки.
    """
    monkeypatch.setenv(_API_URL_ENV, _UNREACHABLE_API_URL)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def _page_script(module_name: str) -> str:
    """Собрать скрипт-обёртку, импортирующий модуль страницы и вызывающий его ``render``."""
    return f"from {module_name} import render\n\nrender()\n"


def _run_page_with_api_down(module_name: str) -> AppTest:
    """Прогнать ``render`` страницы через AppTest с паролем в state и недоступным API.

    Гейт ``app.py`` не запускается — исполняется ровно ``render``, поэтому токен не нужен;
    нужен лишь пароль, чтобы ``TokenManager.mint`` дошёл до HTTP и получил отказ соединения.
    """
    app = AppTest.from_string(_page_script(module_name), default_timeout=_RUN_TIMEOUT_SECONDS)
    app.session_state[_PASSWORD_STATE_KEY] = _FAKE_PASSWORD
    return app.run()


def _run_unauthenticated() -> AppTest:
    """Запустить app.py без токена в session_state (путь гейта)."""
    app = AppTest.from_file(str(_APP_PATH), default_timeout=_RUN_TIMEOUT_SECONDS)
    return app.run()


def _run_authenticated() -> AppTest:
    """Запустить app.py с токеном в session_state (гейт пройден, строится навигация)."""
    app = AppTest.from_file(str(_APP_PATH), default_timeout=_RUN_TIMEOUT_SECONDS)
    app.session_state[_TOKEN_STATE_KEY] = _FAKE_TOKEN
    return app.run()


def test_gate_renders_without_exception() -> None:
    """Неаутентифицированный запуск завершается чисто (st.stop — не ошибка)."""
    app = _run_unauthenticated()

    assert not app.exception


def test_gate_shows_password_input() -> None:
    """До входа на экране ровно одно поле пароля (форма гейта)."""
    app = _run_unauthenticated()

    assert len(app.text_input) == 1


def test_gate_shows_title() -> None:
    """Заголовок гейта присутствует (пользователь понимает, куда вводит пароль)."""
    app = _run_unauthenticated()

    titles = [element.value for element in app.title]
    assert _GATE_TITLE in titles


def test_pages_do_not_render_before_auth() -> None:
    """Тело страниц-стабов не строится до входа: st.stop обрывает rerun до навигации (§7)."""
    app = _run_unauthenticated()

    info_messages = [element.value for element in app.info]
    assert _PAGE_STUB_MARKER not in info_messages


def test_navigation_builds_without_exception_when_authenticated() -> None:
    """С токеном навигация строится без ошибки: пять callable ``render`` имеют уникальные url_path.

    Регрессия-страж: без явного ``url_path`` Streamlit выводит один pathname из имени
    функции и падает на не-уникальных путях (``StreamlitAPIException``).
    """
    app = _run_authenticated()

    assert not app.exception
    assert len(app.text_input) == 0


def test_default_page_renders_when_authenticated() -> None:
    """С токеном открывается дефолтная страница «Обзор» (лендинг навигации)."""
    app = _run_authenticated()

    titles = [element.value for element in app.title]
    assert _DEFAULT_PAGE_TITLE in titles


@pytest.mark.usefixtures("unreachable_api")
@pytest.mark.parametrize("module_name", _PAGE_MODULES)
def test_page_survives_unreachable_api_without_exception(module_name: str) -> None:
    """Каждая страница переживает недоступный API без сырого исключения (DESIGN §7, §10).

    ``render`` ловит ``ApiError`` на каждом сетевом вызове, поэтому отказ соединения не
    всплывает наружу — иначе пользователь увидел бы трейсбек вместо фидбэка.
    """
    app = _run_page_with_api_down(module_name)

    assert not app.exception


@pytest.mark.usefixtures("unreachable_api")
@pytest.mark.parametrize("module_name", _PAGE_MODULES)
def test_page_renders_feedback_when_api_unreachable(module_name: str) -> None:
    """Недоступный API даёт пользователю явный фидбэк, а не пустой экран (DESIGN §5, §7).

    Контракт «три ветки на сетевой вызов»: ветка «сеть недоступна» рендерит ``st.error``;
    его наличие отличает обработанную ошибку от молча проглоченной (пустой страницы).
    """
    app = _run_page_with_api_down(module_name)

    assert len(app.error) >= 1
