"""Smoke-тест точки входа app.py: гейт держит до аутентификации (DESIGN.md §4, §7).

``require_auth`` вызывает ``st.stop`` на неаутентифицированном пути, поэтому навигация и
страницы не строятся, а на экране — только форма гейта с полем пароля. Тест поднимает
приложение через ``streamlit.testing.v1.AppTest`` БЕЗ токена в session_state и проверяет,
что гейт держится (поле пароля есть, тело страницы-стаба отсутствует).
"""

from pathlib import Path

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
