"""Точка входа дашборда StockLens (DESIGN.md §4, §6, §7, §10).

Порядок строго фиксирован:

1. ``st.set_page_config`` — обязан быть первым ``st.*``-вызовом (wide-раскладка, §4).
2. Инжект тонкого CSS-слоя один раз (``assets/dashboard.css``, §5).
3. ``require_auth`` — парольный гейт до навигации: на неаутентифицированном пути он
   вызывает ``st.stop`` и навигация не строится (§7).
4. ``st.navigation`` с одной секцией из пяти страниц + ``.run()`` (§4, §10).

Страница «Прогнозы» отложена до фазы ML (DESIGN §10.4) и в навигацию не входит.
"""

from pathlib import Path

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from dashboard.auth import get_api_client, require_auth
from dashboard.components.sidebar import render_market_context
from dashboard.pages import monitoring, news, overview, portfolio, stocks

#: Заголовок вкладки браузера (RU-копи — пользовательская строка).
_PAGE_TITLE = "StockLens"

#: Путь к тонкому CSS-слою: parents[2] = services/dashboard (app.py лежит в src/dashboard).
_CSS_PATH = Path(__file__).resolve().parents[2] / "assets" / "dashboard.css"

#: Название единственной секции навигации (RU-копи).
_NAV_SECTION = "Разделы"

#: Бренд-блок сайдбара: идентичность сервиса под навигацией, заполняет пустую нижнюю
#: часть сайдбара (DESIGN §4). RU-копи — пользовательские строки; HTML — доверенная разметка.
_SIDEBAR_BRAND_HTML = (
    '<div class="sl-sidebar-brand">'
    '<span class="sl-sidebar-brand__mark"></span>'
    '<span class="sl-sidebar-brand__name">StockLens</span>'
    '<span class="sl-sidebar-brand__tag">Аналитика рынка MOEX</span>'
    "</div>"
)


def _inject_css() -> None:
    """Подключить тонкий CSS-слой дашборда один раз (DESIGN §5)."""
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _render_sidebar_brand() -> None:
    """Бренд-блок в сайдбаре под навигацией: идентичность сервиса + заполнение пустоты (§4)."""
    with st.sidebar:
        st.markdown(_SIDEBAR_BRAND_HTML, unsafe_allow_html=True)


def _build_navigation() -> StreamlitPage:
    """Собрать навигацию: одна секция из пяти страниц с RU-заголовками и Material-иконками.

    Каждая страница — callable ``render`` под ``st.Page`` (DESIGN §4, §10). Все пять
    callable называются ``render``, поэтому ``url_path`` задаётся явно: иначе Streamlit
    выводит один и тот же pathname из имени функции и падает на не-уникальных путях.
    Первая страница — ``default`` (лендинг навигации). Страница «Прогнозы» отложена до
    фазы ML и здесь отсутствует.
    """
    pages = [
        st.Page(
            overview.render,
            title="Обзор",
            icon=":material/dashboard:",
            url_path="overview",
            default=True,
        ),
        st.Page(stocks.render, title="Акции", icon=":material/trending_up:", url_path="stocks"),
        st.Page(news.render, title="Новости", icon=":material/newspaper:", url_path="news"),
        st.Page(
            portfolio.render,
            title="Портфель",
            icon=":material/account_balance_wallet:",
            url_path="portfolio",
        ),
        st.Page(
            monitoring.render,
            title="Мониторинг",
            icon=":material/monitor_heart:",
            url_path="monitoring",
        ),
    ]
    return st.navigation({_NAV_SECTION: pages})


def main() -> None:
    """Запустить дашборд: конфиг страницы → CSS → гейт → навигация (DESIGN §4, §7)."""
    st.set_page_config(layout="wide", page_title=_PAGE_TITLE)
    _inject_css()
    require_auth()
    navigation = _build_navigation()
    render_market_context(get_api_client())
    _render_sidebar_brand()
    navigation.run()


main()
