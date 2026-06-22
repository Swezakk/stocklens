"""Страница «Новости» дашборда (DESIGN.md §10.3, §9, §2.2).

Фильтры (тикер / тональность / период) → лента статей с SentimentChip и тикерами +
агрегаты тональности (динамика тона, частотные слова) над period-bounded корпусом с
ВИДИМЫМ реальным размером выборки (DESIGN §9: честная статистика, без молчаливого
усечения).

render() — тонкая оркестрация: всё нетривиальное преобразование данных вынесено в
чистые типизированные хелперы (`_resolve_date_range`, `_resolve_sentiment_filter`,
`_filter_by_sentiments`, `_format_published_at`, `_news_row_markdown`, `_corpus_caption`),
которые покрыты unit-тестами. Каждый сетевой вызов проходит три ветки фидбэка (успех /
ошибка сервера / сеть недоступна) — пустых экранов нет.
"""

import html
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from stocklens_core.enums import SentimentLabel

from dashboard import theme
from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import NewsOut
from dashboard.api_client.errors import ApiError
from dashboard.api_client.fetch import (
    fetch_all_securities,
    fetch_news,
    fetch_news_corpus,
)
from dashboard.auth import get_api_client
from dashboard.components import filters
from dashboard.components.charts import (
    build_sentiment_trend_chart,
    build_word_frequency_chart,
    render_chart,
)
from dashboard.components.feedback import render_empty, render_error
from dashboard.components.sentiment import render_sentiment_chip
from dashboard.components.transforms import group_sentiment_by_date, word_frequencies
from dashboard.settings import get_settings

#: Заголовок страницы (RU-копи — пользовательская строка).
_PAGE_TITLE = "Новости"

#: Часовой пояс отображения меток времени (CLAUDE.md: «отображение — Europe/Moscow»).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")

#: Формат отображения даты-времени публикации (московское локальное время).
_PUBLISHED_AT_FORMAT = "%d.%m.%Y %H:%M"

#: Размер страницы ленты новостей (одна страница fetch_news; DESIGN §9: _MAX_LIMIT = 200).
_FEED_PAGE_SIZE = 50

#: Сколько частотных слов показывать в баре (DESIGN §10.3).
_WORD_FREQUENCY_TOP_N = 15

#: Минимум точек данных, ниже которого график агрегата вырождается (одна точка в гигантской
#: рамке с абсурдной осью). Меньше порога → текстовая заметка вместо графика (DESIGN §4, §10).
_MIN_AGGREGATE_POINTS = 3

#: Ключи виджетов фильтров (разводят виджеты на странице без коллизий).
_TICKER_KEY = "news_ticker"
_SENTIMENT_KEY = "news_sentiment"
_DATE_RANGE_KEY = "news_date_range"
_FEED_OFFSET_KEY = "news_feed_offset"
_FEED_FILTER_KEY = "news_feed_filter_signature"

#: Заголовки секций (RU-копи).
_SECTION_FEED = "Лента"
_SECTION_TREND = "Динамика тона"
_SECTION_WORDS = "Частотные слова"

#: Подписи пустых результатов (RU-копи; успех без данных — не сбой).
_EMPTY_FEED = "Новостей по выбранным фильтрам нет."
_EMPTY_TREND = "Недостаточно данных для динамики тона за период."
_EMPTY_WORDS = "Недостаточно данных для частотных слов за период."

#: Подписи кнопок пагинации ленты.
_PREV_LABEL = "Назад"
_NEXT_LABEL = "Вперёд"


def render() -> None:
    """Отрисовать страницу «Новости» (DESIGN §10.3): фильтры → лента → агрегаты.

    Тонкая оркестрация: фетчит справочник тикеров, рисует фильтры, затем независимо
    ленту (одна страница) и агрегаты (корпус по периоду). Каждый блок — со своей
    тройной обработкой сетевого вызова, поэтому сбой одного не гасит остальные.
    """
    st.title(_PAGE_TITLE)

    client = get_api_client()

    tickers = _load_tickers(client)
    selected_ticker, selected_sentiments, date_range = _render_filters(tickers)
    date_from, date_to = _resolve_date_range(date_range)

    _render_feed(client, selected_ticker, selected_sentiments, date_from, date_to)
    _render_aggregates(client, selected_ticker, selected_sentiments, date_from, date_to)


def _load_tickers(client: ApiClient) -> list[str]:
    """Загрузить список тикеров для фильтра; сетевой сбой → фидбэк, пустой список."""
    try:
        securities = fetch_all_securities(client, is_active=True)
    except ApiError as exc:
        render_error(exc.user_message)
        return []
    return [security.ticker for security in securities]


def _render_filters(
    tickers: Sequence[str],
) -> tuple[str | None, list[SentimentLabel], object]:
    """Отрисовать фильтры (тикер / тональность / период) в три колонки.

    Возвращает выбранный тикер, список меток тональности и сырое значение date_input
    (нормализуется чистым `_resolve_date_range`, чтобы орхестрация осталась тонкой).
    """
    settings = get_settings()
    default_to = datetime.now(_MOSCOW_TZ).date()
    default_from = default_to - timedelta(days=settings.news_corpus_period_days)

    ticker_col, sentiment_col, date_col = st.columns(3)
    with ticker_col:
        selected_ticker = filters.select_ticker(tickers, key=_TICKER_KEY)
    with sentiment_col:
        selected_sentiments = filters.select_sentiments(key=_SENTIMENT_KEY)
    with date_col:
        date_range = st.date_input(
            "Период",
            value=(default_from, default_to),
            max_value=default_to,
            key=_DATE_RANGE_KEY,
        )
    return selected_ticker, selected_sentiments, date_range


def _render_feed(
    client: ApiClient,
    ticker: str | None,
    sentiments: list[SentimentLabel],
    date_from: date | None,
    date_to: date | None,
) -> None:
    """Отрисовать ленту новостей (одна страница fetch_news) с пагинацией.

    Серверный фильтр тональности — одиночный (контракт API), поэтому при выборе двух+
    меток сервер не фильтруется (`_resolve_sentiment_filter` вернёт None), а сужение
    идёт чистым `_filter_by_sentiments` над уже скоренными статьями (presentation, §3).

    Offset сбрасывается в 0 при смене любого фильтра (тикер/тональность/период), иначе
    переход на узкий фильтр со старым offset отдал бы пустую страницу без выхода назад.
    Пагинация рисуется ВСЕГДА (даже над пустой страницей), чтобы offset>0 можно было
    откатить кнопкой «Назад» — пустой результат не должен «запирать» пользователя.
    """
    st.subheader(_SECTION_FEED)
    _reset_feed_offset_on_filter_change(ticker, sentiments, date_from, date_to)
    offset = _feed_offset()
    server_sentiment = _resolve_sentiment_filter(sentiments)
    try:
        page = fetch_news(
            client,
            ticker=ticker,
            sentiment=server_sentiment,
            date_from=_iso_or_none(date_from),
            date_to=_iso_or_none(date_to),
            limit=_FEED_PAGE_SIZE,
            offset=offset,
        )
    except ApiError as exc:
        render_error(exc.user_message)
        return

    articles = _filter_by_sentiments(page.items, sentiments)
    if not articles:
        render_empty(_EMPTY_FEED)
    else:
        for article in articles:
            st.markdown(_news_row_markdown(article), unsafe_allow_html=True)
            st.divider()

    _render_feed_pagination(offset, len(articles), page.total)


def _filter_signature(
    ticker: str | None,
    sentiments: Sequence[SentimentLabel],
    date_from: date | None,
    date_to: date | None,
) -> tuple[str | None, tuple[str, ...], str | None, str | None]:
    """Хэшируемая подпись активных фильтров ленты для детекта их изменения между rerun."""
    labels = tuple(sorted(label.value for label in sentiments))
    return ticker, labels, _iso_or_none(date_from), _iso_or_none(date_to)


def _reset_feed_offset_on_filter_change(
    ticker: str | None,
    sentiments: Sequence[SentimentLabel],
    date_from: date | None,
    date_to: date | None,
) -> None:
    """Сбросить offset ленты в 0, если набор фильтров изменился с прошлого rerun.

    Подпись фильтров хранится в session_state; при расхождении offset обнуляется, чтобы
    смена фильтра всегда показывала первую страницу нового результата (а не пустую N-ю).
    """
    signature = _filter_signature(ticker, sentiments, date_from, date_to)
    if st.session_state.get(_FEED_FILTER_KEY) != signature:
        st.session_state[_FEED_FILTER_KEY] = signature
        st.session_state[_FEED_OFFSET_KEY] = 0


def _feed_caption(offset: int, shown: int, total: int) -> str:
    """Честная подпись объёма ленты «Показаны N–M из total» (DESIGN §9: видимый объём).

    ``shown`` — число фактически отрисованных строк (после клиентского сужения по 2+ меткам
    тональности), ``total`` — размер серверной выборки. Конец диапазона считается от ``shown``,
    а не от размера страницы: при клиентском сужении видно меньше 50 строк, и подпись это
    отражает, не завышая до полного ``page.total``. Пустая страница → «Показано 0 из total».
    """
    if shown == 0:
        return f"Показано 0 из {total}"
    shown_from = offset + 1
    shown_to = offset + shown
    return f"Показаны {shown_from}–{shown_to} из {total}"


def _render_feed_pagination(offset: int, shown: int, total: int) -> None:
    """Кнопки пагинации ленты + честная подпись объёма (видимое число строк, §9)."""
    st.caption(_feed_caption(offset, shown, total))

    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button(_PREV_LABEL, disabled=offset <= 0, key="news_feed_prev"):
            st.session_state[_FEED_OFFSET_KEY] = max(0, offset - _FEED_PAGE_SIZE)
            st.rerun()
    with next_col:
        has_next = offset + _FEED_PAGE_SIZE < total
        if st.button(_NEXT_LABEL, disabled=not has_next, key="news_feed_next"):
            st.session_state[_FEED_OFFSET_KEY] = offset + _FEED_PAGE_SIZE
            st.rerun()


def _render_aggregates(
    client: ApiClient,
    ticker: str | None,
    sentiments: list[SentimentLabel],
    date_from: date | None,
    date_to: date | None,
) -> None:
    """Отрисовать агрегаты тональности (динамика тона, частотные слова) над корпусом.

    Корпус берётся period-bounded циклом (DESIGN §9) с честным размером выборки в
    подписи. Серверный фильтр тональности одиночный — при двух+ метках сужаем корпус
    чистым `_filter_by_sentiments` после фетча.
    """
    server_sentiment = _resolve_sentiment_filter(sentiments)
    try:
        corpus, truncated, total = fetch_news_corpus(
            client,
            ticker=ticker,
            sentiment=server_sentiment,
            date_from=_iso_or_none(date_from),
            date_to=_iso_or_none(date_to),
        )
    except ApiError as exc:
        render_error(exc.user_message)
        return

    articles = _filter_by_sentiments(corpus, sentiments)
    caption = _corpus_caption(total, len(articles), truncated, date_from, date_to)

    _render_trend(articles, caption)
    _render_word_frequencies(articles, caption)


def _render_trend(articles: list[NewsOut], caption: str) -> None:
    """Отрисовать график динамики среднего тона по дням с честной подписью выборки."""
    st.subheader(_SECTION_TREND)
    st.caption(caption)
    series = group_sentiment_by_date(articles)
    if len(series) < _MIN_AGGREGATE_POINTS:
        render_empty(_EMPTY_TREND)
        return
    render_chart(build_sentiment_trend_chart(series))


def _render_word_frequencies(articles: list[NewsOut], caption: str) -> None:
    """Отрисовать бар топ-частотных слов заголовков с честной подписью выборки."""
    st.subheader(_SECTION_WORDS)
    st.caption(caption)
    frequencies = word_frequencies(
        [article.title for article in articles],
        top_n=_WORD_FREQUENCY_TOP_N,
    )
    if len(frequencies) < _MIN_AGGREGATE_POINTS:
        render_empty(_EMPTY_WORDS)
        return
    render_chart(build_word_frequency_chart(frequencies))


def _feed_offset() -> int:
    """Текущий offset ленты из session_state (по умолчанию 0)."""
    raw = st.session_state.get(_FEED_OFFSET_KEY, 0)
    return raw if isinstance(raw, int) and raw >= 0 else 0


def _resolve_date_range(value: object) -> tuple[date | None, date | None]:
    """Нормализовать значение st.date_input в пару `(date_from, date_to)`.

    st.date_input с tuple-`value` возвращает кортеж дат, но при незавершённом выборе
    диапазона — кортеж из одной даты; одиночный режим вернул бы голую date. Все формы
    приводятся к паре границ: неполный выбор → вторая граница None (фильтр открыт справа).
    """
    if isinstance(value, date):
        return value, value
    if isinstance(value, (tuple, list)):
        dates = [item for item in value if isinstance(item, date)]
        if not dates:
            return None, None
        if len(dates) == 1:
            return dates[0], None
        return dates[0], dates[1]
    return None, None


def _resolve_sentiment_filter(selected: Sequence[SentimentLabel]) -> SentimentLabel | None:
    """Свести мультивыбор тональности к одиночному серверному фильтру (контракт API).

    API принимает одну метку (`SentimentLabel | None`). Ровно одна выбранная метка →
    серверный фильтр; ноль или две+ → None (две+ сужаются клиентски в presentation-слое,
    чтобы не делать N серверных запросов и не терять честность выборки).
    """
    if len(selected) == 1:
        return selected[0]
    return None


def _filter_by_sentiments(
    articles: Sequence[NewsOut],
    selected: Sequence[SentimentLabel],
) -> list[NewsOut]:
    """Сузить статьи до выбранных меток тональности (presentation-фильтр, §3 «no compute»).

    Применяется только при двух+ выбранных метках (одну фильтрует сервер, ноль — без
    фильтра). Статьи без тональности при активном фильтре исключаются. Пустой `selected`
    или ≤1 метка → вход возвращается как список без изменений (сервер уже отфильтровал).
    """
    chosen = set(selected)
    if len(chosen) <= 1:
        return list(articles)
    return [
        article
        for article in articles
        if article.sentiment is not None and article.sentiment.label in chosen
    ]


def _format_published_at(published_at: datetime) -> str:
    """Отформатировать UTC-метку публикации в московское локальное время (DESIGN, CLAUDE.md)."""
    return published_at.astimezone(_MOSCOW_TZ).strftime(_PUBLISHED_AT_FORMAT)


def _news_row_markdown(article: NewsOut) -> str:
    """Собрать markdown-строку ленты: заголовок-ссылка, источник, время МСК, чип, тикеры.

    Заголовок — ссылка на url статьи; тональность — SentimentChip (текст+цвет, a11y);
    тикеры — инлайн-чипы. Строка рендерится через ``unsafe_allow_html=True``, а title /
    url / source приходят из внешних RSS-лент (НЕ доверенный контракт API), поэтому
    экранируются ``html.escape`` против stored-HTML-инъекции — как в kpi.py и sentiment.py.
    Тикеры — контролируемый MOEX-словарь, оборачиваются в inline-code и не экранируются.
    """
    chip = render_sentiment_chip(article.sentiment.label) if article.sentiment else ""
    tickers = " ".join(f"`{ticker}`" for ticker in article.tickers)
    published = _format_published_at(article.published_at)
    title = html.escape(article.title)
    url = html.escape(article.url, quote=True)
    meta = f"{html.escape(article.source)} · {published}"
    return (
        f"**[{title}]({url})**  {chip}\n\n"
        f'<span style="color:{theme.MUTED_TEXT}">{meta}</span>  {tickers}'
    )


def _corpus_caption(
    total: int,
    shown: int,
    truncated: bool,
    date_from: date | None,
    date_to: date | None,
) -> str:
    """Собрать честную подпись агрегатов: реальный размер выборки + период (DESIGN §9).

    Всегда показывает реальный `total` по версии API и фактически использованное число
    статей (`shown`). Если корпус усечён потолком — это сказано явно, без молчаливого
    усечения. Период включается, если хотя бы одна граница задана.
    """
    period = _period_text(date_from, date_to)
    base = f"Выборка: {shown} из {total} статей{period}."
    if truncated:
        return f"{base} Корпус усечён до потолка — динамика по части периода."
    return base


def _period_text(date_from: date | None, date_to: date | None) -> str:
    """Текст периода для подписи: « за DD.MM.YYYY–DD.MM.YYYY» либо пустая строка."""
    if date_from is None and date_to is None:
        return ""
    start = date_from.strftime("%d.%m.%Y") if date_from is not None else "…"
    end = date_to.strftime("%d.%m.%Y") if date_to is not None else "…"
    return f" за {start}–{end}"


def _iso_or_none(value: date | None) -> str | None:
    """ISO-строка даты для query-параметров fetch, либо None (фильтр отсутствует)."""
    return value.isoformat() if value is not None else None
