"""Unit-тесты для PageParams."""

from api.core.pagination import PageParams


def test_page_params_clamps_limit_above_max() -> None:
    params = PageParams(limit=9999, offset=0)
    assert params.limit == 200


def test_page_params_clamps_limit_below_min() -> None:
    params = PageParams(limit=0, offset=0)
    assert params.limit == 1


def test_page_params_clamps_negative_offset() -> None:
    params = PageParams(limit=10, offset=-5)
    assert params.offset == 0


def test_page_params_default_values() -> None:
    params = PageParams()
    assert params.limit == 50
    assert params.offset == 0
