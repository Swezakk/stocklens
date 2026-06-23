"""SQL-репозиторий рыночных рядов для сборки фич волатильности (ml-spec §8.5).

Читает свечи/дивиденды/сплиты бумаги и отдаёт DataFrame'ы в форме, ожидаемой сборщиком фич
``stocklens_ml.features.assemble`` (тот же расчёт, что при обучении — без train/serve skew).
Weekend-сессии не фильтруются здесь: их исключает сам сборщик (ему нужен флаг).
"""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Candle, Dividend, Split


class SqlVolatilityFeatureRepository:
    """Чтение candles/dividends/splits бумаги как pandas-DataFrame через AsyncSession."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def load_candles(self, security_id: int) -> pd.DataFrame:
        """Все свечи бумаги (по возрастанию даты) с полным OHLC и флагом weekend-сессии."""
        result = await self._session.execute(
            select(Candle).where(Candle.security_id == security_id).order_by(Candle.trade_date)
        )
        rows = list(result.scalars().all())
        return pd.DataFrame(
            {
                "trade_date": [c.trade_date for c in rows],
                "open": [float(c.open) for c in rows],
                "high": [float(c.high) for c in rows],
                "low": [float(c.low) for c in rows],
                "close": [float(c.close) for c in rows],
                "volume": [c.volume for c in rows],
                "is_weekend_session": [c.is_weekend_session for c in rows],
            }
        )

    async def load_dividends(self, security_id: int) -> pd.DataFrame:
        """Дивиденды бумаги (по возрастанию ex_date)."""
        result = await self._session.execute(
            select(Dividend).where(Dividend.security_id == security_id).order_by(Dividend.ex_date)
        )
        rows = list(result.scalars().all())
        return pd.DataFrame(
            {
                "ex_date": [d.ex_date for d in rows],
                "value": [float(d.value) for d in rows],
                "currency": [d.currency for d in rows],
            }
        )

    async def load_splits(self, security_id: int) -> pd.DataFrame:
        """Сплиты бумаги (по возрастанию split_date)."""
        result = await self._session.execute(
            select(Split).where(Split.security_id == security_id).order_by(Split.split_date)
        )
        rows = list(result.scalars().all())
        return pd.DataFrame(
            {
                "split_date": [s.split_date for s in rows],
                "before": [s.before for s in rows],
                "after": [s.after for s in rows],
            }
        )
