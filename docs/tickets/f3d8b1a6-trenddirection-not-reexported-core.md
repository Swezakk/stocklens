---
id: f3d8b1a6
title: "TrendDirection не реэкспортируется из stocklens_core (остальные 5 StrEnum — да)"
status: open
priority: low
component: packages/stocklens-core
discovered: 2026-06-26
discovered-from: [docs-per-service-readme]
tags: ["api-surface", "enums", "consistency"]
---

# f3d8b1a6: TrendDirection отсутствует в публичном экспорте stocklens_core

## Что обнаружено

`TrendDirection` определён в `packages/stocklens-core/src/stocklens_core/enums.py`, но **не
добавлен в `stocklens_core/__init__.py`** (`__all__` / реэкспорт), тогда как остальные пять
доменных StrEnum (`CollectorRunStatus`, `SentimentLabel`, `PredictionKind`, `Currency`,
`AlertKind`) реэкспортируются. Потребитель `from stocklens_core import TrendDirection` падает —
приходится импортировать `from stocklens_core.enums import TrendDirection`, асимметрично
остальным enum'ам.

## Почему это проблема

Непоследовательность публичного API пакета: один из доменных enumّов доступен по другому
пути, чем братья. CLAUDE.md требует использовать `StrEnum` из `stocklens-core` как единый
источник типов — неполный реэкспорт нарушает ожидание «все доменные enum'ы — из корня пакета».
Низкий приоритет: функционально обходится прямым импортом из `.enums`.

## Что нужно сделать

Добавить `TrendDirection` в реэкспорт `stocklens_core/__init__.py` (и в `__all__`, если он
есть) рядом с остальными StrEnum. Регрессионный тест: `from stocklens_core import
TrendDirection` импортируется (зеркаля существующую проверку реэкспорта прочих enum'ов, если
такая есть).

## Sources

- `packages/stocklens-core/src/stocklens_core/enums.py` — `TrendDirection` определён.
- `packages/stocklens-core/src/stocklens_core/__init__.py` — реэкспорт 5 из 6 StrEnum.
- Обнаружено: написание per-service README (эта сессия, документация).
