"""Тесты классификатора тональности.

Unit-тесты: FakeScorer (без модели).
Integration-тест: OnnxSentimentScorer с реальной ONNX-моделью.
Модель экспортируется один раз в session-scoped фикстуру и кэшируется
в .pytest-onnx-cache/ — при первом запуске требует сеть.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from ingestor.sentiment import (
    OnnxSentimentScorer,
    SentimentResult,
    SentimentScorer,
    _softmax,
    build_scoring_text,
)
from stocklens_core.enums import SentimentLabel

pytestmark = pytest.mark.integration

_ONNX_CACHE = Path(__file__).parent / ".pytest-onnx-cache"
_MODEL_ID = "cointegrated/rubert-tiny-sentiment-balanced"


@pytest.fixture(scope="session")
def onnx_model_dir() -> Path:
    """Экспортировать ONNX-модель в кэш-директорию (один раз на сессию).

    Если модель уже экспортирована — повторный экспорт не выполняется.
    Требует сеть при первом запуске.
    """
    model_dir = _ONNX_CACHE / _MODEL_ID.replace("/", "--")

    if not (model_dir / "model.onnx").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "optimum.exporters.onnx",
                "--model",
                _MODEL_ID,
                "--task",
                "text-classification",
                str(model_dir),
            ],
            check=True,
        )

    return model_dir


class _FakeScorer:
    @property
    def model_version(self) -> str:
        return "test-fake"

    def score(self, text: str) -> SentimentResult:
        return SentimentResult(label=SentimentLabel.NEUTRAL, score=1.0)


class TestSentimentScorerProtocol:
    def test_fake_scorer_satisfies_protocol(self) -> None:
        scorer: SentimentScorer = _FakeScorer()
        result = scorer.score("тест")
        assert isinstance(result.label, SentimentLabel)
        assert 0.0 <= result.score <= 1.0

    def test_build_scoring_text_with_summary(self) -> None:
        text = build_scoring_text("Заголовок", "Описание")
        assert text == "Заголовок. Описание"

    def test_build_scoring_text_without_summary(self) -> None:
        text = build_scoring_text("Заголовок", None)
        assert text == "Заголовок"

    def test_build_scoring_text_empty_summary(self) -> None:
        text = build_scoring_text("Заголовок", "")
        assert text == "Заголовок"

    def test_softmax_sums_to_one(self) -> None:
        logits = [1.0, 2.0, 0.5]
        probs = _softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-6


class TestOnnxSentimentScorer:
    def test_positive_news_classified(self, onnx_model_dir: Path) -> None:
        scorer = OnnxSentimentScorer(model_dir=onnx_model_dir, model_id=_MODEL_ID)
        result = scorer.score("Сбербанк показал рекордную прибыль и повысил дивиденды")
        assert result.label in SentimentLabel
        assert 0.0 < result.score <= 1.0

    def test_negative_news_classified(self, onnx_model_dir: Path) -> None:
        scorer = OnnxSentimentScorer(model_dir=onnx_model_dir, model_id=_MODEL_ID)
        result = scorer.score("Компания объявила о банкротстве, убытки многомиллиардные")
        assert result.label in SentimentLabel
        assert result.score > 0.5

    def test_model_version_matches_id(self, onnx_model_dir: Path) -> None:
        scorer = OnnxSentimentScorer(model_dir=onnx_model_dir, model_id=_MODEL_ID)
        assert scorer.model_version == _MODEL_ID
