"""Классификация тональности новостных текстов через ONNX-модель.

Входной текст: title + ". " + summary (если есть), усечение до 512 токенов.
Обоснование: rubert-tiny-sentiment-balanced имеет max_position_embeddings=512;
модель обучена на коротких текстах, но summary добавляет контекст без риска
выхода за лимит при усечении токенизатором.

Зависимости runtime: onnxruntime, tokenizers (без torch и transformers).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import onnxruntime as ort
from stocklens_core.enums import SentimentLabel
from tokenizers import Tokenizer

_MAX_TOKENS = 512

_LABEL_MAP: dict[int, SentimentLabel] = {
    0: SentimentLabel.NEGATIVE,
    1: SentimentLabel.NEUTRAL,
    2: SentimentLabel.POSITIVE,
}


@dataclass(frozen=True)
class SentimentResult:
    """Результат классификации тональности."""

    label: SentimentLabel
    score: float


@runtime_checkable
class SentimentScorer(Protocol):
    """Протокол классификатора тональности текста."""

    @property
    def model_version(self) -> str:
        """Строковый идентификатор модели (HuggingFace id или путь)."""
        ...

    def score(self, text: str) -> SentimentResult:
        """Классифицировать текст.

        Args:
            text: Произвольный русскоязычный текст.

        Returns:
            SentimentResult с меткой и уверенностью (0..1).
        """
        ...


class OnnxSentimentScorer:
    """Классификатор тональности на базе ONNX-модели rubert-tiny-sentiment-balanced.

    Загружает tokenizer.json и model.onnx из указанной директории.
    Вычисляет softmax по логитам, возвращает метку с максимальной вероятностью.

    Args:
        model_dir: Директория с файлами tokenizer.json и model.onnx.
        model_id: Строковый идентификатор модели для model_version.
    """

    def __init__(self, model_dir: Path, model_id: str) -> None:
        tokenizer_path = model_dir / "tokenizer.json"
        model_path = model_dir / "model.onnx"

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=_MAX_TOKENS)
        self._tokenizer.enable_padding(length=_MAX_TOKENS)

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # ERROR only — подавляет INFO-шум ONNX Runtime
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._model_id = model_id
        self._label_map = _load_label_map(model_dir)

    @property
    def model_version(self) -> str:
        return self._model_id

    def score(self, text: str) -> SentimentResult:
        """Классифицировать текст через ONNX-инференс.

        Args:
            text: Русскоязычный текст для классификации.

        Returns:
            SentimentResult с тональностью и уверенностью модели.
        """
        encoding = self._tokenizer.encode(text)

        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoding.type_ids], dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        logits: list[float] = outputs[0][0].tolist()
        probabilities = _softmax(logits)
        best_idx = int(np.argmax(probabilities))

        label = self._label_map.get(best_idx, SentimentLabel.NEUTRAL)
        return SentimentResult(label=label, score=float(probabilities[best_idx]))


def build_scoring_text(title: str, summary: str | None) -> str:
    """Собрать текст для скоринга: title + summary через точку.

    Args:
        title: Заголовок статьи.
        summary: Аннотация или None.

    Returns:
        Объединённый текст для передачи в scorer.score().
    """
    if summary:
        return f"{title}. {summary}"
    return title


def _softmax(logits: list[float]) -> list[float]:
    """Вычислить softmax над списком логитов.

    Вычитаем max для численной стабильности (стандартный трюк).
    """
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _load_label_map(model_dir: Path) -> dict[int, SentimentLabel]:
    """Загрузить id2label из config.json модели.

    Возвращает захардкоженный _LABEL_MAP как fallback если config.json отсутствует,
    чтобы инференс работал даже при минимальном наборе файлов экспорта.

    Args:
        model_dir: Директория с экспортированной моделью.

    Returns:
        Словарь {class_id: SentimentLabel}.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return _LABEL_MAP

    with config_path.open(encoding="utf-8") as fh:
        config: dict[str, object] = json.load(fh)

    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        return _LABEL_MAP

    result: dict[int, SentimentLabel] = {}
    label_str_map = {label.value: label for label in SentimentLabel}

    for key, value in id2label.items():
        try:
            idx = int(key)
            label = label_str_map.get(str(value).lower())
            if label is not None:
                result[idx] = label
        except ValueError:
            continue

    return result if result else _LABEL_MAP
