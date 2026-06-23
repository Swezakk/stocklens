"""FastAPI-зависимость для доступа к загруженным ML-моделям из app.state (ml-spec §8.1)."""

from typing import Annotated

from fastapi import Depends, Request

from api.ml.bundle import ModelBundle


def get_ml_bundle(request: Request) -> ModelBundle:
    """Получить контейнер загруженных ML-моделей из app.state.ml.

    Lifespan всегда выставляет app.state.ml (§8.1); пустой bundle по умолчанию — защита от
    обращения до инициализации (readiness тогда репортит модель недоступной, а не 500).
    """
    bundle: ModelBundle | None = getattr(request.app.state, "ml", None)
    return bundle if bundle is not None else ModelBundle()


MlBundleDep = Annotated[ModelBundle, Depends(get_ml_bundle)]
