"""FSM-состояния мастера /subscribe (aiogram 3.x StatesGroup).

Каждое состояние — конкретный шаг мастера подписки на алерт:
choosing_kind → choosing_ticker → entering_ticker (ручной ввод) | entering_level → (создание).
"""

from aiogram.fsm.state import State, StatesGroup


class SubscribeWizard(StatesGroup):
    """Состояния мастера создания подписки на алерт."""

    choosing_kind = State()
    choosing_ticker = State()
    entering_ticker = State()
    entering_level = State()
