"""add watchlist

Revision ID: f6be1bcedab1
Revises: 9ed4c9dd4b75
Create Date: 2026-06-11 16:22:06.234239

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6be1bcedab1"
down_revision: str | Sequence[str] | None = "9ed4c9dd4b75"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "watchlist",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_watchlist")),
        sa.UniqueConstraint("ticker", name="uq_watchlist_ticker"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("watchlist")
