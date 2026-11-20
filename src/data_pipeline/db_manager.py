"""
PostgreSQL database manager for the Customer Lifetime Value & Churn
Intelligence System.

Uses SQLAlchemy + psycopg2 to manage the ``churn_intelligence`` database.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Make config importable when running from any directory
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import DATABASE_URL, DB_CONFIG  # noqa: E402

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thin wrapper around SQLAlchemy for the churn-intelligence database."""

    def __init__(self, connection_url: str | None = None) -> None:
        """Initialise with a PostgreSQL connection URL.

        Parameters
        ----------
        connection_url : str, optional
            SQLAlchemy connection string.  Defaults to ``DATABASE_URL`` from
            ``config.settings``.
        """
        self.connection_url: str = connection_url or DATABASE_URL
        self.engine: Engine | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_engine(self) -> Engine:
        if self.engine is None:
            self.engine = create_engine(self.connection_url)
        return self.engine

    def _get_server_engine(self) -> Engine:
        """Return an engine connected to the *default* ``postgres`` database
        (used for admin operations such as ``CREATE DATABASE``)."""
        server_url = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
        )
        return create_engine(server_url)

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    def create_database(self) -> None:
        """Create the ``churn_intelligence`` database if it does not exist."""
        db_name = DB_CONFIG["database"]
        engine = self._get_server_engine()
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": db_name},
            )
            if result.fetchone() is None:
                # Use raw SQL – parameterised DDL is not supported
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info("Database '%s' created.", db_name)
            else:
                logger.info("Database '%s' already exists.", db_name)
        engine.dispose()

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_data_to_db(
        self, df: pd.DataFrame, table_name: str = "customers"
    ) -> None:
        """Load a DataFrame into a PostgreSQL table (replace if exists)."""
        engine = self._get_engine()
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info(
            "Loaded %d rows into table '%s'.", len(df), table_name
        )

    def read_data_from_db(self, table_name: str = "customers") -> pd.DataFrame:
        """Read a full table into a DataFrame."""
        engine = self._get_engine()
        df = pd.read_sql_table(table_name, engine)
        logger.info("Read %d rows from table '%s'.", len(df), table_name)
        return df

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute an arbitrary SQL query and return the result as a DataFrame."""
        engine = self._get_engine()
        df = pd.read_sql(text(query), engine)
        return df

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Dispose of the underlying engine connection pool."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed.")
