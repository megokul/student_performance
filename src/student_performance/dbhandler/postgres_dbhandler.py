import psycopg2
from psycopg2 import sql
import pandas as pd
import yaml

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.entity.config_entity import PostgresDBHandlerConfig


class PostgresDBHandler(DBHandler):
    def __init__(self, postgres_config: PostgresDBHandlerConfig) -> None:
        self.postgres_config = postgres_config
        self._connection: psycopg2.extensions.connection | None = None
        self._cursor: psycopg2.extensions.cursor | None = None

    def _connect(self) -> None:
        if not self._connection or self._connection.closed:
            try:
                self._connection = psycopg2.connect(
                    host=self.postgres_config.host,
                    port=self.postgres_config.port,
                    dbname=self.postgres_config.dbname,
                    user=self.postgres_config.user,
                    password=self.postgres_config.password,
                )
                self._cursor = self._connection.cursor()
            except Exception as e:
                msg = "Failed to establish PostgreSQL connection"
                raise StudentPerformanceError(e, msg) from e

    def close(self) -> None:
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()

    def ping(self) -> None:
        try:
            self._connect()
            self._cursor.execute("SELECT 1;")
            self._cursor.fetchone()
        except Exception as e:
            msg = "PostgreSQL ping failed"
            raise StudentPerformanceError(e, msg) from e

    def load_from_source(self) -> pd.DataFrame:
        try:
            self._connect()
            query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(self.table_name))
            df = pd.read_sql_query(query, self._connection)
            logger.info(f"DataFrame loaded from PostgreSQL table: {self.table_name}")
            return df
        except Exception as e:
            msg = f"Failed to load data from PostgreSQL table: {self.table_name}"
            raise StudentPerformanceError(e, msg) from e

    def create_table_if_not_exists(self, table_name: str, schema: dict) -> None:
        """
        Create a PostgreSQL table if it doesn't exist.

        Args:
            table_name (str): The name of the table to create.
            schema (dict): A dictionary mapping column names to SQL types.

        Example:
            schema = {
                "id": "SERIAL PRIMARY KEY",
                "name": "VARCHAR(100)",
                "score": "FLOAT",
                "exam_date": "DATE"
            }

        Raises:
            StudentPerformanceError: If table creation fails.
        """
        try:
            self._connect()

            # Load schema from YAML file
            with open("config/schema.yaml", "r") as f:
                schema_yaml = yaml.safe_load(f)

            table_schema = schema_yaml["table_schema"][table_name]["columns"]

            # Format columns into SQL syntax
            columns_def = ", ".join(f"{col} {table_schema[col]['type']}" for col in table_schema)

            create_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    {}
                )
            """).format(
                sql.Identifier(table_name),
                sql.SQL(columns_def)
            )

            self._cursor.execute(create_query)
            self._connection.commit()

            logger.info(f"Table '{table_name}' created (if it did not exist).")
        except Exception as e:
            msg = f"Failed to create table: '{table_name}'"
            raise StudentPerformanceError(e, msg) from e
