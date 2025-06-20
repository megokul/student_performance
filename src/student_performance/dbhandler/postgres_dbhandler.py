import psycopg2
from psycopg2 import sql
import pandas as pd

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.entity.config_entity import PostgresDBHandlerConfig
from box import ConfigBox


class PostgresDBHandler(DBHandler):
    def __init__(self, config: PostgresDBHandlerConfig) -> None:
        logger.info("Initializing PostgresDBHandler")
        self.config = config
        self._connection: psycopg2.extensions.connection | None = None
        self._cursor: psycopg2.extensions.cursor | None = None

    def _connect(self) -> None:
        logger.info("Attempting to connect to PostgreSQL")
        if not self._connection or self._connection.closed:
            try:
                self._connection = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    dbname=self.config.dbname,
                    user=self.config.user,
                    password=self.config.password,
                )
                self._cursor = self._connection.cursor()
                logger.info("Successfully connected to PostgreSQL")
            except Exception as e:
                msg = "Failed to establish PostgreSQL connection"
                raise StudentPerformanceError(e, msg) from e

    def close(self) -> None:
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
            logger.info("PostgreSQL connection closed")

    def ping(self) -> None:
        logger.info("Pinging PostgreSQL")
        try:
            self._connect()
            logger.info("Executing ping query")
            self._cursor.execute("SELECT 1;")
            self._cursor.fetchone()
            logger.info("PostgreSQL connection successful (ping passed).")
        except Exception as e:
            msg = "PostgreSQL ping failed"
            raise StudentPerformanceError(e, msg) from e
        logger.info("PostgreSQL ping completed")

    def load_from_source(self) -> pd.DataFrame:
        logger.info(f"Loading data from PostgreSQL table: {self.config.table_name}")
        try:
            self._connect()
            query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(self.config.table_name))
            logger.info(f"Executing query: {query.as_string(self._connection)}")
            df = pd.read_sql_query(query.as_string(self._connection), self._connection)
            logger.info(f"DataFrame loaded from PostgreSQL table: {self.config.table_name}")
            return df
        except Exception as e:
            msg = f"Failed to load data from PostgreSQL table: {self.config.table_name}"
            raise StudentPerformanceError(e, msg) from e

    def get_table_list(self) -> list[str]:
        """
        Get a list of all tables in the PostgreSQL database.

        Returns:
            list[str]: A list of table names.

        Raises:
            StudentPerformanceError: If listing tables fails.
        """
        logger.info("Retrieving list of tables")
        try:
            self._connect()
            query = sql.SQL("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE';
            """)
            logger.info(f"Executing query: {query.as_string(self._connection)}")
            self._cursor.execute(query)
            tables = [table[0] for table in self._cursor.fetchall()]
            logger.info("Successfully retrieved list of tables.")
            return tables
        except Exception as e:
            msg = "Failed to retrieve list of tables"
            raise StudentPerformanceError(e, msg) from e

    def create_table_from_schema(self) -> None:
        """
        Create a PostgreSQL table if it doesn't exist using schema from ConfigBox.

        Args:
            table_name (str): The name of the table to create.
            schema (ConfigBox): Parsed schema.yaml with dot-access support.

        Raises:
            StudentPerformanceError: If table creation fails.
        """
        logger.info(f"Creating table from schema: {self.config.table_name}")
        try:
            self._connect()

            table_name = self.config.table_name

            # Check if table exists
            query = sql.SQL("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = {}
                );
            """).format(sql.Literal(table_name))
            logger.info(f"Executing query: {query.as_string(self._connection)}")
            self._cursor.execute(query)
            table_exists = self._cursor.fetchone()[0]

            if table_exists:
                logger.info(f"Table '{table_name}' already exists.")
                return

            # Access columns via dot-notation
            table_schema = self.config.table_schema[table_name].columns

            column_definitions = []

            for col_name, col_def in table_schema.items():
                col_type = col_def.type
                constraints = col_def.get("constraints", {})

                column_sql = f"{col_name} {col_type}"

                # ENUM-style value check
                if "allowed_values" in constraints:
                    allowed = ", ".join("'{}'".format(val.replace("'", "''")) for val in constraints.allowed_values)
                    column_sql += f" CHECK ({col_name} IN ({allowed}))"

                # Numeric bounds
                if "min" in constraints and "max" in constraints:
                    column_sql += f" CHECK ({col_name} BETWEEN {constraints.min} AND {constraints.max})"
                elif "min" in constraints:
                    column_sql += f" CHECK ({col_name} >= {constraints.min})"
                elif "max" in constraints:
                    column_sql += f" CHECK ({col_name} <= {constraints.max})"

                column_definitions.append(column_sql)

            # Final CREATE query
            create_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    {}
                );
            """).format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(map(sql.SQL, column_definitions))
            )
            logger.info(f"Executing query: {create_query.as_string(self._connection)}")
            self._cursor.execute(create_query)
            self._connection.commit()
            logger.info(f"Table '{table_name}' created.")

        except Exception as e:
            msg = f"Failed to create table: '{table_name}'"
            raise StudentPerformanceError(e, msg) from e
        logger.info(f"Finished creating table from schema: {self.config.table_name}")

    def insert_data_from_csv(self) -> None:
        """
        Insert data from the configured CSV file into the PostgreSQL table.

        Raises:
            StudentPerformanceError: If data insertion fails.
        """
        logger.info(f"Inserting data from CSV into table: {self.config.table_name}")
        try:
            self._connect()
            
            # Read the CSV file into a Pandas DataFrame
            csv_filepath = self.config.input_data_filepath
            logger.info(f"Reading CSV file: {csv_filepath}")
            df = pd.read_csv(csv_filepath)
            
            # Get the table name
            table_name = self.config.table_name
            
            # Define the SQL INSERT query
            columns = ', '.join(df.columns)
            values = ', '.join(['%s'] * len(df.columns))
            insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table_name),
                sql.SQL(columns),
                sql.SQL(values)
            )
            
            # Execute the INSERT query for each row in the DataFrame
            logger.info("Inserting data into table")
            for _, row in df.iterrows():
                self._cursor.execute(insert_query, row.tolist())
            
            # Commit the changes to the database
            self._connection.commit()
            
        except Exception as e:
            msg = f"Failed to insert data from CSV into table: {self.config.table_name}"
            raise StudentPerformanceError(e, msg) from e
        logger.info(f"Successfully inserted data from CSV into table: {self.config.table_name}")
        logger.info(f"Finished inserting data from CSV into table: {self.config.table_name}")

    def read_data_to_df(self) -> pd.DataFrame:
        """
        Reads data from the PostgreSQL table into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the data from the table.

        Raises:
            StudentPerformanceError: If reading data fails.
        """
        logger.info(f"Reading data from PostgreSQL table: {self.config.table_name}")
        try:
            self._connect()
            query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(self.config.table_name))
            logger.info(f"Executing query: {query.as_string(self._connection)}")
            df = pd.read_sql_query(query.as_string(self._connection), self._connection)
            logger.info(f"Successfully read data from PostgreSQL table: {self.config.table_name}")
            logger.info(f"Finished reading data from PostgreSQL table: {self.config.table_name}")
            return df
        except Exception as e:
            msg = f"Failed to read data from PostgreSQL table: {self.config.table_name}"
            raise StudentPerformanceError(e, msg) from e
