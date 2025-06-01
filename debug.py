from src.student_performance.config.configuration import ConfigurationManager
from src.student_performance.dbhandler.postgres_dbhandler import PostgresDBHandler
from dotenv import load_dotenv
load_dotenv()

cmg = ConfigurationManager()
postgres_handler_config = cmg.get_postgres_handler_config()

print(postgres_handler_config)


postgres_dbhandler = PostgresDBHandler(postgres_handler_config)

with postgres_dbhandler:
    postgres_dbhandler.ping()
