from box import ConfigBox
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PostgresDBHandlerConfig:
    root_dir: Path
    host: str
    port: int
    dbname: str
    user: str
    password: str
    table_name: str
    input_data_filepath: Path

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.input_data_filepath = Path(self.input_data_filepath)

    def __repr__(self) -> str:
        return (
            "\nPostgres DB Handler Config:\n"
            f"  - Root Dir:         '{self.root_dir.as_posix()}'\n"
            f"  - Host:             {self.host}\n"
            f"  - Port:             {self.port}\n"
            f"  - Database Name:    {self.dbname}\n"
            f"  - User:             {self.user}\n"
            f"  - Password:         {'*' * 8} (hidden)\n"
            f"  - Table:            {self.table_name}\n"
            f"  - Input Filepath:   '{self.input_data_filepath.as_posix()}'\n"
        )
