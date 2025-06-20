import logging
import sys
from io import BytesIO
from pathlib import Path

import boto3
import yaml

from src.student_performance.constants.constants import (
    CONFIG_ROOT,
    CONFIG_FILENAME,
    LOGS_ROOT,
)
from src.student_performance.utils.timestamp import get_utc_timestamp


class S3LogHandler(logging.Handler):
    """
    A logging.Handler that buffers all log lines in memory
    and, on each emit, PUTs the full buffer to S3 so that
    the object is always up-to-date. No local file is ever written.
    """
    def __init__(self, bucket: str, key: str, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.bucket = bucket
        self.key = key
        self.buffer = BytesIO()
        self.s3 = boto3.client("s3")
        self.setFormatter(logging.Formatter(
            "[%(asctime)s] - %(levelname)s - %(module)s - %(message)s"
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            self.buffer.write(line.encode("utf-8"))
            self.buffer.seek(0)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=self.buffer.getvalue()
            )
            self.buffer.seek(0, 2)
        except Exception:
            self.handleError(record)


def setup_logger(name: str = "app_logger", level: int = logging.DEBUG) -> logging.Logger:
    """
    - If data_backup.local_enabled=False and data_backup.s3_enabled=True
      → attach only S3LogHandler (no local file).
    - Otherwise → write locally under LOGS_ROOT/<ts>/<ts>.log.
    Always leaves a console handler for stdout.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    timestamp = get_utc_timestamp()

    # load config.yaml directly
    config_root = Path(CONFIG_ROOT)
    config_filepath = config_root / CONFIG_FILENAME
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_backup = config.get("data_backup", {})
    local_enabled = bool(data_backup.get("local_enabled"))
    s3_enabled = bool(data_backup.get("s3_enabled"))
    log_prefix = data_backup.get("s3_log_prefix", "").rstrip("/")

    s3_handler_cfg = config.get("s3_handler", {})
    bucket = s3_handler_cfg.get("final_model_s3_bucket")

    log_key = f"{LOGS_ROOT}/{timestamp}/{timestamp}.log"
    full_s3_key = f"{log_prefix}/{log_key}" if log_prefix else log_key

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # console handler
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
               for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] - %(levelname)s - %(module)s - %(message)s"
        ))
        logger.addHandler(console_handler)

    # only S3
    if not local_enabled and s3_enabled and bucket:
        if not any(isinstance(h, S3LogHandler) for h in logger.handlers):
            s3_handler = S3LogHandler(bucket, full_s3_key, level=level)
            logger.addHandler(s3_handler)

    # local file + console
    else:
        log_dir = Path(LOGS_ROOT) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filepath = log_dir / f"{timestamp}.log"

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_filepath)
                   for h in logger.handlers):
            file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] - %(levelname)s - %(module)s - %(message)s"
            ))
            logger.addHandler(file_handler)

    return logger
