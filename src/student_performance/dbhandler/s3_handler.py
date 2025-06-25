from pathlib import Path
import os
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from io import StringIO
from io import BytesIO
import joblib
import yaml

from src.student_performance.entity.config_entity import S3HandlerConfig
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.dbhandler.base_handler import DBHandler


class S3Handler(DBHandler):
    """
    AWS S3 Handler for file, directory, and DataFrame (CSV) operations.
    """

    def __init__(self, config: S3HandlerConfig) -> None:
        try:
            self.config = config
            self._client = boto3.client("s3", region_name=self.config.aws_region)
            logger.info(
                "S3Handler initialized for bucket '%s' in region '%s'",
                self.config.bucket_name,
                self.config.aws_region,
            )
        except Exception as e:
            logger.exception("Failed to initialize S3 client.")
            raise StudentPerformanceError(e, logger) from e
    def __enter__(self):
        logger.info("S3Handler context entered")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("S3Handler context exited")


    def close(self) -> None:
        logger.info("S3Handler.close() called. No persistent connection to close.")

    def load_from_source(self) -> pd.DataFrame:
        raise NotImplementedError("S3Handler does not support load_from_source directly.")

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        """
        Upload a local file to S3.
        """
        try:
            if not local_path.is_file():
                raise FileNotFoundError(f"Local file not found: {local_path.as_posix()}")

            self._client.upload_file(
                Filename=str(local_path),
                Bucket=self.config.bucket_name,
                Key=s3_key,
            )
            logger.info(
                "Uploaded: %s -> s3://%s/%s",
                local_path.as_posix(),
                self.config.bucket_name,
                s3_key,
            )
        except ClientError as e:
            logger.error("AWS ClientError during file upload: %s", str(e))
            raise StudentPerformanceError(e, logger) from e
        except Exception as e:
            logger.error("Unexpected error during file upload: %s", str(e))
            raise StudentPerformanceError(e, logger) from e

    def sync_directory(self, local_dir: Path, s3_prefix: str) -> None:
        """
        Recursively uploads a directory to S3.
        """
        try:
            if not local_dir.is_dir():
                raise NotADirectoryError(f"Local directory not found: {local_dir.as_posix()}")

            logger.info(
                "Starting directory sync: %s -> s3://%s/%s",
                local_dir.as_posix(),
                self.config.bucket_name,
                s3_prefix,
            )

            for root, _, files in os.walk(local_dir):
                for file in files:
                    local_file_path = Path(root) / file
                    relative_path = local_file_path.relative_to(local_dir)
                    remote_key = f"{s3_prefix}/{relative_path.as_posix()}"
                    self.upload_file(local_file_path, remote_key)

            logger.info(
                "Directory successfully synced: %s -> s3://%s/%s",
                local_dir.as_posix(),
                self.config.bucket_name,
                s3_prefix,
            )
        except Exception as e:
            logger.error("Directory sync to S3 failed.")
            raise StudentPerformanceError(e, logger) from e

    def load_csv(self, s3_uri: str) -> pd.DataFrame:
        """
        Load a CSV file from S3 into a DataFrame.
        """
        try:
            bucket, key = self._parse_s3_uri(s3_uri)
            obj = self._client.get_object(Bucket=bucket, Key=key)
            return pd.read_csv(obj["Body"])
        except Exception as e:
            logger.exception("Failed to load CSV from S3.")
            raise StudentPerformanceError(e, logger) from e

    def stream_csv(self, df: pd.DataFrame, s3_key: str) -> str:
        """
        Streams a DataFrame as CSV to S3 (in-memory, no local write).
        """
        try:
            buf = StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=buf.getvalue().encode("utf-8")
            )
            s3_uri = f"s3://{self.config.bucket_name}/{s3_key}"
            logger.info(f"Streamed CSV to: {s3_uri}")
            return s3_uri
        except Exception as e:
            logger.exception("Failed to stream CSV to S3.")
            raise StudentPerformanceError(e, logger) from e

    def stream_yaml(self, data: dict, s3_key: str) -> str:
        """
        Streams a Python dict (or list) as YAML to S3 (in-memory, no local write),
        converting any NumPy types into native Python scalars first.
        """
        def _convert(obj):
            if isinstance(obj, dict):
                return { _convert(k): _convert(v) for k, v in obj.items() }
            if isinstance(obj, list):
                return [ _convert(v) for v in obj ]
            if isinstance(obj, tuple):
                return tuple(_convert(v) for v in obj)
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        try:
            python_data = _convert(data)

            buf = StringIO()
            yaml.safe_dump(python_data, buf)
            buf.seek(0)

            self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=buf.getvalue().encode("utf-8"),
            )

            s3_uri = f"s3://{self.config.bucket_name}/{s3_key}"
            logger.info(f"Streamed YAML to: {s3_uri}")
            return s3_uri

        except Exception as e:
            logger.exception("Failed to stream YAML to S3.")
            raise StudentPerformanceError(e, logger) from e


    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        Parses s3://bucket/key into (bucket, key).
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        parts = s3_uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        return parts[0], parts[1]

    def stream_object(self, obj: object, s3_key: str) -> str:
        """
        Serialize a Python object (e.g. a fitted pipeline) via joblib
        and stream it to S3 under key=s3_key (in‐memory, no temp file).
        Returns the s3:// URI.
        """
        try:
            buf = BytesIO()
            # dump into the buffer
            joblib.dump(obj, buf)
            buf.seek(0)

            self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=buf.read(),
            )

            uri = f"s3://{self.config.bucket_name}/{s3_key}"
            logger.info("Streamed object to: %s", uri)
            return uri

        except Exception as e:
            logger.exception("Failed to stream object to S3.")
            raise StudentPerformanceError(e, logger) from e

    def stream_npy(self, array: np.ndarray, s3_key: str) -> str:
        """
        Serialize a NumPy array in .npy format into memory and upload it to S3.
        """
        try:
            buf = BytesIO()
            # write the .npy header + data
            np.save(buf, array, allow_pickle=False)
            buf.seek(0)

            self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=buf.read(),
            )

            uri = f"s3://{self.config.bucket_name}/{s3_key}"
            logger.info("Streamed .npy to: %s", uri)
            return uri

        except Exception as e:
            logger.exception("Failed to stream .npy to S3.")
            raise StudentPerformanceError(e, logger) from e

    def load_npy(self, s3_uri: str) -> np.ndarray:
        """
        Load a .npy‐serialized NumPy array from S3 into memory.
        
        Args:
            s3_uri:  The full S3 URI (e.g. "s3://bucket/key.npy").
        Returns:
            The deserialized NumPy array.
        """
        try:
            # parse out bucket and key
            bucket, key = self._parse_s3_uri(s3_uri)
            # fetch the object
            resp = self._client.get_object(Bucket=bucket, Key=key)
            data = resp["Body"].read()
            # load into a BytesIO and then np.load
            buf = BytesIO(data)
            buf.seek(0)
            arr = np.load(buf, allow_pickle=False)
            logger.info("Loaded .npy from S3: %s", s3_uri)
            return arr

        except Exception as e:
            logger.exception("Failed to load .npy from S3.")
            raise StudentPerformanceError(e, logger) from e
        
    def load_object(self, s3_uri: str) -> object:
        """
        Load a joblib‐serialized object from S3 into memory.

        Args:
            s3_uri: The S3 URI where the object was stored (e.g. "s3://bucket/key.joblib").

        Returns:
            The deserialized Python object.
        """
        try:
            # parse bucket & key
            bucket, key = self._parse_s3_uri(s3_uri)

            # fetch from S3
            resp = self._client.get_object(Bucket=bucket, Key=key)
            data = resp['Body'].read()

            # load via joblib from an in‐memory buffer
            buf = BytesIO(data)
            buf.seek(0)
            obj = joblib.load(buf)

            logger.info("Loaded object from S3: %s", s3_uri)
            return obj

        except Exception as e:
            logger.exception("Failed to load object from S3: %s", s3_uri)
            raise StudentPerformanceError(e, logger) from e

    def stream_df_as_csv(self, df: pd.DataFrame, s3_key: str) -> str:
        """
        Convert a pandas DataFrame to CSV in memory and stream it to S3.

        Args:
            df (pd.DataFrame): The DataFrame to stream.
            s3_key (str): The S3 key under which the CSV will be saved.

        Returns:
            str: The full s3:// URI where the CSV is stored.
        """
        try:
            buf = BytesIO()
            df.to_csv(buf, index=False)
            buf.seek(0)

            self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=buf.read(),
                ContentType='text/csv',
            )

            uri = f"s3://{self.config.bucket_name}/{s3_key}"
            logger.info("Streamed DataFrame as CSV to: %s", uri)
            return uri

        except Exception as e:
            logger.exception("Failed to stream DataFrame as CSV to S3.")
            raise StudentPerformanceError(e, logger) from e