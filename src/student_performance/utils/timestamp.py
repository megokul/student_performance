from datetime import datetime, timezone

_timestamp_cache: str | None = None

def get_utc_timestamp() -> str:
    format="%Y_%m_%dT%H_%M_%SZ"
    global _timestamp_cache
    if _timestamp_cache is None:
        _timestamp_cache = datetime.now(timezone.utc).strftime(format)
    return _timestamp_cache
