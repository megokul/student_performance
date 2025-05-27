from pathlib import Path
from ensure import ensure_annotations
from box import ConfigBox
import yaml
from box.exceptions import BoxKeyError, BoxTypeError, BoxValueError

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    if not path_to_yaml.exists():
        msg =f"YAML file not found at: '{path_to_yaml.as_posix()}'"
        raise FileNotFoundError(msg)
    try:
        with path_to_yaml.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except:
        raise Exception 