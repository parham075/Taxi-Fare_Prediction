from pathlib import Path
from taxi.utils.utils import read_yaml
import os

CONFIG_FILE_PATH = Path("config.yaml")
CONFIG = read_yaml(CONFIG_FILE_PATH)
PARAMS_PATH = Path("params.yaml")
PARAMS = read_yaml(PARAMS_PATH)