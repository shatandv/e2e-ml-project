import os
import sys
from typing import Any

import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


def save_object(obj: Any, path: str) -> None:
    try:
        dir_path = os.path.dirname(path)

        if not os.path.exists(dir_path):
            os.makedirs([dir_path])
            logging.info(f"Created directory at {dir_path}")

        with open(path, "wb") as file:
            dill.dump(obj, file)
        logging.info(f"Saved object at {path}")

    except Exception as e:
        msg = f"Error while saving object: {e}"
        logging.error(msg)
        raise CustomException(msg, sys)
