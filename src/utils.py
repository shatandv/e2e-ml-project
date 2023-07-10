import os
import sys
from typing import Any, Dict

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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


def evaluate_models(x_train, y_train, x_test, y_test, models) -> Dict[str, float]:
    try:
        scores = {}

        for key, model in models.items():
            model.fit(x_train, y_train)
            # y_train_pred = model.predict(x_train)
            # train_score = r2_score(y_train, y_train_pred)
            y_test_pred = model.predict(x_test)
            test_score = r2_score(y_test, y_test_pred)
            scores[key] = test_score

        return scores

    except Exception as e:
        raise CustomException(e, sys)
