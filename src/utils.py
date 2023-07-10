import itertools
import os
import sys
from typing import Any, Dict, List

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

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


def load_object(path: str) -> Any:
    try:
        with open(path, "rb") as file:
            obj = dill.load(file)
            logging.info(f"Loaded object from {path}")

        return obj
    except Exception as e:
        msg = f"Error while loading object: {e}"
        logging.error(msg)
        raise CustomException(msg, sys)


def get_param_combinations(params: Dict[str, Any]) -> List[dict]:
    try:
        if len(params) == 0:
            param_combinations = [{}]
        else:
            param_names = params.keys()
            values = params.values()
            param_combinations = [
                dict(zip(param_names, v)) for v in itertools.product(*values)
            ]

        return param_combinations

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    x_train, y_train, x_test, y_test, models, model_params
) -> Dict[str, float]:
    try:
        scores = {}

        for key, model_algo in models.items():
            params = get_param_combinations(model_params[key])
            scores[key] = []

            for param in tqdm(params):
                logging.info(f"Model: {key}, params: {param}")
                model = model_algo(**param)

                model.fit(x_train, y_train)

                y_test_pred = model.predict(x_test)
                test_score = r2_score(y_test, y_test_pred)

                scores[key].append((model, test_score))

        return scores

    except Exception as e:
        raise CustomException(e, sys)


def get_best_model(scores: Dict[str, float], models: Dict[str, Any]) -> str:
    try:
        best_scores_per_model_type = {
            model_type: max(model_scores, key=lambda x: x[1])
            for model_type, model_scores in scores.items()
        }

        logging.info(f"Best scores per model: {best_scores_per_model_type}")

        best_model_name, (best_model, best_score) = max(
            best_scores_per_model_type.items(), key=lambda x: x[1][1]
        )
        logging.info(
            f"Best model: {best_model_name}, score: {best_score}, with params: {best_model}"
        )
        return best_model, best_score
    except Exception as e:
        raise CustomException(e, sys)
