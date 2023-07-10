from dataclasses import dataclass
import os
import sys

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, get_best_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_filepath: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression,
                "Decision Tree": DecisionTreeRegressor,
                "K-Neighbors": KNeighborsRegressor,
                "Random Forest": RandomForestRegressor,
                "AdaBoost": AdaBoostRegressor,
                "Gradient Boost": GradientBoostingRegressor,
                "XGBoost": XGBRegressor,
                "CatBoost": CatBoostRegressor,
                "LightGBM": LGBMRegressor,
            }

            model_params = {
                "Linear Regression": {},
                "Decision Tree": {
                    "max_depth": [2, 4, 6, 8, 10, 12],
                    # "min_samples_split": [2, 4, 6, 8, 10, 12],
                    # "min_samples_leaf": [2, 4, 6, 8, 10, 12],
                },
                "K-Neighbors": {
                    "n_neighbors": [2, 4, 6, 8, 10, 12],
                    # "weights": ["uniform", "distance"],
                    # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    # "leaf_size": [2, 4, 6, 8, 10, 12],
                    # "p": [1, 2],
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [6, 8, 10, 12],
                    # "min_samples_split": [2, 4, 6, 8, 10, 12],
                    # "min_samples_leaf": [2, 4, 6, 8, 10, 12],
                },
                "AdaBoost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    # "loss": ["linear", "square", "exponential"],
                },
                "Gradient Boost": {
                    # "loss": ["ls", "lad", "huber", "quantile"],
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "n_estimators": [100, 200, 300],
                    # "subsample": [0.1, 0.5],
                    # "max_depth": [6, 8, 10, 12],
                },
                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [6, 8, 10, 12],
                    # "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    # "subsample": [0.1, 0.5],
                },
                "CatBoost": {
                    "iterations": [100, 200, 300],
                    # "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "depth": [6, 8, 10, 12],
                },
                "LightGBM": {
                    # "boosting_type": ["gbdt", "dart", "goss", "rf"],
                    "max_depth": [6, 8, 10, 12],
                    # "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "n_estimators": [100, 200, 300],
                },
            }

            model_scores = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                model_params=model_params,
            )

            best_model, best_score = get_best_model(model_scores, models)

            if best_score < 0.6:
                raise CustomException(
                    "Best model score is less than 0.6. Please try again with different data.",
                    sys,
                )

            logging.info(f"Best model has score {best_score}")

            save_object(best_model, self.model_trainer_config.trained_model_filepath)

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
