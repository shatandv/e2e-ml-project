from dataclasses import dataclass
import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

    transformed_train_data_path: str = os.path.join(
        "artifacts", "train_transformed.csv"
    )
    transformed_test_data_path: str = os.path.join("artifacts", "train_test.csv")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Make the preprocessor object
        """
        try:
            num_features = ["writing_score", "reading_score"]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical pipeline created")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Categorical pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", num_pipeline, num_features),
                    ("categorical", cat_pipeline, cat_features),
                ]
            )
            logging.info("Preprocessor created")

            return preprocessor

        except Exception as e:
            msg = f"Error while getting the preprocessor: {e}"
            logging.error(msg)
            raise CustomException(msg, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.array, np.array, str]:
        # ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform the raw data - feature engineering, feature selection, etc.
        """
        logging.info("Starting data transformation")

        try:
            # Read the train and test data
            logging.info("Read train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Get preprocessor")
            preprocessor = self.get_preprocessor()

            # Features
            target_col = "math_score"
            num_cols = ["writing_score", "reading_score"]
            cat_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Drop the target columns from the input features
            logging.info("Drop target column from input features")
            feature_train_df = train_df.drop(columns=[target_col])
            target_train_df = train_df[target_col]
            feature_test_df = test_df.drop(columns=[target_col])
            target_test_df = test_df[target_col]

            # # Split the features into numeric and categorical
            # num_features = train_df.select_dtypes(
            #     exclude=["object", "category"]
            # ).columns
            # cat_features = train_df.select_dtypes(
            #     include=["object", "category"]
            # ).columns

            # num_transformer = StandardScaler()
            # cat_transformer = OneHotEncoder()

            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ("categorical", cat_transformer, cat_features),
            #         ("numeric", num_transformer, num_features),
            #     ],
            # )

            # Transform input features
            logging.info("Run preprocessor on train and test input features")
            # feature_train_df = pd.DataFrame(
            #     preprocessor.fit_transform(feature_train_df, target_train_df).toarray(),
            #     columns=preprocessor.get_feature_names_out(feature_train_df.columns),
            # )
            # feature_test_df = pd.DataFrame(
            #     preprocessor.transform(feature_test_df).toarray(),
            #     columns=preprocessor.get_feature_names_out(feature_test_df.columns),
            # )
            # # Add target back to DFs
            # logging.info("Concatenate target to transformed features")
            # train_df = pd.concat([train_df, target_train_df], axis=1)
            # test_df = pd.concat([test_df, target_test_df], axis=1)

            # Different way to transform
            feature_train_arr = preprocessor.fit_transform(feature_train_df)
            feature_test_arr = preprocessor.transform(feature_test_df)

            train_arr = np.c_[feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[feature_test_arr, np.array(target_test_df)]

            # Make sure the artifacts directory exists
            logging.info("Make sure the artifacts directory exists")
            os.makedirs(
                os.path.dirname(self.transformation_config.transformed_train_data_path),
                exist_ok=True,
            )

            # Save the preprocessing pipeline
            logging.info("Save the preprocessing pipeline")
            save_object(
                preprocessor,
                path=self.transformation_config.preprocessor_file_path,
            )

            # # Save the transformed data
            # logging.info("Save the transformed data")
            # train_df.to_csv(
            #     self.transformation_config.transformed_train_data_path,
            #     index=False,
            #     header=True,
            # )
            # test_df.to_csv(
            #     self.transformation_config.transformed_test_data_path,
            #     index=False,
            #     header=True,
            # )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            msg = f"Error while transforming the data: {e}"
            logging.error(msg)
            raise CustomException(msg, sys)
