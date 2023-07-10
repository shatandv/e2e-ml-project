import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            processed_data = preprocessor.transform(data)
            pred = model.predict(processed_data)

            return pred

        except Exception as e:
            raise CustomException(e, sys)


class InputData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def as_df(self):
        try:
            input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(input_dict)

            return df
        except Exception as e:
            raise CustomException(e, sys)
