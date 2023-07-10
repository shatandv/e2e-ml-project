from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import InputData, PredictPipeline

app = Flask(__name__)


# Homepage
@app.route("/")
def index():
    return render_template("index.html")


# Prediction endpoint
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")

    data = InputData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=request.form.get("reading_score"),
        writing_score=request.form.get("writing_score"),
    )

    data_df = data.as_df()

    pipeline = PredictPipeline()
    preds = pipeline.predict(data_df)
    pred = preds[0]

    return render_template("home.html", results=pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
