from metaflow import FlowSpec, step, Parameter, conda
import pandas as pd
import mlflow
import mlflow.sklearn
import datetime
import os

class ScoreModelFlow(FlowSpec):
    """
    Scoring Model Flow
    Author: Andrea Quiroz
    Description: Loads Framingham dataset, scores predictions using an MLflow model,
    and writes output to CSV.
    """

    data_path = Parameter(
        'data_path',
        help="Path to CSV for scoring",
        default="https://storage.googleapis.com/storage-metaflandrea-metaflow-default/data/framingham.csv"
    )
    model_name = Parameter('model_name', help="Registered MLflow model name", default="FraminghamLogReg")

    @conda(python="3.9.16", libraries={
        "pandas": "2.2.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.11.1"
    })
    @step
    def start(self):
        print("Loading data for scoring...")
        df = pd.read_csv(self.data_path).dropna()

        self.X = df.drop("TenYearCHD", axis=1)  # remove target if present
        self.y = df["TenYearCHD"]  # keep for evaluating if you want
        self.df_ids = df.reset_index()[["index"]]  # keep index for saving

        self.next(self.load_model)

    @conda(python="3.9.16", libraries={
        "pandas": "2.2.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.11.1"
    })
    @step
    def load_model(self):
        print("Loading registered model from MLflow...")
        mlflow.set_tracking_uri("http://mlflow:5000")

        mlflow.set_tracking_uri("http://mlflow:5000")
        model_uri = "gs://your-public-bucket/path/to/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        
        print("Model loaded successfully.")
        self.next(self.predict)

    @conda(python="3.9.16", libraries={
        "pandas": "2.2.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.11.1"
    })
    @step
    def predict(self):
        print("Scoring data...")
        self.preds = self.model.predict(self.X)
        self.df_preds = self.df_ids.copy()
        self.df_preds["PredictedCHD"] = self.preds
        self.df_preds["TrueCHD"] = self.y.values  # optional
        self.next(self.save_predictions)

    @conda(python="3.9.16", libraries={
        "pandas": "2.2.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.11.1"
    })
    @step
    def save_predictions(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"predictions/predictions_{timestamp}.csv"
        self.df_preds.to_csv(self.output_path, index=False)
        print(f"Predictions saved to {self.output_path}")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")

if __name__ == '__main__':
    ScoreModelFlow()
