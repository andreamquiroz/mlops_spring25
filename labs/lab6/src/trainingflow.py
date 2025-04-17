from metaflow import FlowSpec, step, Parameter, NBRunner
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TrainModelFlow(FlowSpec):
    """
    Metaflow Training Flow
    Author: Andrea Quiroz
    Description: Loads Framingham dataset, trains logistic regression model,
    logs + registers model using MLflow.
    """
    data_path = Parameter('data_path', help="Path to Framingham CSV", default="data/framingham.csv")

    @step
    def start(self):
        # Load and split the dataset
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df = df.dropna() # drop rows with missing values
        self.X = df.drop("TenYearCHD", axis=1)
        self.y = df["TenYearCHD"]

        # save feat names for reference in mlflow ui
        self.feature_names = self.X.columns.tolist()
        self.next(self.split_data)

    @step
    def split_data(self):
        print("Splitting into train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.next(self.train_model)
	   
    @step
    def train_model(self):
        # Training a simple model (e.g., linear regression)
        # using log reg
        print("Training logistic regression...")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        print(f"Model output: {self.model}")
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        print("Evaluating model...")
        preds = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, preds)
        print(f"Accuracy: {self.accuracy:.4f}")
        self.next(self.register_model)

    @step
    def register_model(self):
        print("Logging and registering model to MLflow...")
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment("FramingHeartStudy_L6")

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            artifact_path = "model"

            mlflow.set_tags({"Model": "log-reg", "Train Data": "all-data"})
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.set_tag("feature_columns", ", ".join(self.feature_names))

            mlflow.sklearn.log_model(self.model, artifact_path=artifact_path)

        # safe to use run_id and experiment_id outside
        client = MlflowClient()

        try:
            client.create_registered_model("log-reg")
        except Exception:
            pass

        source_uri = f"runs:/{run_id}/{artifact_path}"

        client.create_model_version(
            name="log-reg",
            source=source_uri,
            run_id=run_id
        )

        self.next(self.end)
	   
    @step
    def end(self):
        # Final step
        print("Training complete. Model registered with MLFlow!")

if __name__ == '__main__':
    TrainModelFlow()