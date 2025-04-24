from metaflow import FlowSpec, step, Parameter, conda
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

    @conda(python="3.9.16", libraries={
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.2",
        "mlflow": "2.12.1",
        "boto3": "1.34.59"
    })
    @step
    def start(self):
        print("Loading data...")
        df = pd.read_csv("https://storage.googleapis.com/storage-metaflandrea-metaflow-default/data/framingham.csv")
        df = df.dropna()
        self.X = df.drop("TenYearCHD", axis=1)
        self.y = df["TenYearCHD"]
        self.feature_names = self.X.columns.tolist()
        self.next(self.split_data)

    @conda(python="3.9.16", libraries={
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.2",
        "mlflow": "2.12.1",
        "boto3": "1.34.59"
    })
    @step
    def split_data(self):
        print("Splitting into train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.next(self.train_model)

    @conda(python="3.9.16", libraries={
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.2",
        "mlflow": "2.12.1",
        "boto3": "1.34.59"
    })
    @step
    def train_model(self):
        print("Training logistic regression...")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train.copy(), self.y_train.copy())
        print(f"Model output: {self.model}")
        self.next(self.evaluate_model)

    @conda(python="3.9.16", libraries={
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.2",
        "mlflow": "2.12.1",
        "boto3": "1.34.59"
    })
    @step
    def evaluate_model(self):
        print("Evaluating model...")
        preds = self.model.predict(self.X_test)
        preds = np.array(preds).astype(int)
        y_true = np.array(self.y_test).astype(int)

        self.accuracy = accuracy_score(y_true, preds)
        print(f"Accuracy: {self.accuracy:.4f}")
        self.next(self.register_model)

    @conda(python="3.9.16", libraries={
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.2",
        "mlflow": "2.12.1",
        "boto3": "1.34.59"
    })
    @step
    def register_model(self):
        print("Registering model to MLflow...")
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_registry_uri("http://mlflow:5000")
        mlflow.set_experiment("Framingham")     

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.sklearn.log_model(self.model, artifact_path="model", input_example=self.X_test[:5])
            mlflow.log_params({"solver": "lbfgs", "max_iter": 1000})
            mlflow.log_metric("accuracy", self.accuracy)

            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="FraminghamLogReg")
            print(f"Model registered with URI: {model_uri}")

        self.next(self.end)

    @conda(python="3.9.16", libraries={
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.2",
        "mlflow": "2.12.1",
        "boto3": "1.34.59"
    })
    @step
    def end(self):
        print("Flow complete.")

if __name__ == '__main__':
    TrainModelFlow()
