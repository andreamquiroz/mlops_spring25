from metaflow import FlowSpec, step, Parameter, conda_base, retry, timeout, catch, resources
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@conda_base(python="3.9.16", libraries={
    "pandas": "2.2.3",
    "numpy": "1.23.5",
    "scikit-learn": "1.2.2",
    "mlflow": "2.11.1"
})
class TrainModelFlow(FlowSpec):
    """
    Metaflow Training Flow
    Author: Andrea Quiroz
    Description: Loads Framingham dataset, trains logistic regression model,
    logs + registers model using MLflow.
    """
    data_path = Parameter('data_path', help="Path to Framingham CSV", default="data/framingham.csv")

    @step
    @retry(times=2)
    @timeout(seconds=120)
    def start(self):
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        self.X = df.drop("TenYearCHD", axis=1)
        self.y = df["TenYearCHD"]
        self.feature_names = self.X.columns.tolist()
        self.next(self.split_data)

    @step
    @retry(times=2)
    @timeout(seconds=60)
    def split_data(self):
        print("Splitting into train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.next(self.train_model)

    @step
    @retry(times=2)
    @timeout(seconds=300)
    @resources(cpu=2, memory=4096)
    def train_model(self):
        print("Training logistic regression...")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        print(f"Model output: {self.model}")
        self.next(self.evaluate_model)

    @step
    @retry(times=2)
    @timeout(seconds=120)
    def evaluate_model(self):
        print("Evaluating model...")
        preds = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, preds)
        print(f"Accuracy: {self.accuracy:.4f}")
        self.next(self.register_model)

    @step
    @retry(times=2)
    @timeout(seconds=120)
    def register_model(self):
        print("Registering model to MLflow...")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, "model", input_example=self.X_test[:5])
            mlflow.log_params({"solver": "lbfgs", "max_iter": 1000})
            mlflow.log_metric("accuracy", self.accuracy)
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, "FraminghamLogReg")
        print("Model registered.")
        self.next(self.end)

    @step
    def end(self):
        print("Flow complete.")

if __name__ == '__main__':
    TrainModelFlow()
