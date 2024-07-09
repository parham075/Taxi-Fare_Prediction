from sklearn.ensemble import RandomForestRegressor as skRFR
from sklearn.linear_model import LinearRegression as skLR
from sklearn.ensemble import GradientBoostingRegressor as skXGR
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score
import pickle
import os
import subprocess
import time
import requests
from loguru import logger
import mlflow
from urllib.parse import urljoin, urlparse


class Model:
    def __init__(self, config, params, x__train, y__train, x__test, y__test) -> None:
        self.config = config
        self.params = params
        self.X_train = x__train
        self.y_train = y__train
        self.X_test = x__test
        self.y_test = y__test

    @staticmethod
    def is_mlflow_server_running():
        """Check if the MLflow server is running by sending a request to the tracking URI."""
        url = "http://127.0.0.1:5000"
        try:
            response = requests.get(url)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    def start_mlflow_server(self):
        """Start the MLflow server if it is not already running."""
        if not self.is_mlflow_server_running():
            print("MLflow server is not running. Starting the server...")
            process = subprocess.Popen(
                [
                    "mlflow",
                    "server",
                    "--backend-store-uri",
                    "sqlite:///mlflow.db",
                    "--default-artifact-root",
                    "./mlruns",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "5000",
                ]
            )
            logger.info("Waiting for the MLflow server to start...")
            time.sleep(5)  # Give the server some time to start
            if self.is_mlflow_server_running():
                print("MLflow server started successfully.")
            else:
                print("Failed to start the MLflow server.")
        else:
            print("MLflow server is already running.")

    def linear_regression_model(self):
        params = self.params.Models.LinearRegressionModel.HYPERPARAMETERS
        self.model = skLR(**params)

    def random_forest_regression_model(self):
        params = self.params.Models.RandomForestModel.HYPERPARAMETERS
        self.model = skRFR(**params)

    def gradient_boost_regression(self):
        params = self.params.Models.XGBoostModel.HYPERPARAMETERS
        self.model = skXGR(**params)

    def training(self, folds=3):
        kf = KFold(folds)
        kf.get_n_splits(self.X_train)
        score = 0.0
        models = []
        for trainIdx, validIdx in kf.split(self.X_train):
            X_train_valid, X_valid = (
                self.X_train.iloc[trainIdx],
                self.X_train.iloc[validIdx],
            )
            y_train_valid, y_test_valid = self.y_train[trainIdx], self.y_train[validIdx]
            self.model.fit(X_train_valid, y_train_valid)
            score = self.model.score(X_valid, y_test_valid)
            print("score = ", score)
            models.append(self.model)
        return models

    def evaluation(self, estimators):
        estimator_idx = 0
        self.best_estimator_rmse = float("inf")
        for estimator in estimators:
            estimator_idx = estimator_idx + 1
            y_test_pred = estimator.predict(self.X_test)
            rmse = root_mean_squared_error(self.y_test, y_test_pred)

            if rmse < self.best_estimator_rmse:
                self.best_estimator_rmse = rmse
                self.best_model = estimator
            logger.info(
                f"\nestimator_idx: {estimator_idx}, current_estimator_rmse: {rmse},best_estimator_rmse: {self.best_estimator_rmse}"
            )

        print(f"Best RMSE: {self.best_estimator_rmse:.4f}")

    def save_and_log_model(self, model_path, model_file, model_name):
        mlflow.sklearn.autolog(
            log_input_examples=False,
            log_model_signatures=True,
            log_models=True,
            log_datasets=True,
            log_post_training_metrics=True,
            serialization_format="cloudpickle",
            registered_model_name=f"{model_name}",
            pos_label=None,
            extra_tags=None,
        )
        mlflow.log_metric("RMSE", self.best_estimator_rmse)

        # Log the model
        mlflow.sklearn.log_model(self.best_model, f"{model_name}")
        # Save model also in a seprate file
        if not os.path.exists(f"{model_path}"):
            os.makedirs(f"{model_path}", exist_ok=True)
            with open(model_path + "/" + model_file, "wb") as file:
                pickle.dump(self.model, file)
