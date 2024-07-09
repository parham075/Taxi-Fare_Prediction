from loguru import logger
from taxi.configs.config import *
from taxi.components.Model import Model
import mlflow
from urllib.parse import urljoin, urlparse


def PipelineTrainAndEvaluation(X_train, y_train, X_test, y_test):
    model_obj = Model(
        config=CONFIG,
        params=PARAMS,
        x__train=X_train[:100000],
        y__train=y_train[:100000],
        x__test=X_test,
        y__test=y_test,
    )
    for model_name in PARAMS.Models:
        model_obj.start_mlflow_server()
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        tracking_url_type_store = urlparse("http://127.0.0.1:5000").scheme
        mlflow.set_experiment("Taxi_Fare")
        with mlflow.start_run() as run:
            # print(mlflow.active_run().info,'\n', mlflow.run.info)

            if model_name == "LinearRegressionModel":
                model_obj.linear_regression_model()
            if model_name == "RandomForestModel":
                model_obj.random_forest_regression_model()
            if model_name == "XGBoostModel":
                model_obj.gradient_boost_regression()

            estimators = model_obj.training()
            model_obj.evaluation(estimators)

            model_obj.save_and_log_model(
                model_path=f"{CONFIG.Model.MODEL_PATH}/{model_name}/",
                model_file=f"{CONFIG.Model.MODEL_FILE}",
                model_name=model_name,
            )
            mlflow.end_run()
