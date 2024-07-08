# pipelines

from taxi.components.Data import Data
from loguru import logger
from taxi.configs.config import *
from taxi.components.Model import Model




def main():
  step_name = 'QA and QA.1'
  print("************************\n************************\n************************")
  logger.info(f"step_name:  {step_name}")
  data_obj = Data()
  df = data_obj.read_dataset()
  percentiles = data_obj.calculate_percentiles_for_each_group()
  data_obj.save_csv(percentiles)
  logger.info("Csv file is saved")

  ####################################
  ####################################
  step_name = 'preprocessing'
  print("************************\n************************\n************************")
  logger.info(f"step_name:  {step_name}")
  
  data_obj.read_dataset()
  df = data_obj.df[PARAMS.DATASET.COLUMNS_TO_USE]
  data_obj.eda()
  data_obj.data_split()
  X_train, y_train = data_obj.preprocessing(data_obj.X_train,data_obj.y_train.values)
  X_test, y_test = data_obj.preprocessing(data_obj.X_test,data_obj.y_test.values)
  logger.info("Data is preprocessed")

  ########################################
  ###################################
  step_name = 'Train and Evaluate'
  print("************************\n************************\n************************")
  logger.info(f"step_name:  {step_name}")
  model_obj = Model(config=CONFIG, params=PARAMS, x__train=X_train[:100000],y__train=y_train[:100000], x__test=X_test,y__test=y_test )
  model_obj.random_forest_regression_model()
  estimators = model_obj.training()
  best_model = model_obj.evaluation(estimators)
  model_obj.save_model(model_path=f'{CONFIG.Model.MODEL_PATH}/{CONFIG.Model.RF_MODEL}/',model_file=f'{CONFIG.Model.MODEL_FILE}')
  logger.info("Model is trained and evaluated")
if __name__ == "__main__":
    # Log the experiment
    main()