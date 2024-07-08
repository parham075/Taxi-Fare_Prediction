from sklearn.ensemble import RandomForestRegressor as skRFR
from sklearn.linear_model import LinearRegression as skLR
from sklearn.ensemble import GradientBoostingRegressor as skXGR
from sklearn.model_selection import  KFold
from sklearn.metrics import root_mean_squared_error, r2_score
import pickle
import os

class Model:
  def __init__(self, config,params, x__train, y__train, x__test, y__test ) -> None:
    self.config  = config
    self.params  = params
    self.X_train = x__train
    self.y_train = y__train
    self.X_test = x__test
    self.y_test = y__test
  def linear_regression_model(self):
    params = self.params.Models.LinearRegressionModel.HYPERPARAMETERS
    self.model = skLR(**params)
  def random_forest_regression_model(self):
    params = self.params.Models.RandomForestModel.HYPERPARAMETERS
    self.model = skRFR(**params)
  def gradient_boost_regression(self):
    params = self.params.Models.XGBoostModel.HYPERPARAMETERS
    self.model = skXGR(**params)
  def training(self, folds= 3):
    kf = KFold(folds)
    kf.get_n_splits(self.X_train)
    score = 0.0
    models = []
    for trainIdx, validIdx in kf.split(self.X_train):
        X_train_valid, X_valid = self.X_train.iloc[trainIdx], self.X_train.iloc[validIdx]
        y_train_valid, y_test_valid = self.y_train[trainIdx], self.y_train[validIdx]
        self.model.fit(X_train_valid, y_train_valid)
        score = self.model.score(X_valid, y_test_valid)
        print("score = ", score)
        models.append(self.model)
    return models
  def evaluation(self, estimators):
    estimator_idx=0
    best_estimator_rmse = float('inf')
    for estimator in estimators:
        estimator_idx = estimator_idx+1
        y_test_pred = estimator.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_test_pred)
        
        if rmse < best_estimator_rmse:
          best_estimator_rmse = rmse
          best_model = estimator
        print("\nestimator_idx: {}, current_estimator_rmse: {},best_estimator_rmse: {}".format(estimator_idx,rmse,best_estimator_rmse))
    
    print(f"Best RMSE: {best_estimator_rmse:.4f}, R2_score: {r2_score(self.y_test, best_model.predict(self.X_test)):.4f}")
        
    return best_model
  def save_model(self,model_path, model_file):
    if not os.path.exists(f'{model_path}'): 
      os.makedirs(f'{model_path}',exist_ok=True)
      with open(model_path+"/"+model_file, 'wb') as file:
        pickle.dump(self.model, file)