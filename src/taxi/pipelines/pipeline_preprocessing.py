from taxi.components.Data import Data
from loguru import logger
from taxi.configs.config import *


def PipelinePreprocessing(data_obj):

    data_obj.read_dataset()
    df = data_obj.df[PARAMS.DATASET.COLUMNS_TO_USE]
    data_obj.eda()
    data_obj.data_split()
    X_train, y_train = data_obj.preprocessing(data_obj.X_train, data_obj.y_train.values)
    X_test, y_test = data_obj.preprocessing(data_obj.X_test, data_obj.y_test.values)

    return (X_train, y_train, X_test, y_test)
