from taxi.components.Data import Data
from loguru import logger
from taxi.configs.config import *


def PiPelineQA():
    data_obj = Data()
    df = data_obj.read_dataset()
    percentiles = data_obj.calculate_percentiles_for_each_group()
    data_obj.save_csv(percentiles)
    logger.info("Csv file is saved")
    return data_obj
