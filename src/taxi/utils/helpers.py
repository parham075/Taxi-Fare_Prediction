import pandas as pd
from taxi.configs.config import *
import numpy as np
def calculate_percentiles(group):
    result = {}
    value_columns = ['fare_amount', 'tip_amount', 'total_amount']
    for column in value_columns:
        for percentile in PARAMS.PERCENTILES:
            result[f'{column}_p_{percentile}'] = np.percentile(group[column].dropna(), percentile)
    return pd.Series(result)