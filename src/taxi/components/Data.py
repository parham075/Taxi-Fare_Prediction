import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from taxi.utils.utils import *
from taxi.configs.config import *
from taxi.utils.helpers import *
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from taxi.utils.utils import *
from taxi.configs.config import *
from taxi.utils.helpers import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class Data:
    def __init__(self):
        self.config = CONFIG

    def read_dataset(self):
        """
        Extracts dataset from a zip file if not already extracted,
        loads it into a Pandas dataframe, and drops specified columns.
        """
        if not (
            os.path.exists(
                f"{self.config.Data.DATA_DIR}/{self.config.Data.DATA_FILE_NAME}"
            )
        ):
            with zipfile.ZipFile(self.config.Data.DATA_DIR_ZIP, "r") as zip_ref:
                zip_ref.extract(
                    self.config.Data.DATA_FILE_NAME, self.config.Data.DATA_DIR
                )
                zip_ref.close()
        self.df = pd.read_csv(
            f"{self.config.Data.DATA_DIR}/{self.config.Data.DATA_FILE_NAME}"
        ).drop(columns=PARAMS.DATASET.COLUMNS_TO_DROP)
        return self.df

    def calculate_percentiles_for_each_group(self):
        """
        Calculates percentiles for specified group columns and optionally for trip_distance categories.
        """
        ## Question A
        results = pd.DataFrame()
        group_columns = ["VendorID", "passenger_count", "payment_type"]
        # Calculate percentiles for each group column
        for group_col in group_columns:
            percentile_result = (
                self.df.groupby(group_col)
                .apply(calculate_percentiles, include_groups=False)
                .reset_index()
            )
            percentile_result[group_col] = percentile_result[group_col].apply(
                lambda x: f"{group_col}_{x}"
            )
            percentile_result.set_index(group_col, inplace=True)
            results = pd.concat([results, percentile_result])

        ####### Question A.1 (optional): Calculate percentiles for trip_distance categories

        # Calculate percentiles for trip_distance > 2.8
        self.df["trip_distance_bucket"] = np.where(
            self.df["trip_distance"] <= 2.8, "trip_distance<=2.8", "trip_distance>2.8"
        )

        percentile_over_2_8 = (
            self.df[self.df["trip_distance_bucket"] == "trip_distance>2.8"]
            .groupby(["trip_distance_bucket"])
            .apply(calculate_percentiles, include_groups=False)
            .reset_index()
        )
        percentile_over_2_8.set_index("trip_distance_bucket", inplace=True)
        percentile_under_eq_2_8 = (
            self.df[self.df["trip_distance_bucket"] == "trip_distance<=2.8"]
            .groupby(["trip_distance_bucket"])
            .apply(calculate_percentiles, include_groups=False)
            .reset_index()
        )
        percentile_under_eq_2_8.set_index("trip_distance_bucket", inplace=True)
        percentile_results = pd.concat(
            [results, percentile_over_2_8, percentile_under_eq_2_8]
        )
        self.df = self.df[PARAMS.DATASET.COLUMNS_TO_USE]
        return percentile_results

    @staticmethod
    def handle_outliers_tukey(df, columns):
        outlier_indices = []
        for col in columns:
            q1 = np.percentile(df[col], 25)  # First quartile (25th percentile)
            q3 = np.percentile(df[col], 75)  # Third quartile (75th percentile)
            iqr = q3 - q1  # Interquartile range
            # Calculate bounds for outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Identify outliers and replace them
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices.extend(df.index[outlier_mask])
            df.loc[outlier_mask, col] = np.clip(
                df.loc[outlier_mask, col], lower_bound, upper_bound
            )
            return df

    def eda(self):
        self.df = self.handle_outliers_tukey(
            self.df, columns=["trip_distance", "total_amount"]
        )
        # # Calculate correlation coefficient
        # correlation_coefficient = self.df['trip_distance'].corr(self.df['total_amount'])

        # # Create figure and axes
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # # Plot correlation matrix
        # correlation_matrix = self.df.corr()
        # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        # sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, center=0,
        #             square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=ax1)
        # ax1.set_title('Correlation Matrix')

        # # Plot scatter plot to know if there is any lineaity between trip_distance and total_amount
        # ax2.scatter(self.df['trip_distance'], self.df['total_amount'], alpha=0.5)
        # ax2.set_title(f'Scatter Plot\nTrip Distance vs Total Amount\nCorrelation: {correlation_coefficient:.2f}')
        # ax2.set_xlabel('Trip Distance')
        # ax2.set_ylabel('Total Amount')

        # Adjust layout
        # plt.tight_layout()
        # # Show plot
        # plt.show()

    def data_split(self):
        features = self.df[PARAMS.DATASET.FEATURES]
        target = self.df[PARAMS.DATASET.TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            target,
            test_size=PARAMS.DATASET.TEST_SIZE,
            random_state=PARAMS.DATASET.RANDOM_STATE,
        )

    @staticmethod
    def preprocessing(features, labels):
        # Handling Numericals
        # trip_distance Bucketization AND encoding
        for key in PARAMS.DATASET.NUMERICAL_FEATURES:
            # Create equal-width buckets for the numerical feature
            features["distance_bucket_equal_width"] = pd.qcut(
                features[key], q=10, duplicates="drop"
            )

            # One-hot encode the bucketized feature
            encoder = OneHotEncoder(sparse_output=False, drop="first")
            distance_buckets_encoded = encoder.fit_transform(
                features[["distance_bucket_equal_width"]]
            )

            # Create a DataFrame from the encoded features
            distance_buckets_encoded_df = pd.DataFrame(
                distance_buckets_encoded,
                columns=encoder.get_feature_names_out(["distance_bucket_equal_width"]),
            )

            # Rename the bucket columns
            new_column_names = {
                col: f"distance_bucket_{i+1}"
                for i, col in enumerate(distance_buckets_encoded_df.columns)
            }
            distance_buckets_encoded_df.rename(columns=new_column_names, inplace=True)

            # Concatenate the encoded features with the original DataFrame (excluding the original bucket column)
            df_encoded = pd.concat(
                [
                    features.drop(
                        columns=["distance_bucket_equal_width", key]
                    ).reset_index(drop=True),
                    distance_buckets_encoded_df.reset_index(drop=True),
                ],
                axis=1,
            )

        # Rescale the label
        # Initialize the RobustScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        labels = labels.reshape(-1, 1)

        # Fit and transform the labels
        scaled_labels = scaler.fit_transform(labels)

        # Convert scaled labels back to a Series
        labels = pd.Series(scaled_labels.flatten())

        return df_encoded, labels

    @staticmethod
    def save_csv(df):
        if not os.path.exists(f"{CONFIG.QA.PERCENTILE_DATAFRAME_PATH}"):
            os.makedirs(f"{CONFIG.QA.PERCENTILE_DATAFRAME_PATH}")
            df.to_csv(
                f"{CONFIG.QA.PERCENTILE_DATAFRAME_PATH}/{CONFIG.QA.PERCENTILE_DATAFRAME_FILE}"
            )
