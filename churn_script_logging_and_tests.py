"""
Test Predict Customer Churn

Author: Monika Venckauskaite
Date: January 13th, 2022
"""
import logging
from urllib import response

import pandas as pd
import pytest
import joblib
import numpy as np
import csv

from churn_library import (
    import_data,
    perform_eda,
    perform_feature_engineering,
    plot_bivariate,
    plot_column_variable,
    plot_heatmap,
)

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.DEBUG,
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    Test data import from file
    """
    try:
        # attempt import
        churn_data = import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        # check if dataframe is loaded
        assert isinstance(churn_data, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err
