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

from churn_library import (classification_report_image, encoder_helper,
                           feature_importance_plot, import_data, perform_eda,
                           perform_feature_engineering, plot_bivariate,
                           plot_classification_report_lr,
                           plot_classification_report_rf, plot_column_variable,
                           plot_heatmap, plot_roc_curves_lr,
                           plot_roc_curves_rf, plot_train_models_lr,
                           plot_train_models_rf, train_logistic_regression,
                           train_models, train_random_forests)

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import_data():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        # attempt import
        data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        # check if dataframe is loaded
        assert isinstance(data_frame_churn_sample, pd.DataFrame)
        logging.info("Checking import_data imported datatype: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file is not convertible to pandas data frame"
        )
        raise err


def test_plot_column_variable():
    """
    ...
    """
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    column_variable = "Customer_Age"
    try:
        assert (
            plot_column_variable(data_frame_churn_sample, column_variable) == "Success"
        )
        logging.info("Checking plot_column_variable is working: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing plot_column_variable: plotting column variable was not successfull."
        )
        raise err


def test_plot_heatmap():
    """
    ...
    """
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    try:
        assert plot_heatmap(data_frame_churn_sample) == "Success"
        logging.info("Checking plot_heatmap imported is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_heatmap: creating heatmap was not successfull")
        raise err


def test_plot_bivariate():
    """
    ...
    """
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    try:
        column_variable_1 = "Customer_Age"
        column_variable_2 = "Gender"
        assert (
            plot_bivariate(
                data_frame_churn_sample, column_variable_1, column_variable_2
            )
            == "Success"
        )
        logging.info("Checking plot_bivariate imported is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_bivariate: creating heatmap was not successfull")
        raise err


def test_perform_eda():
    """
    ...
    """
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    try:
        data_frame_churn_sample = perform_eda(data_frame_churn_sample)
        assert isinstance(data_frame_churn_sample, pd.DataFrame)
        logging.info("Checking perform_eda is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: EDA was not successfull")
        raise err


def test_perform_feature_engineering():
    """
    ...
    """
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data_frame_churn_sample, '_Churn'
        )
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info("Checking perform_feature_engineering is working: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: feature engineering was not successfull"
        )
        raise err


def test_encoder_helper():
    """
    ...
    """
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    columns_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    try:
        assert isinstance(
            encoder_helper(data_frame_churn_sample, columns_to_encode, '_Churn'),
            pd.DataFrame,
        )
        logging.info("Checking encoder_helper is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: not successfull")
        raise err


def test_classification_report_image():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    # x_train, x_test, y_train, y_test = perform_feature_engineering(
    #     data_frame_churn_sample, '_Churn'
    # )
    # y_test_preds_rf, y_train_preds_rf = train_random_forests(x_train, x_test, y_train)
    y_train = [0, 0, 0 ,0, 0, 1, 1, 1, 1, 1]
    y_test = [0, 0, 0 ,0, 1, 0, 1, 1, 1, 1]
    y_train_preds_rf = [0, 0, 0 ,1, 0, 1, 1, 1, 1, 1]
    y_test_preds_rf = [0, 0, 0 ,0, 0, 1, 1, 0, 1, 1]
    try:
        assert (
            classification_report_image(
                y_train, y_test, y_train_preds_rf, y_test_preds_rf
            )
            == "Success"
        )
        logging.info("Checking classification_report_image is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing classification_report_image: not successfull")
        raise err

def test_train_random_forests():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    try:
        # y_train = np.array([0,0,1,0,1,0,1,0,0,0,0,1,0,0])
        y_test_preds_rf, y_train_preds_rf = train_random_forests(x_train, x_test, y_train, './test_results/rfc_model.pkl')
        assert isinstance(y_test_preds_rf, np.ndarray)
        assert isinstance(y_train_preds_rf, np.ndarray)
        logging.info("Checking train_random_forests is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_random_forests: not successfull")
        raise err

def test_train_logistic_regression():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    try:
        with open('result.txt', 'w') as csv_file:
            cw = csv.writer(csv_file, delimiter=',')
            cw.writerow(y_train)
        y_train = np.array([0,0,1,0,1,0,1,0,0,0,0,1,0,0])
        y_test_preds_rf, y_train_preds_rf = train_logistic_regression(x_train, x_test, y_train, './test_results/rf_model.pkl')
        assert isinstance(y_test_preds_rf, np.ndarray)
        assert isinstance(y_train_preds_rf, np.ndarray)
        logging.info("Checking train_logistic_regression is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_logistic_regression: not successfull")
        raise err

def test_train_models():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    try:
        assert train_models(x_train, x_test, y_train, y_test) == 'Success'
        logging.info("Checking train_models is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: not successfull")
        raise err


def test_feature_importance_plot():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    cv_rfc = joblib.load('./test_data/rfc_model.pkl')
    try:
        assert feature_importance_plot(
            cv_rfc,
            x_test,
            './test_results/rfc_feature_importance.png') == 'Success'
        logging.info("Checking feature_importance_plot is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_importance_plot: not successfull")
        raise err


def test_plot_roc_curves_lr():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    try:
        assert plot_roc_curves_lr(x_test, y_test, './test_data/lr_model.pkl', './test_results/ROC_curve_lr.png') == 'Success'
        logging.info("Checking plot_roc_curves_lr is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_roc_curves_lr: not successfull")
        raise err


def test_plot_roc_curves_rf():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    try:
        assert plot_roc_curves_rf(x_test, y_test, './test_data/rfc_model.pkl', './test_results/ROC_curve_rfc.png') == 'Success'
        logging.info("Checking plot_roc_curves_rfc is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_roc_curves_rfc: not successfull")
        raise err


def test_plot_classification_report_lr():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    y_test_preds_lr, y_train_preds_lr = train_logistic_regression(x_train, x_test, y_train, './test_results/lr_model.pkl')
    try:
        assert plot_classification_report_lr(
                y_test,
                y_train,
                y_test_preds_lr,
                y_train_preds_lr) == 'Success'
        logging.info("Checking plot_classification_report_lr is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_classification_report_lr: not successfull")
        raise err


def test_plot_classification_report_rf():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    y_test_preds_rf, y_train_preds_rf = train_random_forests(x_train, x_test, y_train, './test_results/rfc_model.pkl')
    try:
        assert plot_classification_report_rf(
                y_test,
                y_train,
                y_test_preds_rf,
                y_train_preds_rf) == 'Success'
        logging.info("Checking plot_classification_report_rf is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_classification_report_rf: not successfull")
        raise err


def test_plot_train_models_lr():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    y_train = np.array([0,0,1,0,1,0,1,0,0,0,0,1,0,0])
    y_test_preds_lr, y_train_preds_lr = train_logistic_regression(x_train, x_test, y_train, './test_results/lr_model.pkl')
    try:
        assert plot_train_models_lr(
                y_test,
                y_train,
                y_test_preds_lr,
                y_train_preds_lr,
                x_test) == 'Success'
        logging.info("Checking plot_train_models_lr is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_train_models_lr: not successfull")
        raise err

def test_plot_train_models_rf():
    data_frame_churn_sample = import_data(r"./test_data/data_churn_test.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame_churn_sample, '_Churn'
    )
    # y_train = np.array([0,0,1,0,1,0,1,0,0,0,0,1,0,0])
    y_test_preds_rf, y_train_preds_rf = train_random_forests(
        x_train, x_test, y_train, './models/rfc_model.pkl')
    try:
        assert plot_train_models_rf(
            y_test,
            y_test_preds_rf,
            y_train_preds_rf,
            y_train,
            x_test) == 'Success'
        logging.info("Checking plot_train_models_rf is working: SUCCESS")
    except AssertionError as err:
        logging.error("Testing plot_train_models_rf: not successfull")
        raise err