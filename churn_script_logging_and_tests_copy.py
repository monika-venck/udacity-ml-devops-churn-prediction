'''
Test Predict Customer Churn

Author: Monika Venckauskaite
Date: January 13th, 2022
'''
import logging
import pandas as pd
from churn_library import import_data
# # import churn_library_solution as cls
# import joblib

# , perform_eda, encoder_helper,\
# perform_feature_engineering, classification_report_image,\
# feature_importance_plot, train_models


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        #attempt import
        df_churn = import_data(r"./data/test_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        #check if dataframe is loaded
        assert isinstance(df_churn, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


# def test_eda():
#     '''
#     test perform eda function
#     '''
#     try:
#         #get test dataframe
#         df_churn = import_data(r"./data/test_data.csv")
#         #attempt to perform eda
#         status = perform_eda(df_churn)
#         assert status=='Done'
#         logging.info("Testing perform_eda: SUCCESS")
#     except AssertionError as err:
#         logging.error(f"Testing perform_eda, ERROR: {err}")


# def test_encoder_helper():
#     '''
#     test encoder helper
#     '''
#     try:
#         #get test data frame
#         df_churn = import_data(r"./data/test_data.csv")
#         #attempt to encode helper
#         df_enc = encoder_helper(df_churn, ['Income_Category', 'Card_Category'], 'response')
#         logging.info("Testing encoder_helper: SUCCESS")
#     except BaseException as err:
#         logging.error(f"Testing encoder_helper, ERROR: {err}")


# def test_perform_feature_engineering():
#     '''
#     test perform_feature_engineering
#     '''
#     try:
#         #get test data frame
#         df_churn = import_data(r"./data/test_data.csv")
#         #atempt to perform feature engineering
#         x_train, x_test, y_train, y_test = perform_feature_engineering(
#             df_churn, 'Done')
#         assert isinstance(x_train, pd.DataFrame)
#         assert isinstance(x_test, pd.DataFrame)
#         assert isinstance(y_train, pd.DataFrame)
#         assert isinstance(y_test, pd.DataFrame)
#         logging.info("Testing perform_feature_engineering: SUCCESS")
#     except AssertionError as err:
#         logging.error(f"Testing perform_feature_engineering, ERROR: {err}")

# def test_classification_report_image():
#     '''test classification_report'''
#     try:
#         #get test data frame
#         df_churn = import_data("./data/test_data.csv")
#         #get needed constants
#         x_train, x_test, y_train, y_test = perform_feature_engineering(
#             df_churn, 'response')
#         rfc_model = joblib.load('./models/rfc_model.pkl')
#         y_train_preds_rf = rfc_model.predict(x_train)
#         y_test_preds_rf = rfc_model.predict(x_test)
#         #attempt classification_report_image
#         status = classification_report_image(y_train, y_test, y_train_preds_rf, y_test_preds_rf)
#         assert status=='Done'
#         logging.info("Testing classification_report_image: SUCCESS")
#     except AssertionError as err:
#         logging.error(f"Testing classification_report_image, ERROR: {err}")

# def test_feature_importance_plot():
#     '''test feature_importance_plot'''
#     try:
#         model = joblib.load('./models/logistic_model.pkl')
#         #get test data frame
#         df_churn = import_data("./data/test_data.csv")
#         #get the needed constants
#         X_train, x_test, y_train, y_test = perform_feature_engineering(
#             df_churn, 'response')
#         x_data = x_test
#         output_pth = './test_feat_imp.png'
#         #atempt feature_importance_plot
#         feature_importance_plot(model, x_data, output_pth)
#         logging.info("Plotting feature_importance: SUCCESS")
#     except BaseException as err:
#         logging.error(f"Testing feature_importance, ERROR: {err}")

# def test_train_models():
#     '''
#     test train_models
#     '''
#     try:
#         # get needed constants
#         df_churn = import_data(r"./data/test_data.csv")
#         x_train, x_test, y_train, y_test = perform_feature_engineering(
#             df_churn, 'Done')
#         # attempt train_models
#         status = train_models(x_train, x_test, y_train, y_test)
#         assert status=='Done'
#         logging.info("Testing train_models: SUCCESS")
#     except AssertionError as err:
#         logging.error(f"Testing train_models, ERROR: {err}")
#         raise err


# if __name__ == "__main__":
#     print("Running main.")
# #     df = import_data(r"./data/test_data.csv")
# #     perform_eda(df)
# #     encoder_helper(df, ['Income_Category', 'Card_Category'], 'response')
# #     X_train, X_test, y_train, y_test = perform_feature_engineering(df, '')
#     #train_models(X_train, X_test, y_train, y_test)
