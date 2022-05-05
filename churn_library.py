'''
Predict Customer Churn

Author: Monika Venckauskaite
Date: January 7th, 2022
'''

import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib as mlp
import matplotlib.pyplot as plt

mlp.use('Agg')
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame_churn: pandas dataframe
    '''
    # Read csv data file into pandas dataframe
    data_frame_churn = pd.read_csv(pth)
    return data_frame_churn


def plot_column_variable(data_frame_churn, column_variable):
    '''
    input:
    '''
    # Plot and save variable distributions
    plt.figure(figsize=(20, 10))
    data_frame_churn[column_variable].hist()
    plt.savefig("./images/eda/"+column_variable+".png")
    return 'Success'


def plot_heatmap(data_frame_churn):
    '''
    input:
    '''
    # Plot and save heatmap for data variable correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        data_frame_churn.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig("./images/eda/Heatmap_data_frame_corr.png")
    return 'Success'

def plot_bivariate(data_frame_churn, column_variable_1, column_variable_2):
    '''
    input:
    '''
    plt.figure(figsize=(20, 10))
    plt.scatter(data_frame_churn[column_variable_1], data_frame_churn[column_variable_2])
    plt.savefig("./images/eda/Bivariate_"+column_variable_1+"_and_"+ column_variable_2 + ".png")
    return 'Success'

def perform_eda(data_frame_churn):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            status: string, return if successfull
    '''
    # Encode churn column
    data_frame_churn['Churn'] = data_frame_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    data_frame_churn.to_csv('./data/bank_data_churn.csv')

    for column_variable in ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']:
        plot_column_variable(data_frame_churn, column_variable)

    plot_heatmap(data_frame_churn)
    plot_bivariate(data_frame_churn, 'Churn', 'Customer_Age')
    return data_frame_churn

def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
    be used for naming variables or index y column]
    output:
            data_frame: pandas dataframe with new columns for
    '''
    # Calculate churn by categorical features and add results to the dataframe
    for category_name in category_lst:
        category_churn_name = category_name + response
        category = []
        groups = data_frame.groupby(category_name).mean()['Churn']

        for val in data_frame[category_name]:
            category.append(groups.loc[val])

        data_frame[category_churn_name] = category

    return data_frame


def perform_feature_engineering(data_frame, flag):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument
    that could be used for naming variables or index y column]
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Initialize X and y datasets
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    data_frame_churn_y = data_frame['Churn']
    data_frame_churn_x = pd.DataFrame()
    # Define columns that will be encoded
    columns_to_encode = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    # Encode churn by feature columns
    data_frame = encoder_helper(data_frame, columns_to_encode, flag)
    # Define columns that will be kept
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    data_frame_churn_x[keep_cols] = data_frame[keep_cols]
    # Split the dataset into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        data_frame_churn_x, data_frame_churn_y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_rf,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             status: strin, return if successfull
    '''
    # Create classification report and save it
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/Classification_report.png')
    return 'Success'


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            status: string, return if successfull
    '''
    # Calculate and plot shapley values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    return 'Success'


def plot_roc_curves_lr(x_test, y_test, model_path, plot_path):
    '''
    inputs:
    outputs:
    '''
    # Plot the Linear Logistic Regression ROC curve and save it

    lr_model = joblib.load(model_path)

    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig(plot_path)
    return 'Success'


def plot_roc_curves_rf(x_test, y_test, model_path, plot_path):
    '''
    inputs:
    outputs:
    '''
    # Load the saved models
    cv_rfc = joblib.load(model_path)
    # Plots
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    # Plot the Random Forest Classifier ROC curve and save it
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=a_x,
        alpha=0.8)
    plt.savefig(plot_path)
    return 'Success'


def plot_classification_report_lr(
        y_test,
        y_train,
        y_test_preds_lr,
        y_train_preds_lr):
    '''
    inputs:
    outputs:
    '''
    # Logistic regression results
    print(classification_report(y_test, y_test_preds_lr))
    print(classification_report(y_train, y_train_preds_lr))
    return 'Success'



def plot_classification_report_rf(
        y_test,
        y_train,
        y_test_preds_rf,
        y_train_preds_rf):
    '''
    inputs:
    outputs:
    '''
    # Calculate and print result scores for both classifiers
    # Random Forest Results
    print(classification_report(y_test, y_test_preds_rf))
    print(classification_report(y_train, y_train_preds_rf))
    # Save classification report image
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf)
    return 'Success'

def plot_train_models_lr(
    y_test,
    y_train,
    y_test_preds_lr,
    y_train_preds_lr,
    x_test):
    '''
    inputs:
    outputs:
    '''
    plot_classification_report_lr(
        y_test,
        y_train,
        y_test_preds_lr,
        y_train_preds_lr)
    plot_roc_curves_lr(x_test, y_test, './models/lr_model.pkl', './images/results/ROC_curve_lrc.png')
    return 'Success'


def plot_train_models_rf(
        y_test,
        y_test_preds_rf,
        y_train_preds_rf,
        y_train,
        x_test):
    '''
    plots all the needed plots for model training results
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
              y_test, y_test_preds_rf, y_train_preds_rf, y_train, y_test_preds_lr,
              y_train_preds_lr, lrc, x_test, cv_rfc
    output:
              status: string, return if successfull
    '''
    plot_classification_report_rf(
        y_test,
        y_train,
        y_test_preds_rf,
        y_train_preds_rf)
    plot_roc_curves_rf(x_test, y_test, './models/rfc_model.pkl', './images/results/ROC_curve_rfc.png')
    return 'Success'


def train_random_forests(x_train, x_test, y_train, model_path):
    '''
    inputs:
    outputs:
    '''
    # Create Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)
    # Set parameters for grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    # Perform Grid Search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Fit the Random Forest Classifier
    cv_rfc.fit(x_train, y_train)
    # Perform test predictions with Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    joblib.dump(cv_rfc.best_estimator_, model_path)
    return y_test_preds_rf, y_train_preds_rf


def train_logistic_regression(x_train, x_test, y_train, model_path):
    '''
    inputs:
    outputs:
    '''
    # Create Logistic Regression Classifier
    lrc = LogisticRegression()
    # Fit the Logistic Linear Regression Classifier
    lrc.fit(x_train, y_train)
    # Perform Test predictions with Linear Regression Classifier
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    joblib.dump(lrc, model_path)
    return y_test_preds_lr, y_train_preds_lr


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              status: string, return if successfull
    '''
    y_test_preds_rf, y_train_preds_rf = train_random_forests(
        x_train, x_test, y_train, './models/rfc_model.pkl')
    y_test_preds_lr, y_train_preds_lr = train_logistic_regression(
        x_train, x_test, y_train, './models/lr_model.pkl')

    plot_train_models_lr(
        y_test,
        y_train,
        y_test_preds_lr,
        y_train_preds_lr,
        x_test)
    plot_train_models_rf(
        y_test,
        y_test_preds_rf,
        y_train_preds_rf,
        y_train,
        x_test)
    return 'Success'

if __name__ == '__main__':
    print('Runing main')
    data_frame_churn = import_data(r"./data/bank_data.csv")
    data_frame_churn = perform_eda(data_frame_churn)
    x_train, x_test, y_train, y_test = perform_feature_engineering(data_frame_churn, '_Churn')
    print(x_train)
    x_train.to_csv('./data/x_train.csv')
    x_test.to_csv('./data/x_test.csv')
    y_train.to_csv('./data/y_train.csv')
    y_test.to_csv('./data/y_test.csv')
        # feature_importance_plot(
        # cv_rfc.best_estimator_,
        # x_test,
        # './images/results/rfc_feature_importance.png')
    print(type(x_train))