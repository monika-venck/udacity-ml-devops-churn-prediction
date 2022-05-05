"""
Predict Customer Churn

Author: Monika Venckauskaite
Date: January 7th, 2022
"""

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

mlp.use("Agg")
sns.set()


def import_data(pth):
    """
    Return dataframe for the csv found at pth

    Args:
    pth: str. A path to the csv.

    Returns:
    data_frame_churn: pandas.DataFrame.
    """
    # Read csv data file into pandas dataframe
    data_frame_churn = pd.read_csv(pth)
    return data_frame_churn


def plot_column_variable(data_frame_churn, column_variable, figure_path):
    """
    Plots a selected column and saves results into figure_path.

    Args:
    data_frame_churn: pandas.DataFrame. Data used for plots.
    column_variable: string. A variable to plot.
    figure_path: string. A path to save figures.
    """
    # plot variable distributions
    plt.figure(figsize=(20, 10))
    data_frame_churn[column_variable].hist()

    # save figures
    plt.savefig(f"{figure_path}{column_variable}.png")


def plot_heatmap(data_frame_churn, figure_path, heatmap_name):
    """
    Plot a heatmap for data variable correlations

    Args:
    data_frame_churn: pandas.DataFrame. Data used for heatmap.
    figure_path: string. A path to save figures.
    heatmap_name: string. A name for heatmap file.
    """
    # plot heatmap for data variable correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame_churn.corr(), annot=False,
                cmap="Dark2_r", linewidths=2)
    # save heatmap
    plt.savefig(f"{figure_path}{heatmap_name}.png")


def plot_bivariate(churn_data, column_variable_1, column_variable_2, figure_path):
    """
    Plot bivariate plot of two columns

    Args:
    churn_data: pandas.DataFrame.
    column_variable_1: string. Variable to plot.
    column_variable_2: string. Variable to plot.
    figure_path: string. A path to save figures.
    """
    # plot bivariate plot
    plt.figure(figsize=(20, 10))
    plt.scatter(churn_data[column_variable_1], churn_data[column_variable_2])
    # save bivariate plot
    plt.savefig(
        f"{figure_path}Bivariate_{column_variable_1}_and_{column_variable_2}.png"
    )


def perform_eda(churn_data, columns_to_plot, figure_path, heatmap_name):
    """
    Perform eda on churn_data and save figures to images folder

    Args:
    churn_data: pandas.DataFrame. A dataset to perform EDA on.

    Returns:
    churn_data: pandas.DataFrame. A dataset with encoded Churn column.
    """
    # encode churn column
    churn_data["Churn"] = churn_data["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # plot column variables
    for column_variable in columns_to_plot:
        plot_column_variable(churn_data, column_variable, figure_path)

    # plot heatmap
    plot_heatmap(churn_data, figure_path, heatmap_name)

    # plot bivariate plots
    for column_variable in columns_to_plot:
        column_1 = column_variable
        for column_2 in columns_to_plot:
            if column_1 != column_2:
                plot_bivariate(churn_data, column_1, column_2, figure_path)
    return churn_data


def encoder_helper(churn_data, category_list, flag="_Churn"):
    """
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Args:
    churn_data: pandas.DataFrame
    category_list: list of columns. Contains categorical features
    tag: string. [optional argument that could
    be used for naming variables or index y column]

    Returns:
    churn_data: pandas.DataFrame. Contains newly encoded columns
    """

    # calculate churn by categorical features and add results to the dataframe
    for category_name in category_list:
        category_churn_name = category_name + flag
        category = []
        groups = churn_data.groupby(category_name).mean()["Churn"]

        for val in churn_data[category_name]:
            category.append(groups.loc[val])

        churn_data[category_churn_name] = category

    return churn_data


def perform_feature_engineering(churn_data_conv, flag):
    """
    Performs feature engineering for a given dataset

    Args:
    churn_data_conv: pandas.DataFrame. A dataset to perform EDA on.
    data_frame: pandas dataframe
              response: string of response name [optional argument
    that could be used for naming variables or index y column]
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # initialize X and y datasets
    churn_y = churn_data_conv["Churn"]
    churn_x = pd.DataFrame()
    # define columns that will be encoded
    columns_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    # encode churn by feature columns
    churn_data_encoded = encoder_helper(
        churn_data_conv, columns_to_encode, flag)
    # define columns that will be kept
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    churn_x[keep_cols] = churn_data_encoded[keep_cols]
    # split the dataset into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        churn_x, churn_y, test_size=0.33, random_state=42
    )

    return x_train, x_test, y_train, y_test


def feature_importance_plot(x_test, model_path, figure_path):
    """
    Creates and stores the feature importances in pth

    Args:
    model: model object. Contains feature_importances_
    x_test: pandas.DataFrame. Contains test values
    figure_path: string. path to store the figure

    """
    # load model
    loaded_model = joblib.load(model_path)

    # calculate and plot shapley values
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")

    # calculate feature importances
    importances = loaded_model.feature_importances_

    # sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # rearrange feature names so they match the sorted feature importances
    names = [x_test.columns[i] for i in indices]

    # create plot
    plt.figure(figsize=(20, 10))

    # create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # add bars
    plt.bar(range(x_test.shape[1]), importances[indices])

    # add feature names as x-axis labels
    plt.xticks(range(x_test.shape[1]), names, rotation=90)
    plt.savefig(figure_path)


def plot_roc(x_test, y_test, model_path, figure_path):
    """
    Plot ROC curve for the best estimator

    Args:
    x_test: pandas.DataFrame. Test dataset x
    y_test: pandas.DataFrame. Test dataset y
    model_path: string. Saved model path
    figure_path string. Path to save the ROC curve figure
    """
    # load the saved model
    loaded_model = joblib.load(model_path)

    # plot ROC curve
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    plot_roc_curve(loaded_model, x_test, y_test, ax=a_x, alpha=0.8)

    # save ROC curve
    plt.savefig(figure_path)


def plot_classification_report(
    y_test, y_train, y_test_preds, y_train_preds, report_names
):
    """
    Creates a classification report for data sets and predictions

    Args:
    y_test: pandas.DataFrame. Test data y
    y_train: pandas.DataFrame. Train data y
    y_test_preds: pandas.DataFrame. Test data y predictions
    y_train_preds: pandas.DataFrame. Train data y predictions

    """

    # extract report and figure names from variable
    report_name, figure_path = report_names

    # plot a classification report for test predictions
    plt.figure(figsize=(10, 10))
    plt.rc("figure", figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(
        0.01, 1.25, str(report_name), {"fontsize": 10}, fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )

    # plot a classification report for train predictions
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01, 0.6, str(report_name), {"fontsize": 10}, fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(f"{figure_path}")


def train_random_forests(x_train, x_test, y_train, model_path):
    """
    Train Random Forests on provided train and test datasets

    Args:
    x_train: pandas.DataFrame. Dataset for training
    x_test: pandas.DataFrame. Dataset for testing
    y_train: pandas.DataFrame. Dataset for training
    model_path: string. Path to save model

    Returns:
    y_test_preds: pandas.DataFrame. Test dataset predictions by model
    y_train_preds: pandas.DataFrame. Train dataset predictions by model

    """
    # create Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)

    # set parameters for grid search
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    # perform Grid Search
    model_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # fit the Random Forest Classifier
    model_rfc.fit(x_train, y_train)

    # perform test predictions with Random Forest Classifier
    # using best estimator
    y_train_preds = model_rfc.best_estimator_.predict(x_train)
    y_test_preds = model_rfc.best_estimator_.predict(x_test)

    # save model file to path
    joblib.dump(model_rfc.best_estimator_, model_path)
    return y_test_preds, y_train_preds


def train_logistic_regression(x_train, x_test, y_train, model_path):
    """
    Train Logistic Regression on provided train and test datasets

    Args:
    x_train: pandas.DataFrame. Dataset for training
    x_test: pandas.DataFrame. Dataset for testing
    y_train: pandas.DataFrame. Dataset for training
    model_path: string. Path to save model

    Returns:
    y_test_preds: pandas.DataFrame. Test dataset predictions by model
    y_train_preds: pandas.DataFrame. Train dataset predictions by model

    """
    # create Logistic Regression Classifier
    model_lrc = LogisticRegression(max_iter=1000)

    # fit the Logistic Linear Regression Classifier
    model_lrc.fit(x_train, y_train)

    # perform Test predictions with Linear Regression Classifier
    y_train_preds = model_lrc.predict(x_train)
    y_test_preds = model_lrc.predict(x_test)

    # save model file to path
    joblib.dump(model_lrc, model_path)
    return y_test_preds, y_train_preds


if __name__ == "__main__":
    # run the entire workflow
    churn_data_global = import_data(r"./data/bank_data.csv")
    # perform EDA
    columns_to_plot_global = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
    ]
    churn_data_conv_global = perform_eda(
        churn_data_global,
        columns_to_plot_global,
        figure_path="./images/eda/",
        heatmap_name="Correlations",
    )
    # churn_data_conv.to_csv("./data/bank_data_churn_encoded.csv")
    x_train_global, x_test_global, y_train_global, y_test_global = perform_feature_engineering(
        churn_data_conv_global, "_Churn"
    )

    # calculate and print result scores for both classifiers

    # train a random forest classifier
    y_test_preds_rfc, y_train_preds_rfc = train_random_forests(
        x_train_global, x_test_global, y_train_global, model_path="./models/rfc_model.pkl"
    )

    # plot classification report for random forests classifier
    plot_classification_report(
        y_test_global,
        y_train_global,
        y_test_preds_rfc,
        y_train_preds_rfc,
        ["./images/results/RFC_classification.png", "RF Classifier"]
    )

    # plot ROC curve for random forests classifier
    plot_roc(
        x_test_global,
        y_test_global,
        model_path="./models/rfc_model.pkl",
        figure_path="./images/results/ROC_curve_rfc.png",
    )

    # plot feature imortance for random forests classifier
    feature_importance_plot(
        x_test_global,
        model_path="./models/rfc_model.pkl",
        figure_path="./images/results/Feature_importance_rfc.png",
    )

    # train logistic regression
    y_test_preds_lr, y_train_preds_lr = train_logistic_regression(
        x_train_global, x_test_global, y_train_global, model_path="./models/lr_model.pkl"
    )

    # plot classification report for logistic regression
    plot_classification_report(
        y_test_global,
        y_train_global,
        y_test_preds_lr,
        y_train_preds_lr,
        ["./images/results/LR_classification.png", "Logistic Regression"],
    )

    # plot ROC curve for logistic regression
    plot_roc(
        x_test_global,
        y_test_global,
        model_path="./models/lr_model.pkl",
        figure_path="./images/results/ROC_curve_lr.png",
    )
