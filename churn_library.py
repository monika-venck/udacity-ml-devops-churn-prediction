# library doc string
'''
Predict Customer Churn

Author: Monika Venckauskaite
Date: January 7th, 2022
'''

# import libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # Read csv data file into pandas dataframe
    df = pd.read_csv(pth)
    return df


def perform_eda(df_churn):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            status: string, return if successfull
    '''
    # Print info about the dataset in the dataframe
    print(df_churn.head())
    print(df_churn.shape)
    print(df_churn.isnull().sum())
    print(df_churn.describe())
    # Encode churn column
    df_churn['Churn'] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # Plot and save variable distributions
    plt.figure(figsize=(20, 10))
    df_churn['Churn'].hist()
    plt.savefig("./images/eda/Churn.png")
    plt.figure(figsize=(20, 10))
    df_churn['Customer_Age'].hist()
    plt.savefig("./images/eda/Customer_Age.png")
    plt.figure(figsize=(20, 10))
    df_churn.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/Marital_status.png")
#     plt.figure(figsize=(20, 10))
#     sns.distplot(df['Total_Trans_Ct'])
#     plt.savefig("./images/eda/Total_Trans_Ct.png")
    # Plot and save heatmap for data variable correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_churn.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/Heatmap_df_corr.png")
    plt.figure(figsize=(20, 10))
    plt.scatter(df_churn['Churn'], df_churn['Customer_Age'])
    plt.savefig("./images/eda/Bivariate_Churn_and_Customer_Age.png")
    return 'Done'

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could 
    be used for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for
    '''
    # Calculate churn by categorical features and add results to the dataframe
    for category_name in category_lst:
        category_churn_name = category_name + '_Churn'
        category = []
        groups = df.groupby(category_name).mean()['Churn']

        for val in df[category_name]:
            category.append(groups.loc[val])

        df[category_churn_name] = category
    print(response)
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument 
    that could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Initialize X and y datasets
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    y = df['Churn']
    X = pd.DataFrame()
    # Define columns that will be encoded
    columns_to_encode = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    # Encode churn by feature columns
    df = encoder_helper(df, columns_to_encode, 'response')
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

    X[keep_cols] = df[keep_cols]
    # Split the dataset into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(response)
    return X_train, X_test, y_train, y_test


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
    return 'Done'

def feature_importance_plot(model, X_data, output_pth):
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
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)    
    plt.savefig(output_pth)
    return 'Done'

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              status: string, return if successfull
    '''

    # Create Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)
    # Create Logistic Regression Classifier
    lrc = LogisticRegression()
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
    cv_rfc.fit(X_train, y_train)
    # Fit the Logistic Linear Regression Classifier
    lrc.fit(X_train, y_train)
    # Perform test predictions with Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # Perform Test predictions with Linear Regression Classifier
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Calculate and print result scores for both classifiers
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    # Save classification report image
    classification_report_image(y_train, y_test, y_train_preds_rf, y_test_preds_rf)
    # Plot the Linear Logistic Regression ROC curve and save it
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    # Plots
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=a_x,
        alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig("./images/results/Training_results.png")
    # Save the best models for both classifiers
    feature_importance_plot(cv_rfc, X_test, './images/results/rfc_feature_importance.png')
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    # Load the saved models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    plt.savefig("ROC_curve_lrc.png")
    # Plot the Random Forest Classifier ROC curve and save it 
    plot_roc_curve(rfc_model, X_test, y_test, ax=a_x, alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig("ROC_curve_rfc.png")
    return 'Done'