# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project predicts the customer churn based on customer data.


## Running Files
In order to run the project, make sure to install the required dependencies:

```pip install -r requirements.txt```

Then, you can run the analysis with:

```python churn_script_logging_and_tests.py```

Script will use churn_library.py and perform analysis. EDA plots will be saved in images/eda. Result plots will appear in images/results. Then, you can find saved best models in models folder.

Script can be tested by running:

```pytest churn_script_logging_and_tests.py```

To test functions one by one, run the following:

```pytest churn_script_logging_and_tests.py::test_import ```

For script logs take a look at logs/churn_library.log
