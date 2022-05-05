FROM python:3.8-slim
WORKDIR churn_prediction
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN mkdir data images models logs
RUN mkdir images/eda images/results
COPY data/bank_data.csv ./data/bank_data.csv
COPY data/test_data.csv ./data/test_data.csv
COPY churn_library.py ./churn_library.py
COPY churn_script_logging_and_tests.py ./churn_script_logging_and_tests.py
RUN python churn_script_logging_and_tests.py
RUN pytest churn_script_logging_and_tests.py
RUN cat logs/churn_library.log