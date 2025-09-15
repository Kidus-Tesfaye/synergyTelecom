# Data analysis and Churn prediction for an imaginary telecommunication company. 

## *Introduction*
This project aims to predict customers who are likely to churn for a telecom company using Machine Learning. As it is much more expensive to gain new customers, it is important to identify churners and take proactive retention measures. <br>

In this project you will find, 
- Data preprocessing with encoding, scaling,       
- Data visualization to illustrate correlation with the target,
- Target prediction using different models, and finally
- SHAP visualization to show feature importance. 


## *Project Structure*

├── data/     &emsp;            # Raw and processed data <br>
├── code/       &emsp;     # Jupyter notebook for EDA & experiments <br>
├── README.md      &emsp;       # Project documentation <br>
└── requirements.txt   &emsp;   # Python dependencies <br>


## *Installation*

1. Clone this repository <br>

git clone https://github.com/Kidus-Tesfaye/synergyTelecom.git <br>
cd synergyTelecom <br>

2. Create your virtual environment <br>
Best to consult the [official documentation](https://docs.python.org/3/library/venv.html) <br>

3. Install dependencies <br>
After creating and activating your virtual environment run the command below. <br>
pip install -r requirements.txt


## *Dataset*

The dataset used for this project can be found at [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

*The processed data doesn't contain the same number of columns as the raw data due to OneHotEncoding. Consult [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).*


## *Feature Engineering*
- Numerical columns like tenure, MonthlyCharges and TotalCharges were scaled.
- Categorical columns were either mapped or encode.
- An additional column called OnlineServices, which aggregates online services.

## *Models*

- [Logistic Regression ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
Both with baseline parameters. 


## *Performance*

Here are the metrics for the Logistic Regression model which outperformed the XGBClassifier

|  | Precision | Recall | F1 score |
|:---:|:---:|:---:|:---:|
| 0 (Non-Churn) | 0.89 | 0.81 | 0.85 |
| 1 (Churn) | 0.55 | 0.70 | 0.62 | 

[roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) : 0.76


## *Explainability*

The SHAP violin summary plot was used to get a better understanding of feature importance and explain how the model has used features to drive its prediction. 


## *Author*

Kidus Tesfaye -- (kidustesfaye34343@gmail.com) <br>
GitHub -- Kidus-Tesfaye
