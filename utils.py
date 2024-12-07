import numpy as np
import pandas as pd
import joblib
import os
from pydantic import BaseModel, Field


# Load the pipeline & models
pipe = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'pipeline.pkl'))
model_forest = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'forest-tuned-class_weights.pkl'))


# Columns in order as user input
columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
          'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']


# Define desired valid dtypes
dtypes = {
    'CreditScore': float,
    'Geography': str,
    'Gender': str,
    'Age': int,
    'Tenure': int,
    'Balance': float,
    'NumOfProducts': int,
    'HasCrCard': int,
    'IsActiveMember': int,
    'EstimatedSalary': float
}


# Define the CustomerData model
class CustomerData(BaseModel):
    CreditScore: float = Field(..., description="Credit score of the customer")
    Geography: str = Field(..., description="Customer's country of residence")
    Gender: str = Field(..., description="Customer's gender")
    Age: int = Field(..., description="Customer's age")
    Tenure: int = Field(..., description="Number of years the customer has been with the bank")
    Balance: float = Field(..., description="Customer's account balance")
    NumOfProducts: int = Field(..., description="Number of products the customer has")
    HasCrCard: int = Field(..., description="1 if the customer has a credit card, otherwise 0")
    IsActiveMember: int = Field(..., description="1 if the customer is an active member, otherwise 0")
    EstimatedSalary: float = Field(..., description="Customer's estimated salary")
    


def predict_new(data: CustomerData) -> str:
    """ This function takes the user input as Pydantic and return the response
    """  

    # Concatenate all features form Pydantic
    input_data = np.array([data.CreditScore, data.Geography, data.Gender, data.Age,
                           data.Tenure, data.Balance, data.NumOfProducts, data.HasCrCard,
                           data.IsActiveMember, data.EstimatedSalary])
    
    # Adjust the column names and dtypes
    input_data = pd.DataFrame([input_data], columns=columns)
    X_new = input_data.astype(dtypes)

    # Apply Transformation
    X_processed = pipe.transform(X_new)

    # Prediction
    y_pred = model_forest.predict(X_processed)[0]

    return 'Exited' if y_pred==1 else 'Not Exited'