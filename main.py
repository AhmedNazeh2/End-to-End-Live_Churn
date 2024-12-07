from fastapi import FastAPI
from utils import predict_new
from pydantic import BaseModel, Field


# Initialize an app
app = FastAPI(title='Churn-Detection')


# Endpoint for healthy check
@app.get('/', tags=['General'])
async def home():
    return {'up & running'}



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
    


# Endpoint for Prediction
@app.post('/predict', tags=['Classfication'])
async def churn_detection(data: CustomerData):

    # Call the function from utils.py
    pred = predict_new(data=data)

    return {f'Prediction is: {pred}'}
