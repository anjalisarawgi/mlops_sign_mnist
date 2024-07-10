from fastapi import FastAPI
import torch
import numpy as np
from mlops_sign_mnist.models.model import SignLanguageMNISTModel
from pydantic import BaseModel

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
app = FastAPI()

# load model
model_checkpoint = "models/sign_language_mnist_model.pth"
model = SignLanguageMNISTModel()
model.load_state_dict(torch.load(model_checkpoint))


# used for data validation for data coming into the endpoint
class tensor(BaseModel):
    image: list

# Endpoint that given tensor image returns the prediction

@app.post("/predict")
def predict(data: tensor): 
    image = torch.tensor(data.image)
    y_pred = model(image)
    correct = (y_pred.argmax(dim=1))
    prediction = ALPHABET[correct.item()]
    return {"pred": prediction}