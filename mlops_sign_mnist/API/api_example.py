import requests
import json
import torch
import numpy as np


# Example tensor data (list of lists of floats)
torch_tensor = torch.rand(1, 1, 28, 28)

# Convert PyTorch tensor to nested Python list (serializable format)
tensor_list = torch_tensor.tolist()
print(type(tensor_list))

# URL of your FastAPI endpoint
url = 'http://localhost:8000/predict'

json_data = {"data": tensor_list}

data={"image": tensor_list}

print(type(json_data))

# Send POST request with JSON data
response = requests.post(url, json=data)

# Print the response
print(response.json())


