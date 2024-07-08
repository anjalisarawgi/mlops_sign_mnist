import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mlops_sign_mnist')))



# FILE HANDELING SECTION TO BE DELETED AND IS ONLY FOR DEBUGGING
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# List the contents of the current directory
print("Directory contents:")
for item in os.listdir(current_directory):
    print(item)


import torch
from mlops_sign_mnist.models.model import SignLanguageMNISTModel

def test_model_forward():
    model = SignLanguageMNISTModel()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 25)  # Assuming the model outputs 25 classes
