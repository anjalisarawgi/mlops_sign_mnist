import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mlops_sign_mnist')))

import torch
from mlops_sign_mnist.models.model import SignLanguageMNISTModel

def test_model_forward():
    model = SignLanguageMNISTModel()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 25)  # Assuming the model outputs 25 classes