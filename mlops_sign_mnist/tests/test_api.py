import torch
from fastapi.testclient import TestClient

from API import api

ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

client = TestClient(api.app)


def test_predict_success():
    # Simulate a valid input tensor image
    valid_image = {"image": torch.rand(1, 28, 28).tolist()}

    response = client.post("/predict", json=valid_image)
    assert response.status_code == 200
    print("THE API GIVES BACK")
    print(response.json()["pred"])
    assert response.json()["pred"] in ALPHABET


def test_predict_invalid_format():
    # Simulate an invalid input tensor image (e.g., string instead of list)
    invalid_image = {"image": "invalid_string"}

    response = client.post("/predict", json=invalid_image)
    assert response.status_code == 422  # Unprocessable Entity


if __name__ == "__main__":
    test_predict_success()
    test_predict_invalid_format()
