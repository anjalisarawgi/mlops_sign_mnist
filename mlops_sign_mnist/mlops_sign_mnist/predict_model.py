import torch
from torch.utils.data import DataLoader, TensorDataset

from mlops_sign_mnist.models.model import SignLanguageMNISTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# load tensors
X_test: torch.Tensor = torch.load("data/processed/X_test.pt")
labels_test: torch.Tensor  = torch.load("data/processed/labels_test.pt")


def predict(model_checkpoint : str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = SignLanguageMNISTModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    test_dataset = TensorDataset(X_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy}")


if __name__ == "__main__":
    predict("models/sign_language_mnist_model.pth")
