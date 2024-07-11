import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from models.model import SignLanguageMNISTModel
import os
import logging

# Create log directory if it doesn't exist
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'train_model.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tensors
labels_train: torch.Tensor = torch.load("data/processed/labels_train.pt")
X_train: torch.Tensor = torch.load("data/processed/X_train.pt")

labels_test = torch.load("data/processed/labels_test.pt")
X_test = torch.load("data/processed/X_test.pt")

def profile_single_batch(model, X_batch, y_batch, optimizer, criterion):
    with profile(
        activities=[ProfilerActivity.CPU],  # Add ProfilerActivity.CUDA if using GPU
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        prof.step()  # Mark the end of an iteration

    # Print profiling results for the batch
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("log/trace.json")
    return loss.item()

def evaluate_model(model, data_loader, criterion):
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    avg_val_loss = val_loss / len(data_loader)
    return accuracy, avg_val_loss

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg):
    lr: float = cfg.hyperparameters.learning_rate
    batch_size: int = cfg.hyperparameters.batch_size
    epochs: int = cfg.hyperparameters.epochs
    wandb.init(
        project="sign_language_mnist",
        name="mlops_sign_mnist",
        config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs},
    )
    print("Training Sign Language MNIST model")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    train_dataset = TensorDataset(X_train, labels_train)
    val_dataset = TensorDataset(X_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SignLanguageMNISTModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if i == 0:  # Profile the first batch iteration of each epoch
                loss = profile_single_batch(model, X_batch, y_batch, optimizer, criterion)
            else:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        test_accuracy, avg_test_loss = evaluate_model(model, test_loader, criterion)
        test_accuracies.append(test_accuracy)
        
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1, "test_accuracy": test_accuracy })
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    torch.save(model.state_dict(), f"models/sign_language_mnist_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = "reports/figures/sign_language_training_loss.png"
    plt.savefig(loss_plot_path)

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = "reports/figures/sign_language_validation_accuracy.png"
    plt.savefig(accuracy_plot_path)

    # Log the plots to wandb
    wandb.log({
        "training_loss_plot": wandb.Image(loss_plot_path),
        "train_accuracy_plot": wandb.Image(accuracy_plot_path)
    })

    wandb.finish()
    print("Training complete")


if __name__ == "__main__":
    main()