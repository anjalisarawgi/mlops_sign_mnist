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

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load tensors
labels_train: torch.Tensor = torch.load("data/processed/labels_train.pt")
X_train: torch.Tensor = torch.load("data/processed/X_train.pt")


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
    prof.export_chrome_trace("trace.json")
    return loss.item()


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg):
    lr: float = cfg.hyperparameters.learning_rate
    batch_size: int = cfg.hyperparameters.batch_size
    epochs: int = cfg.hyperparameters.epochs
    wandb.init(
        project="sign_language_mnist",
        name="mlops_train_loss",
        config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs},
    )
    print("Training Sign Language MNIST model")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    train_dataset = TensorDataset(X_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SignLanguageMNISTModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

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
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), f"models/sign_language_mnist_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/figures/sign_language_training_loss.png")
    wandb.log({"training_loss_plot": wandb.Image("reports/figures/sign_language_training_loss.png")})

    wandb.finish()
    print("Training complete")


if __name__ == "__main__":
    main()