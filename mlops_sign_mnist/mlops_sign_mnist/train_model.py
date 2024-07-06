import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.model import SignLanguageMNISTModel
import hydra
from datetime import datetime # used to name models to keep track of them
import wandb
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# load tensors
labels_train: torch.Tensor = torch.load("data/processed/labels_train.pt")
X_train: torch.Tensor = torch.load("data/processed/X_train.pt")




"""
@click.group()
def cli():
    Command line interface.
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=50, help="number of epochs to train for")

"""

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg):
    lr : float = cfg.hyperparameters.learning_rate
    batch_size : int = cfg.hyperparameters.batch_size
    epochs : int = cfg.hyperparameters.epochs
    wandb.init(project="sign_language_mnist",name="mlops_train_loss", config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs})
    print("Training Sign Language MNIST model")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    train_dataset = TensorDataset(X_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SignLanguageMNISTModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    # val_losses = []
    # Initialize the profiler
    # with profile(
    #     activities=[ProfilerActivity.CPU], # ProfilerActivity.CUDA
    #     record_shapes=True,
    #     profile_memory=True,
    #     on_trace_ready=tensorboard_trace_handler("./log/sign_language_mnist")) as prof:
    #     print("Profiler started")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # prof.step()  # Mark the end of an iteration
        print("Iteration logged")
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), f"models/sign_language_mnist_model_{current_time}.pth") # change this
        print("Model saved")
        # wandb.save("models/sign_language_mnist_model.pth")

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        # plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("reports/figures/sign_language_training_loss123.png")
        print("Loss plot saved")
        wandb.log({"training_loss_plot": wandb.Image("reports/figures/sign_language_training_loss.png")})

    wandb.finish()
# Print profiling results
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    
    # Export profiling results for visualization
    # prof.export_chrome_trace("trace.json")
    print("Profiling complete")



if __name__ == "__main__":
    main()
