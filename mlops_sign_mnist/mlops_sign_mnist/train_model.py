import os
import click
import matplotlib.pyplot as plt
import torch
import pandas as pd
from models.model import SignLanguageMNISTModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# load tensors
labels_train = torch.load("data/processed/labels_train.pt")
X_train = torch.load("data/processed/X_train.pt")
@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=50, help="number of epochs to train for")
def train(lr, batch_size, epochs) -> None:
    print("Training Sign Language MNIST model")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    train_path = 'data/processed/sign_mnist_train.csv'
    test_path = 'data/processed/sign_mnist_test.csv'

    # file would start from here 
    train_dataset = TensorDataset(X_train, labels_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SignLanguageMNISTModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

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

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}')

    torch.save(model.state_dict(), "models/sign_language_mnist_model.pth")
    print("Model saved")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/figures/sign_language_training_loss.png")
    print("Loss plot saved")

if __name__ == "__main__":
    train()