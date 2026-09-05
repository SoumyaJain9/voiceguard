import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

# Support PyTorch AMP
from torch.cuda.amp import GradScaler, autocast

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str, scaler: GradScaler, accumulation_steps: int = 4) -> Tuple[float, float]:
    """Executes a single forward-backward training epoch with gradient accumulation and AMP."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training Epoch", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)

        # Automatic Mixed Precision (AMP)
        with autocast(enabled=(device == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        # Backward pass with gradient scaling
        if device == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            if device == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(1, len(dataloader))
    epoch_acc = 100.0 * correct / max(1, total)
    return epoch_loss, epoch_acc

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """Evaluates model performance on the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation Epoch", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(enabled=(device == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / max(1, len(dataloader))
    val_acc = 100.0 * correct / max(1, total)
    return val_loss, val_acc

def train_pipeline(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5, device: str = "cuda" if torch.cuda.is_available() else "cpu", checkpoint_dir: str = "checkpoints"):
    """
    Complete Training and Fine-tuning Pipeline for VoxGuard AI.
    Features:
    - AdamW optimizer with weight decay
    - Mixed Precision training (AMP)
    - Gradient accumulation for larger effective batch size
    - Checkpoint saving on best validation accuracy
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    # Balanced class weights
    weights = torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val_acc = 0.0
    accumulation_steps = 4

    print(f"🚀 Starting VoxGuard Training on Device: {device} | Epochs: {epochs}")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"🎯 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New Best Model Saved with Validation Accuracy: {best_val_acc:.2f}% -> {save_path}")

    print("\n✅ Training Pipeline Complete.")
