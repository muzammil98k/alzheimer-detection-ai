import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from utils.mri_dataset import MRIDataset
from models.cnn3d import CNN3D


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------
dataset = MRIDataset(
    data_dir="data/processed",
    label_file="data/labels.csv",
    augment=True
)

print("Total dataset size:", len(dataset))


# --------------------------------------------------
# Save directory
# --------------------------------------------------
os.makedirs("models_saved", exist_ok=True)


# --------------------------------------------------
# Cross Validation Setup
# --------------------------------------------------
k_folds = 5
epochs = 25

kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []


for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

    print("\n============================")
    print(f"FOLD {fold+1}")
    print("============================")

    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)

    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=2)

    model = CNN3D().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    best_accuracy = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---------------- Validation ----------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                preds = torch.argmax(outputs, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_accuracy = correct / total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # -------- Save best model for this fold --------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                f"models_saved/cnn3d_fold{fold+1}.pth"
            )

            print("Best model for this fold saved!")

    print(f"\nBest accuracy for fold {fold+1}: {best_accuracy:.4f}")

    fold_results.append(best_accuracy)


# --------------------------------------------------
# Final Results
# --------------------------------------------------

print("\n===================================")
print("Cross Validation Results")
print("===================================")


for i, acc in enumerate(fold_results):

    print(f"Fold {i+1}: {acc:.4f}")


print("\nAverage Accuracy:", np.mean(fold_results))
