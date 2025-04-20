import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize after cropping
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize to ImageNet's mean and std
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        cluster_label = self.data.iloc[idx, 3]

        # Convert to tensor for classification
        cluster_target = torch.tensor(cluster_label, dtype=torch.long)

        return img, cluster_target
    
class ClusterClassifier(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.resnet = resnet18(weights='IMAGENET1K_V1')

        # Unfreeze layer4
        for name, param in self.resnet.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True

        # Classifier head 
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_clusters),
        )

    def forward(self, x):
        cluster_logits = self.resnet(x)

        return cluster_logits
    
def main():
    csv_path = r"D:\4999 Data\cluster_balanced.csv"
    df = pd.read_csv(csv_path)
    all_labels = df['cluster'].tolist()
    indices = list(range(len(all_labels)))
    dataset = GeoDataset(csv_path)

    batch_size = 50
    num_epochs = 15

    # Stratified split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=all_labels,
        random_state=10
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]

    print("Train label counts:", Counter(train_labels))
    print("Val label counts:", Counter(val_labels))

    # Dataloaders
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
    logger.info(f"Number of batches in train_dataloader: {len(train_dataloader)}")
    logger.info(f"Number of batches in val_dataloader: {len(val_dataloader)}")

    # Try to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClusterClassifier(num_clusters=21).to(device)
    logger.info(f"Using device: {device}")  # Should print 'cuda:0' if model is on GPU

    # Load the cluster labels from the CSV file
    train_indicies = train_subset.indices
    train_cluster_labels = dataset.data.iloc[train_indicies, 3].values

    # Calculate the class weights based on the distribution of cluster labels
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_cluster_labels), 
        y=train_cluster_labels
    )

    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    param_group = [
    {'params': model.resnet.layer4.parameters(), 'lr': 1e-5},
    {'params': model.resnet.fc.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ]
    # Optimizer
    optimizer = torch.optim.AdamW(param_group)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()
        correct_train = 0
        total_train = 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started")

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass through the model
            cluster_probs = model(images)

            # Cluster classification loss (CrossEntropy for cluster prediction)
            loss = criterion(cluster_probs, targets)

            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            epoch_train_loss += loss.item()

             # Compute training accuracy
            predicted = torch.argmax(cluster_probs, dim=1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)
    
        # Store training loss for this epoch
        train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Time: {time.time() - start_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validate the model after each epoch
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_targets = []
        # Disable gradient calculation for validation
        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

                # Compute validation accuracy
                predicted = torch.argmax(outputs, dim=1)
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

                # Collect for classification report
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Store validation loss for this epoch
        val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Generate classification report
        class_report = classification_report(
            all_targets, 
            all_preds,
            target_names=[str(i) for i in range(21)],  
            zero_division=0
        )

        logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}")
        logger.info(f"\nEpoch {epoch+1} Classification Report:\n{class_report}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_cluster_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete!")

    # Save weights
    weight_path = "cluster_weights.pth"
    torch.save(model.state_dict(), weight_path)
    logger.info(f"Weights saved as {weight_path}")

    # Save model
    model_path = "cluster_model.pth"
    torch.save(model, model_path)
    logger.info(f"Model saved as {model_path}")

    os.makedirs("plots", exist_ok=True)
    # Plotting the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss", color='blue')
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss", color='orange')
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plot_path = "plots/cluster_loss_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training/validation loss plot to {plot_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy", color='green')
    plt.plot(range(len(val_accuracies)), val_accuracies, label="Validation Accuracy", color='red')
    plt.title("Training and Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    acc_plot_path = "plots/cluster_accuracy_curve.png"
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training/validation accuracy plot to {acc_plot_path}")

     # Run validation one final time after training
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Confusion matrix (after final epoch)
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(i) for i in range(21)],
                yticklabels=[str(i) for i in range(21)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Validation Set - Final Epoch)")
    plt.tight_layout()

    cm_path = "plots/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")



if __name__ == "__main__":
    main()