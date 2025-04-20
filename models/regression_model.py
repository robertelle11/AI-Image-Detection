import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from collections import Counter
from cluster_model import ClusterClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)

        self.data.sort_values(by='cluster', inplace=True)

        # Calculate mean and standard deviation of long/lat for z-score normalization
        self.lat_mean = self.data.iloc[:, 1].mean()
        self.lon_mean = self.data.iloc[:, 2].mean()
        self.lat_std = self.data.iloc[:, 1].std()
        self.lon_std = self.data.iloc[:, 2].std()

        # Calculate cluster means
        cluster_means = self.data.groupby('cluster')[['latitude', 'longitude']].mean().values
        self.cluster_means = torch.tensor(cluster_means, dtype=torch.float32)

        # Normalize cluster means
        self.cluster_means[:, 0] = (self.cluster_means[:, 0] - self.lat_mean) / self.lat_std
        self.cluster_means[:, 1] = (self.cluster_means[:, 1] - self.lon_mean) / self.lon_std

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

        lat, lon = self.data.iloc[idx, 1], self.data.iloc[idx, 2]

        # Convert to tensor for regression
        target = torch.tensor([lat, lon], dtype=torch.float32)

        return img, target
    
class HaversineLoss(nn.Module):
    def __init__(self, lat_mean, lon_mean, lat_std, lon_std):
        super(HaversineLoss, self).__init__()
        self.R = 6371 # Earth's radius in km
        self.lat_mean, self.lon_mean = lat_mean, lon_mean
        self.lat_std, self.lon_std = lat_std, lon_std

    def forward(self, preds, targets):
        # Convert to radians
        lat1, lon1 = torch.deg2rad(targets[:, 0]), torch.deg2rad(targets[:, 1])
        lat2, lon2 = torch.deg2rad(preds[:, 0]), torch.deg2rad(preds[:, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        return torch.mean(self.R * c)  # Mean distance loss in km

def compute_mae(preds, targets):
    # Computes mean absolute error for lat/lon (takes into account latitude wrap around)
    mae_lat = torch.mean(torch.abs(((preds[:, 0] - targets[:, 0] + 180) % 360) - 180)).item()
    mae_lon = torch.mean(torch.abs(preds[:, 1] - targets[:, 1])).item()
    return mae_lat, mae_lon

class GeoRegression(nn.Module):
    def __init__(self, num_clusters, cluster_means, lat_mean, lat_std, lon_mean, lon_std):
        super().__init__()

        # Feature extractor
        base = resnet18(weights='IMAGENET1K_V1')

        self.register_buffer("cluster_means", cluster_means.clone().detach())
        self.register_buffer("lat_mean", lat_mean.clone().detach())
        self.register_buffer("lat_std", lat_std.clone().detach())
        self.register_buffer("lon_mean", lon_mean.clone().detach())
        self.register_buffer("lon_std", lon_std.clone().detach())

        # Freeze all layers 
        for param in base.parameters():
            param.requires_grad = False

        # Unfreeze layer3 and layer4
        for name, param in base.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(base.children())[:-1]) # Remove fc layer
        self.feature_dim = base.fc.in_features

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(num_clusters, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, self.feature_dim),
            nn.Tanh()
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Output layer for lat/lon
        )

    def forward(self, image, cluster_probs):
        # Extract features from the image
        features = self.feature_extractor(image)
        features = features.view(features.size(0), -1)

        # Feature gating
        cluster_probs = torch.softmax(cluster_probs, dim=1)
        gate = self.gate(cluster_probs)
        gated_features = features * (1 + gate)

        delta = self.regression_head(gated_features)

        base_coords = cluster_probs @ self.cluster_means
        pred_coords = base_coords + delta

        # Denormalize the predictions
        preds_denorm = torch.zeros_like(pred_coords)
        preds_denorm[:, 0] = (pred_coords[:, 0] * self.lat_std) + self.lat_mean
        preds_denorm[:, 1] = (pred_coords[:, 1] * self.lon_std) + self.lon_mean

        return preds_denorm
    
# Function to save the model state along with additional information
def save_model_with_metadata(model, dataset, file_path):
    model_state_dict = model.state_dict()
    
    # Add parameters to the state dict
    model_state_dict['cluster_means'] = dataset.cluster_means
    model_state_dict['lat_mean'] = dataset.lat_mean
    model_state_dict['lat_std'] = dataset.lat_std
    model_state_dict['lon_mean'] = dataset.lon_mean
    model_state_dict['lon_std'] = dataset.lon_std

    # Save the model with the extra metadata
    torch.save(model_state_dict, file_path)

def main():
    csv_path = r"D:\4999 Data\cluster_balanced.csv"
    dataset = GeoDataset(csv_path)
    all_labels = dataset.data['cluster'].tolist()
    indices = list(range(len(dataset)))

    c_model = ClusterClassifier(num_clusters=21)
    c_model.load_state_dict(torch.load("cluster_weights.pth"))    
    c_model.eval()
    logger.info("Cluster model loaded")

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
    model = GeoRegression(
        num_clusters=21,
        cluster_means=dataset.cluster_means,
        lat_mean=torch.tensor(dataset.lat_mean, dtype=torch.float32),
        lat_std=torch.tensor(dataset.lat_std, dtype=torch.float32),
        lon_mean=torch.tensor(dataset.lon_mean, dtype=torch.float32),
        lon_std=torch.tensor(dataset.lon_std, dtype=torch.float32)
    ).to(device)
    c_model.to(device)
    logger.info(f"Using device: {device}")  # Should print 'cuda:0' if model is on GPU

    # Loss function
    criterion = HaversineLoss(dataset.lat_mean, dataset.lon_mean, dataset.lat_std, dataset.lon_std)

    param_group = [
        {'params': model.feature_extractor.parameters(), 'lr': 0.1e-5},
        {'params': model.gate.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.regression_head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ]
    # Optimizer
    optimizer = torch.optim.AdamW(param_group)

    train_losses = []
    val_losses = []

    train_mae_lat_list, train_mae_lon_list = [], []
    val_mae_lat_list, val_mae_lon_list = [], []

    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0

    # Sanity check: print the cluster means
    print("Cluster means shape:", dataset.cluster_means.shape)  # Expected shape: (num_clusters, 2)
    print("Cluster means:", dataset.cluster_means)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started")

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                cluster_probs = c_model(images)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass through the model
            pred_coords = model(images, cluster_probs)

            loss = criterion(pred_coords, targets)

            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            epoch_train_loss += loss.item()

            all_train_preds.append(pred_coords.detach().cpu())
            all_train_targets.append(targets.detach().cpu())

        # Store training loss for this epoch
        train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)

        # Compute metrics
        train_mae_lat, train_mae_lon = compute_mae(all_train_preds, all_train_targets)

        # Store the metrics
        train_mae_lat_list.append(train_mae_lat)
        train_mae_lon_list.append(train_mae_lon)

        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Time: {time.time() - start_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validate the model after each epoch
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        # Disable gradient calculation for validation
        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)

                cluster_probs = c_model(images)

                pred_coords = model(images, cluster_probs)

                loss = criterion(pred_coords, targets)
                epoch_val_loss += loss.item()

                all_val_preds.append(pred_coords.cpu())
                all_val_targets.append(targets.cpu())

        # Store validation loss for this epoch
        val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(val_loss)
        logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)

        # Compute metrics
        val_mae_lat, val_mae_lon = compute_mae(all_val_preds, all_val_targets)

        # Store the metrics
        val_mae_lat_list.append(val_mae_lat)
        val_mae_lon_list.append(val_mae_lon)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_with_metadata(model, dataset, "best_regressor_model_with_metadata.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete!")

    # Save model weights
    weight_path = "regressor_weights_with_metadata.pth"
    save_model_with_metadata(model, dataset, weight_path)
    logger.info(f"Weights saved as {weight_path}")

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

    plot_path = "plots/cnr_loss_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training/validation loss plot to {plot_path}")

    # Plot MAE in degrees
    plt.figure(figsize=(8, 6))
    plt.plot(train_mae_lat_list, label="Train MAE (Lat)", marker='o')
    plt.plot(val_mae_lat_list, label="Val MAE (Lat)", marker='s')
    plt.plot(train_mae_lon_list, label="Train MAE (Lon)", marker='o')
    plt.plot(val_mae_lon_list, label="Val MAE (Lon)", marker='s')

    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error (degrees)")
    plt.title("Mean Absolute Error Over Time (Denormalized)")
    plt.legend()
    plt.grid(True)

    plot_path = "plots/cnr_mae_degrees.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE plot to {plot_path}")

if __name__ == "__main__":
    main()