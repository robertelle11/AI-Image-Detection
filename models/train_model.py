import logging
import torch
import torch.nn as nn
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# Configuration
CSV_PATH = r"D:\4999 Data\cluster_balanced.csv"
BATCH_SIZE = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 10
TRAIN_RATIO = 0.8
NUM_WORKERS = 6
PLOTS_DIR = "plots"
WEIGHTS_PATH = "model_weights.pth"
MODEL_PATH = "geolocation_model.pth"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        
        # Calculate mean and standard deviation of long/lat for z-score normalization
        self.lat_mean = self.data.iloc[:, 1].mean()
        self.lon_mean = self.data.iloc[:, 2].mean()
        self.lat_std = self.data.iloc[:, 1].std()
        self.lon_std = self.data.iloc[:, 2].std()

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
        img = self.transform(img)

        lat, lon = self.data.iloc[idx, 1], self.data.iloc[idx, 2]

        # Z-score normalize
        lat = (lat - self.lat_mean) / self.lat_std
        lon = (lon - self.lon_mean) / self.lon_std

        target = torch.tensor([lat, lon], dtype=torch.float32) # Regression

        return img, target
    
class HaversineLoss(nn.Module):
    def __init__(self, lat_mean, lon_mean, lat_std, lon_std):
        super(HaversineLoss, self).__init__()
        self.R = 6371 # Earth's radius in km
        self.lat_mean, self.lon_mean = lat_mean, lon_mean
        self.lat_std, self.lon_std = lat_std, lon_std

    def forward(self, preds, targets):
        # Denormalize predictions and targets
        preds_denorm = torch.zeros_like(preds)
        preds_denorm[:, 0] = torch.clamp((preds[:, 0] * self.lat_std) + self.lat_mean, -90, 90)
        preds_denorm[:, 1] = torch.clamp((preds[:, 1] * self.lon_std) + self.lon_mean, -180, 180)

        targets_denorm = torch.zeros_like(targets)
        targets_denorm[:, 0] = torch.clamp((targets[:, 0] * self.lat_std) + self.lat_mean, -90, 90)
        targets_denorm[:, 1] = torch.clamp((targets[:, 1] * self.lon_std) + self.lon_mean, -180, 180)

        # Convert to radians
        lat1, lon1 = torch.deg2rad(targets_denorm[:, 0]), torch.deg2rad(targets_denorm[:, 1])
        lat2, lon2 = torch.deg2rad(preds_denorm[:, 0]), torch.deg2rad(preds_denorm[:, 1])

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

class GeoLocator(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(weights='IMAGENET1K_V1')

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze layer3 and layer4
        for name, param in self.resnet.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True

        # Replace the final classification layer with regression head
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [latitude, longitude]
        )

    def forward(self, x):
        return self.resnet(x)

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    dataset = GeoDataset(CSV_PATH)

    # Split dataset into training and validation subsets
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = int(len(dataset)) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)
    logger.info(f"Number of batches in train_dataloader: {len(train_dataloader)}")
    logger.info(f"Number of batches in val_dataloader: {len(val_dataloader)}")

    # Try to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoLocator().to(device)
    logger.info(f"Using device: {device}")  # Should print 'cuda' if model is on GPU

    # Loss function
    criterion = HaversineLoss(dataset.lat_mean, dataset.lon_mean, dataset.lat_std, dataset.lon_std)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses, val_losses = [], []
    train_mae_lat_list, train_mae_lon_list = [], []
    val_mae_lat_list, val_mae_lon_list = [], []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_targets = [], []
        start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} started")

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad() # Reset gradients

            outputs = model(images) # Forward pass
            loss = criterion(outputs, targets) # Compute loss

            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            epoch_train_loss += loss.item()

            all_train_preds.append(outputs.detach().cpu())
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

                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

                all_val_preds.append(outputs.cpu())
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

    logger.info("Training complete!")

    # Save weights
    torch.save(model.state_dict(), WEIGHTS_PATH)
    logger.info(f"Weights saved as {WEIGHTS_PATH}")

    # Save model
    torch.save(model, MODEL_PATH)
    logger.info(f"Model saved as {MODEL_PATH}")

    # Plotting the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(NUM_EPOCHS), train_losses, label="Training Loss", color='blue')
    plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss", color='orange')
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(PLOTS_DIR, "loss_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training/validation loss plot to {plot_path}")

    # Denormalize MAE values
    train_mae_lat_degrees = [mae * dataset.lat_std for mae in train_mae_lat_list]
    val_mae_lat_degrees = [mae * dataset.lat_std for mae in val_mae_lat_list]
    train_mae_lon_degrees = [mae * dataset.lon_std for mae in train_mae_lon_list]
    val_mae_lon_degrees = [mae * dataset.lon_std for mae in val_mae_lon_list]

    # Plot MAE in real degrees
    plt.figure(figsize=(8, 6))
    plt.plot(train_mae_lat_degrees, label="Train MAE (Lat)", marker='o')
    plt.plot(val_mae_lat_degrees, label="Val MAE (Lat)", marker='s')
    plt.plot(train_mae_lon_degrees, label="Train MAE (Lon)", marker='o')
    plt.plot(val_mae_lon_degrees, label="Val MAE (Lon)", marker='s')

    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error (degrees)")
    plt.title("Mean Absolute Error Over Time (Denormalized)")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(PLOTS_DIR, "mae_degrees.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE plot to {plot_path}")

if __name__ == "__main__":
    main()