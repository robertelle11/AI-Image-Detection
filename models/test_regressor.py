import torch
from train_model import GeoLocator
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Load the model
model = GeoLocator()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Move model to device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TestImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['id']).convert('RGB')
        image = self.transform(image)
        lat = torch.tensor(row['latitude'], dtype=torch.float32)
        lon = torch.tensor(row['longitude'], dtype=torch.float32)
        cluster = torch.tensor(row['cluster'], dtype=torch.long)  # if needed
        return image, lat, lon, cluster

def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

test_csv_path = r"D:\4999 Data\test_clustered.csv"
df = pd.read_csv(test_csv_path)

test_dataset = TestImageDataset(df)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, lat_true, lon_true, cluster_true in test_loader:
        images = images.to(device)
        preds = model(images)
        all_preds.append(preds.cpu())
        all_labels.append(torch.stack([lat_true, lon_true], dim=1))

# Combine results
predictions = torch.cat(all_preds, dim=0).numpy()
ground_truths = torch.cat(all_labels, dim=0).numpy()

def denormalize_coords(normalized, lat_mean=30.192587, lat_std=27.278425, lon_mean=-1.689700, lon_std=74.681434):
    lat = lat_mean + normalized[:, 0] * lat_std
    lon = lon_mean + normalized[:, 1] * lon_std
    return np.stack([lat, lon], axis=1)

predictions = denormalize_coords(predictions)

def compute_distances(predictions, ground_truths):
    distances = []
    for pred, true in zip(predictions, ground_truths):
        lat_pred, lon_pred = pred
        lat_true, lon_true = true
        dist = haversine(lat_pred, lon_pred, lat_true, lon_true)
        distances.append(dist)
    return np.array(distances)

distances = compute_distances(predictions, ground_truths)

# Haversine Distance Stats
print("Haversine Distance Statistics (in km):")
print(f"Min:     {np.min(distances):.2f}")
print(f"25th %:  {np.percentile(distances, 25):.2f}")
print(f"Median:  {np.median(distances):.2f}")
print(f"75th %:  {np.percentile(distances, 75):.2f}")
print(f"90th %:  {np.percentile(distances, 90):.2f}")
print(f"95th %:  {np.percentile(distances, 95):.2f}")
print(f"Max:     {np.max(distances):.2f}")

# Print mean distance
print(f"Mean Haversine distance: {distances.mean():.2f} km")

bins = [0, 50, 200, 750, 1500, 3000, 8000, 16000, np.inf]
labels = [
    'Same City (≤50km)',
    'Same State/Province (50-200km)',
    'Same Country (200-750km)',
    'Neighboring Country (750-1500km)', 
    'Same Region (1500-3000km)',
    'Same Continent (3000-8000km)',
    'Same Hemisphere',
    'Opposite Hemisphere'
]
categories = pd.cut(distances, bins=bins, labels=labels, right=False)

category_counts = pd.Series(categories).value_counts(normalize=True).sort_index()
# Print results
print("\nGeolocation Accuracy by Distance Category:")
for label, proportion in category_counts.items():
    print(f"{label:<20}: {proportion*100:.2f}%")

# Plot the histogram of distances
plt.hist(distances, bins=30, edgecolor='black')
plt.xlabel('Haversine Distance (km)')
plt.ylabel('Frequency')
plt.title('Histogram of Haversine Distances')
plot_path = "plots/reg_only_test_histogram.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Assume predictions and ground_truths are numpy arrays of shape [N, 2]
lat_preds = predictions[:, 0]
lon_preds = predictions[:, 1]
lat_trues = ground_truths[:, 0]
lon_trues = ground_truths[:, 1]

def angular_error(pred, true):
    """Returns minimal absolute angular difference (0 to 180 degrees)"""
    return np.abs((pred - true + 180) % 360 - 180)

# Metrics
lat_errors = np.abs(lat_preds - lat_trues)
lon_errors = angular_error(lon_preds, lon_trues)

lat_mae = np.mean(lat_errors)
lon_mae = np.mean(lon_errors)

print(f"Latitude MAE: {lat_mae:.4f}°")
print(f"Longitude MAE: {lon_mae:.4f}°")

# MAE Plot
plt.figure(figsize=(5, 4))
plt.bar(['Latitude', 'Longitude'], [lat_mae, lon_mae], color='cornflowerblue')
plt.ylabel('MAE (degrees)')
plt.title('Mean Absolute Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plot_path = "plots/reg_only_test_mae.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Stats
def print_stats(name, arr):
    print(f"\n{name} Error Stats (in degrees):")
    print(f"Min:     {np.min(arr):.4f}")
    print(f"25th %:  {np.percentile(arr, 25):.4f}")
    print(f"Median:  {np.median(arr):.4f}")
    print(f"75th %:  {np.percentile(arr, 75):.4f}")
    print(f"90th %:  {np.percentile(arr, 90):.4f}")
    print(f"95th %:  {np.percentile(arr, 95):.4f}")
    print(f"Max:     {np.max(arr):.4f}")

print_stats("Latitude", lat_errors)
print_stats("Longitude", lon_errors)