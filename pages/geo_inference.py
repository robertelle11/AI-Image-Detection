import torch
from cluster_model import ClusterClassifier
from regression_model import GeoRegression
from torchvision import transforms
from PIL import Image

# Config
REGRESSION_MODEL_PATH = "regressor_weights_with_metadata.pth"
CLASSIFIER_MODEL_PATH = "cluster_weights.pth"
NUM_CLUSTERS = 21

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img):
    return transform(img).unsqueeze(0).to(device)

# Load classifier
def load_classifier(path, num_clusters):
    model = ClusterClassifier(num_clusters=num_clusters).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model

# Load the regression model and its metadata
def load_regressor(path, num_clusters):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    cluster_means = checkpoint.pop('cluster_means')
    lat_mean = torch.tensor(checkpoint.pop('lat_mean'), dtype=torch.float32)
    lat_std  = torch.tensor(checkpoint.pop('lat_std'),  dtype=torch.float32)
    lon_mean = torch.tensor(checkpoint.pop('lon_mean'), dtype=torch.float32)
    lon_std  = torch.tensor(checkpoint.pop('lon_std'),  dtype=torch.float32)

    model = GeoRegression(
        num_clusters=num_clusters,
        cluster_means=cluster_means,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std
    ).to(device)

    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Load models only once
classifier_model = load_classifier(CLASSIFIER_MODEL_PATH, NUM_CLUSTERS)
regressor_model = load_regressor(REGRESSION_MODEL_PATH, NUM_CLUSTERS)

def predict_coordinates(image: Image.Image) -> tuple[float, float]:
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        class_logits = classifier_model(input_tensor)
        pred = regressor_model(input_tensor, class_logits)

    lat = pred[0, 0].item()
    lon = pred[0, 1].item()
    return lat, lon
