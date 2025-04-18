import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import pydeck as pdk
import base64
import pandas as pd
from train_model import GeoClassifier

# App config
st.set_page_config(layout="wide")
st.title("🧭 Image-based Geolocation Prediction")

# Load trained model
@st.cache_resource
def load_model():
    model = GeoClassifier()
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = load_model()
device = next(model.parameters()).device

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_images = st.file_uploader(
    "📷 Upload images for location prediction",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_images:
    predictions = []
    for img_file in uploaded_images:
        file_bytes = img_file.read()  # Read once!
        img = Image.open(img_file).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
        lat = output[0, 0].item()
        lon = output[0, 1].item()

        # Encode the image for tooltip
        extension = img_file.name.split('.')[-1].lower()
        mime = f"image/{'jpeg' if extension in ['jpg', 'jpeg'] else extension}"
        encoded = base64.b64encode(file_bytes).decode()
        image_url = f"data:{mime};base64,{encoded}"

        predictions.append({
            "file_path": img_file.name,
            "lat": lat,
            "long": lon,
            "image_url": image_url
        })

    if predictions:
        df = pd.DataFrame(predictions)
        df = df[(df["lat"].between(-90, 90)) & (df["long"].between(-180, 180))]

        if df.empty:
            st.warning("⚠️ All predictions were out of bounds. Try retraining your model or adjusting input images.")
        else:
            st.success(f"✅ Processed {len(df)} image(s) with valid coordinates")

            view_state = pdk.ViewState(
                latitude=df["lat"].mean(),
                longitude=df["long"].mean(),
                zoom=4,
                pitch=45
            )

            tooltip = {
                "html": """
                    <div style="width: 220px">
                        <b>{file_path}</b><br/>
                        <small>Lat: {lat}<br/>Lon: {long}</small><br/>
                        <img src="{image_url}" style="width: 100%; border-radius: 4px;" />
                    </div>
                """,
                "style": {
                    "backgroundColor": "white",
                    "color": "black",
                    "fontSize": "12px",
                    "padding": "10px"
                }
            }

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[long, lat]',
                get_color=[0, 128, 255, 200],
                get_radius=800,
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/streets-v11',
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip,
                height=850
            ))
else:
    st.info("🖼️ Upload image(s) above to begin location predictions.")
