import streamlit as st
from PIL import Image
import base64
import pandas as pd
import pydeck as pdk
from geo_inference import predict_coordinates  # make sure geo_inference.py is in the same directory

# App config
st.set_page_config(layout="wide")
st.title("🧭 Image-based Geolocation Prediction (New Model)")

uploaded_images = st.file_uploader(
    "📷 Upload images for location prediction",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_images:
    predictions = []

    for img_file in uploaded_images:
        try:
            # Read file bytes first
            file_bytes = img_file.read()

            # Reload the file buffer for PIL after reading
            img = Image.open(img_file).convert("RGB")

            # Predict coordinates using new model
            lat, lon = predict_coordinates(img)

            # Validate and filter
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                st.warning(f"Prediction out of bounds for {img_file.name}. Latitude: {lat}, Longitude: {lon}")
                continue

            # Encode image for tooltip
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

        except Exception as e:
            st.warning(f"Prediction failed for {img_file.name}: {e}")

    if predictions:
        df = pd.DataFrame(predictions)

        view_state = pdk.ViewState(
            latitude=df["lat"].mean(),
            longitude=df["long"].mean(),
            zoom=3,
            pitch=0
        )

        tooltip = {
            "html": """
                <div style="width: 300px">
                    <b>{file_path}</b><br/>
                    <small>Lat: {lat}<br/>Lon: {long}</small><br/>
                    <img src="{image_url}" style="width: 100%; border-radius: 4px;" />
                </div>
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black",
                "fontSize": "13px",
                "padding": "12px"
            }
        }

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[long, lat]',
            get_color=[0, 128, 255, 160],  # Blue pins
            radiusMinPixels=5,
            radiusMaxPixels=50,
            pickable=True
        )

        st.success(f"✅ Plotted {len(df)} image(s) on the map")
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/streets-v11',
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
            height=850
        ))
    else:
        st.warning("⚠️ All predictions were out of bounds or failed.")
else:
    st.info("🖼️ Upload image(s) above to begin location predictions.")
