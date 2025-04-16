import streamlit as st
import pandas as pd
import pydeck as pdk
import base64
import os
import random

st.set_page_config(layout="wide")
st.title("🌍 AI-Powered Image Geolocation (Experimental)")

# Simulated AI prediction function (mock version)
def predict_location_from_image(img_file):
    # Placeholder: Replace with real model prediction logic
    return {
        "lat": random.uniform(45.0, 60.0),   # Example: Canada-ish
        "long": random.uniform(-140.0, -50.0)
    }

# Upload images
uploaded_images = st.file_uploader(
    "🖼️ Upload images to estimate their location using AI",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_images:
    ai_data = []

    for img_file in uploaded_images:
        filename = os.path.basename(img_file.name)
        extension = filename.split('.')[-1].lower()
        mime = f"image/{'jpeg' if extension in ['jpg', 'jpeg'] else extension}"
        encoded = base64.b64encode(img_file.read()).decode()
        image_url = f"data:{mime};base64,{encoded}"

        # Predict using mock AI
        prediction = predict_location_from_image(img_file)
        lat = prediction['lat']
        lon = prediction['long']

        ai_data.append({
            "file_path": filename,
            "lat": lat,
            "long": lon,
            "image_url": image_url
        })

    df = pd.DataFrame(ai_data)
    st.success(f"✅ Predicted locations for {len(df)} images using AI")

    view_state = pdk.ViewState(
        latitude=df['lat'].mean(),
        longitude=df['long'].mean(),
        zoom=4,
        pitch=45
    )

    tooltip = {
        "html": """
            <div style="width: 220px">
                <b>{file_path}</b><br/>
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
        get_color=[0, 200, 150, 200],
        get_radius=800,
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/streets-v11',
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip,
        height=800
    ))

else:
    st.info("📂 Upload images to estimate and plot locations using AI.")