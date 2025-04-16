import streamlit as st
import pandas as pd
import pydeck as pdk
import base64
import os

st.set_page_config(layout="wide")
st.title("📍 Geotagged Images")

distinct_colors = [
    [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
    [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255],
    [255, 165, 0, 255], [128, 0, 128, 255], [0, 128, 128, 255],
    [255, 255, 255, 255]
]

uploaded_csvs = st.file_uploader(
    "📄 Upload one or more CSV files (columns: `file_path`, `lat`, `long`)",
    type='csv',
    accept_multiple_files=True
)

uploaded_images = st.file_uploader(
    "🖼️ Upload images (select all in folder)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

# Dictionary based on numeric filename match
image_dict = {}
if uploaded_images:
    for img in uploaded_images:
        filename = os.path.basename(img.name).lower()
        base_id = os.path.splitext(filename)[0]  # e.g. "1001.jpg" → "1001"
        extension = filename.split('.')[-1].lower()
        mime = f"image/{'jpeg' if extension in ['jpg', 'jpeg'] else extension}"
        encoded = base64.b64encode(img.read()).decode()
        image_dict[base_id] = f"data:{mime};base64,{encoded}"

if uploaded_csvs:
    layers = []
    latitudes = []
    longitudes = []
    color_index = 0
    total_points = 0

    for csv in uploaded_csvs:
        try:
            df = pd.read_csv(csv)

            if {'file_path', 'lat', 'long'}.issubset(df.columns):
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
                df['long'] = pd.to_numeric(df['long'], errors='coerce')
                df = df.dropna(subset=['lat', 'long'])

                if not df.empty:
                    # Normalize file_path and match to image_dict by numeric ID
                    df['file_path'] = df['file_path'].astype(str)
                    df['image_url'] = df['file_path'].apply(lambda x: image_dict.get(x.strip(), ""))

                    # Add display flags for conditional rendering
                    df['img_display'] = df['image_url'].apply(lambda x: 'block' if x else 'none')
                    df['text_display'] = df['image_url'].apply(lambda x: 'none' if x else 'block')

                    # Show warning if any images not found
                    missing = df[df['image_url'] == ""]
                    if not missing.empty:
                        st.warning(f"⚠️ {len(missing)} images missing for `{csv.name}`")
                        st.dataframe(missing[['file_path']])

                    # Assign distinct color
                    color = distinct_colors[color_index % len(distinct_colors)]
                    color_index += 1

                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=df,
                        get_position='[long, lat]',
                        get_color=color,
                        get_radius=800,
                        pickable=True
                    )
                    layers.append(layer)

                    latitudes.extend(df['lat'].tolist())
                    longitudes.extend(df['long'].tolist())
                    total_points += len(df)

                    st.success(f"✅ Loaded {csv.name} ({len(df)} points)")

                else:
                    st.warning(f"⚠️ No valid coordinates in {csv.name}")
            else:
                st.error(f"❌ {csv.name} is missing required columns.")
        except Exception as e:
            st.error(f"❌ Error loading {csv.name}: {e}")

    if layers and latitudes and longitudes:
        avg_lat = sum(latitudes) / len(latitudes)
        avg_long = sum(longitudes) / len(longitudes)

        view_state = pdk.ViewState(
            latitude=avg_lat,
            longitude=avg_long,
            zoom=6,
            pitch=45
        )

        tooltip = {
            "html": """
                <div style="width: 220px">
                    <b>{file_path}</b><br/>
                    <img src="{image_url}" style="width: 100%; border-radius: 4px; display: {img_display};" />
                    <i style="display: {text_display};">No image found</i>
                </div>
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black",
                "fontSize": "12px",
                "padding": "10px"
            }
        }

        st.write(f"🧭 Total points plotted: `{total_points}`")
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/streets-v11',
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip,
            height=850,
        ))

else:
    st.info("📂 Upload CSV files and matching images to display points.")
