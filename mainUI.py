try:
    import streamlit as st
    from PIL import Image
    import gps_plot as predict_location
    from tests import gsv_test as analyze_landmark
    from landmarks_dataset import map_landmark_id
except ImportError as e:
    st.error(f"An error occurred while importing modules: {e}")

st.set_page_config(page_title="AI Image Geolocation", layout="centered")
st.title("🌍 AI-Powered Image Geolocation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            lat, lon, confidence = predict_location(image)
            landmark_result = analyze_landmark(image)

        st.subheader("📍 Predicted Location")
        st.markdown(f"**Latitude:** {lat:.4f}, **Longitude:** {lon:.4f}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        if landmark_result:
            name, l_lat, l_lon = map_landmark_id(landmark_result)
            st.subheader("🗿 Landmark Detected")
            st.markdown(f"**Landmark:** {name}")
            st.markdown(f"**Coordinates:** {l_lat}, {l_lon}")

        st.map({"lat": [lat], "lon": [lon]})
