import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import json
import os
from tests import gld_test
from tests import google_vision_test
from tests import gsv_test
# from scripts import filter_images
from scripts import gps_plot
import train_model

st.set_page_config(page_title="Project Content Page", page_icon="🌍")

st.markdown("Project Content Page")
st.title("AI-Powered Image Geolocation")
st.sidebar.header("Project Content Page")
st.sidebar.success("Project Content Page")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)


progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run") # (make sure to add pages file in your root directiory)