# UI For Project
import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import json
import os
from tests import gld_test
from tests import google_vision_test
from tests import gsv_test
from scripts import filter_images
from scripts import gps_plot
import train_model


st.set_page_config(page_title="AI Image Geolocation", layout="centered")
st.title("🌍 AI-Powered Image Geolocation")

st.sidebar.success("Home Page.")

st.markdown(
    """
    AI-Image-Detection develops an AI-powered system that determines the geographic location of an image using machine learning, computer vision, and external data sources. The system analyzes visual elements such as landmarks, terrain, and architectural styles to generate an estimated location. A trained machine learning model, combined with geolocated datasets and API integrations, enhances accuracy and confidence in predictions. This tool is designed to support researchers, journalists, and digital investigators in verifying image origins and conducting location-based analysis. By combining advanced AI techniques with geospatial data, our system provides a reliable solution for image geolocation in various investigative and analytical applications.
    AI-Image-Detection is a project that is CSI4999 - Senior Capstone Project for the School of Engineering and Computer Science at Oakland University in Rochester, MI, USA.
    The Contributors to this Project are: Cassie Allard, Alan Shen, Justin DePorter, Michael Berman, Robert Elle, Michael Szpotek, Philip Przekora, Matthew Robertson, and Noah Pontzer.

    README.md - This file contains the information about the project, how to set it up, and how to run it.

    
Excerpt From "REAMDME.md"
gps-plot.py = plots the coordinates from both the Landmarks & StreetView Datasets onto a map to give us an idea of where our images are located.

New Task - Dataset Team********* find images in the following areas - United States [preferably south like Louisiana area, & Yellowstone area, Canada, Middle East-[Saudi Arabia], China, North/Central Africa

upload the actual images to philip's google drive = https://drive.google.com/drive/u/1/folders/1KGRL6GmgpFKY3N46CDj6s2PPgypr7N1n

Any images added must be updated in the extra_images.csv in the following format - File_Path,Latitude,Longitude. Don't worry about the file path as once all the images we have are uploaded, they will be renamed and organized later. Once we have enough images in our new dataset [extra images], we'll plot that along with the other 2 datasets and the AI team should have enough to work with.

Here's a reference picture of what we need to update 1-plottedpoints-landmarks+streetview

Testing Setup

---Before trying either of these files 2 seperate folders need to be downloaded that I sent as a Google Drvie since the files are too big for github.

-train folder - dataset containing a lot of images to train our model

-dataset folder - contains 2 files that will be used in the gld_test, those files are train.csv, and train_label_to_category.csv. You can keep these in a folder, I didn't, just make sure the path is right when testing.

gld_test.py - This file tests the Google Landmarks Dataset, in order to run this file you need to download the dataset of images in the train folder, as well as download the following .csv files - train.csv, and train_label_to_category.csv.

Correct ouput should look like this

Image found at: train\0\0\2\002a196a7a4d4e48.jpg Landmark ID: 90864 Category: http://commons.wikimedia.org/wiki/Category:Basilica_of_the_Assumption_(Prague) Landmark Name: Category:Basilica_of_the_Assumption_(Prague) Coordinates not found

--So far it only accepts images within the dataset, but it's a good starting point.

google_vision_test.py - This file uses the Google vision api using an api key. I've tested this by creating my own api key on my google cloud account, but everyone should be able to use it in testing. Also need to make sure the training set is downloaded in order to use this as well.

Correct output should look like this

Detected Landmark: Frederiksborg Castle Coordinates: 55.93496520000001, 12.3012724 Real-World Location: Indre Slotsgård, Hillerødsholm, Hillerød, Hillerød Kommune, Region Hovedstaden, 3400, Danmark

In the future if we get both of these working for everyone, these resources can be combined to make a very useful dataset.
"""
)