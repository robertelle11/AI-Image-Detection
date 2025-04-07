import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os

# === Dataset 1 ===
csv_path_1 = "new_landmarks.csv"
image_folder_1 = "train/"
df1 = pd.read_csv(csv_path_1)

# === Dataset 2 ===
csv_path_2 = "new_ex_images.csv"
image_folder_2 = "Extra Images/"
df2 = pd.read_csv(csv_path_2)

# === Setup world map ===
plt.figure(figsize=(16, 8))
m = Basemap(projection="cyl", resolution="c")

m.drawcoastlines()
m.drawcountries()
m.drawparallels(range(-90, 91, 30), labels=[1, 0, 0, 0])
m.drawmeridians(range(-180, 181, 60), labels=[0, 0, 0, 1])

# === Plot Dataset 1 ===
for idx, row in df1.iterrows():
    image_name = row["image_name"]
    lat = row["latitude"]
    lon = row["longitude"]
    image_path = os.path.join(image_folder_1, image_name)
    
    print(f"[DF1 - {idx}] Processing {image_name} at ({lat}, {lon})")
    m.scatter(lon, lat, latlon=True, c="blue", s=60, edgecolors="black", zorder=5, label="Dataset 1" if idx == 0 else "")

# === Plot Dataset 2 ===
for idx, row in df2.iterrows():
    image_name = row["image_name"]
    lat = row["latitude"]
    lon = row["longitude"]
    image_path = os.path.join(image_folder_2, image_name)

    print(f"[DF2 - {idx}] Processing {image_name} at ({lat}, {lon})")
    m.scatter(lon, lat, latlon=True, c="red", s=60, edgecolors="black", zorder=5, label="Dataset 2" if idx == 0 else "")

# === Legend & Display ===
plt.legend(loc="lower left")
plt.title("🗺️ Image Locations: Dataset 1 (blue) vs Dataset 2 (red)")
plt.show()

