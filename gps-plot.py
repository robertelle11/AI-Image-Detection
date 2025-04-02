import os
import exifread
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Function to convert EXIF GPS format to decimal degrees
def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

# Extract GPS metadata from image
def get_gps_info(image_path):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)

    if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
        lat = [float(x.num) / float(x.den) for x in tags["GPS GPSLatitude"].values]
        lon = [float(x.num) / float(x.den) for x in tags["GPS GPSLongitude"].values]
        lat, lon = convert_to_degrees(lat), convert_to_degrees(lon)

        # Handle North/South & East/West
        if tags["GPS GPSLatitudeRef"].values != "N":
            lat = -lat
        if tags["GPS GPSLongitudeRef"].values != "E":
            lon = -lon

        return lat, lon
    return None, None

# Function to plot image location on a world map
def plot_on_map(lat, lon, image_path):
    plt.figure(figsize=(12, 6))
    m = Basemap(projection="cyl", resolution="c")

    # Draw map details
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(range(-90, 91, 30), labels=[1, 0, 0, 0])
    m.drawmeridians(range(-180, 181, 60), labels=[0, 0, 0, 1])

    # Plot the image location
    x, y = lon, lat
    m.scatter(x, y, latlon=True, c="red", marker="o", edgecolors="black", s=100, label="Image Location")

    # Display the image as a thumbnail near the point
    img = Image.open(image_path)
    img.thumbnail((100, 100))
    plt.imshow(img, extent=(x - 5, x + 5, y - 5, y + 5), zorder=10)

    plt.title(f"📍 Image Location: ({lat:.6f}, {lon:.6f})")
    plt.legend()
    plt.show()

# Example usage
image_path = "your-image.jpg"  ######### Replace with the actual image path
latitude, longitude = get_gps_info(image_path)

if latitude and longitude:
    print(f"📍 Image Geolocation: {latitude}, {longitude}")
    plot_on_map(latitude, longitude, image_path)
else:
    print("❌ No GPS metadata found in image.")
