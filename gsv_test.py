import pandas as pd
import requests

def load_dataset(file_path):
    """Load the dataset containing latitude and longitude."""
    df = pd.read_csv(file_path)
    return df

def get_image_location(image_id, df):
    """Retrieve latitude and longitude for the given image ID."""
    try:
        image_id = int(image_id)  # Ensure image ID is an integer
        if 0 <= image_id < len(df):
            lat, long = df.iloc[image_id]
            return lat, long
        else:
            return None, "Invalid image ID"
    except ValueError:
        return None, "Image ID must be a number"

def get_location_from_coordinates(lat, lng):
    """Uses OpenStreetMap's Nominatim API to get the real-world location from coordinates"""
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
    
    response = requests.get(url, headers={"User-Agent": "GeoLocator/1.0"})
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", "Unknown Location")
    return "Unknown Location"

def detect_location_from_metadata(image_path):
    """Extracts GPS metadata from image EXIF data to determine location."""
    try:
        image = image.open(image_path)
        exif_data = image._getexif()

        if exif_data is not None and 34853 in exif_data:
            gps_info = exif_data[34853]

            if gps_info and 2 in gps_info and 4 in gps_info:  # Latitude & Longitude exist
                lat = convert_to_degrees(gps_info[2])  # GPSLatitude
                lng = convert_to_degrees(gps_info[4])  # GPSLongitude
                
                # Check hemisphere (N/S, E/W)
                if gps_info[1] == "S":
                    lat = -lat
                if gps_info[3] == "W":
                    lng = -lng

                location = get_location_from_coordinates(lat, lng)
                print(f"GPS Metadata Found: {lat}, {lng}")
                print(f"Real-World Location (EXIF): {location}")
                return None, lat, lng, location

    except Exception as e:
        print("Error reading EXIF metadata:", e)
    
    return None, None, None, "Location could not be determined"

def convert_to_degrees(value):
    """Helper function to convert EXIF GPS coordinates to decimal degrees."""
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def main():
    file_path = "dataset/coords.csv"  # Update this path if needed
    df = load_dataset(file_path)
    
    image_id = 67  # Replace with the actual image ID
    lat, long = get_image_location(image_id, df)
    
    if lat is not None:
        location = get_location_from_coordinates(lat, long)
        print(f"Real-World Location: {location}")
        print(f"Coordinates: ({lat}, {long})")
    else:
        print(f"Error: {long}")  # Print error message

if __name__ == "__main__":
    main()
