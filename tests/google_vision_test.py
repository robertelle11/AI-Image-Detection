import base64
import requests
import piexif
import piexif.helper
from PIL import Image
from io import BytesIO

# Google Cloud API Key
API_KEY = "Place api key here"

def detect_landmark(image_path):
    """Detects landmarks in an image using Google Vision API."""
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    request_payload = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "LANDMARK_DETECTION", "maxResults": 5}],  # Increase results for better accuracy
            }
        ]
    }

    response = requests.post(url, json=request_payload)

    if response.status_code == 200:
        landmarks = response.json().get("responses", [])[0].get("landmarkAnnotations", [])

        if landmarks:
            best_landmark = landmarks[0]  # Pick the most confident detection
            name = best_landmark["description"]
            lat, lng = best_landmark["locations"][0]["latLng"]["latitude"], best_landmark["locations"][0]["latLng"]["longitude"]
            
            location = get_location_from_coordinates(lat, lng)

            print(f"Detected Landmark: {name}")
            print(f"Coordinates: {lat}, {lng}")
            print(f"Real-World Location: {location}")
            return name, lat, lng, location
        else:
            print("No landmarks detected. Checking EXIF metadata...")

    return detect_location_from_metadata(image_path)

def get_location_from_coordinates(lat, lng):
    """Uses OpenStreetMap's Nominatim API to get the real-world location from coordinates."""
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
    
    response = requests.get(url, headers={"User-Agent": "GeoLocator/1.0"})
    
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", "Unknown Location")
    
    print("Nominatim failed. Trying Google Geocoding API...")
    return get_location_from_google(lat, lng)

def get_location_from_google(lat, lng):
    """Uses Google Geocoding API for better accuracy."""
    google_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={API_KEY}"
    
    response = requests.get(google_url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"][0]["formatted_address"]
    
    return "Location not found"

def detect_location_from_metadata(image_path):
    """Extracts GPS metadata from image EXIF data to determine location."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data and 34853 in exif_data:
            gps_info = exif_data[34853]

            if gps_info and 2 in gps_info and 4 in gps_info:  # Latitude & Longitude exist
                lat = convert_to_degrees(gps_info[2])
                lng = convert_to_degrees(gps_info[4])
                
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
    """Converts EXIF GPS coordinates to decimal degrees."""
    try:
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)
    except:
        return None

# Test with an image
image_path = "train/0/0/2/002a636eb80ee968.jpg"
detect_landmark(image_path)
