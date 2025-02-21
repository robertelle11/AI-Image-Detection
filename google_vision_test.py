import requests
import base64

# Replace with your actual Google Cloud Vision API key
API_KEY = "AIzaSyBMcY-_vzB7WycgYPmxcBf8CZxGCTi92DM"

def detect_landmark(image_path):
    """Detects landmarks in an image and returns their names and locations"""

    # Encode image as base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # API Endpoint
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    # Request Payload
    request_payload = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "LANDMARK_DETECTION"}],
            }
        ]
    }

    # Send request to Google Vision API
    response = requests.post(url, json=request_payload)

    if response.status_code == 200:
        landmarks = response.json().get("responses", [])[0].get("landmarkAnnotations", [])

        if landmarks:
            for landmark in landmarks:
                name = landmark["description"]  # Landmark name
                lat, lng = landmark["locations"][0]["latLng"]["latitude"], landmark["locations"][0]["latLng"]["longitude"]
                print(f"Detected Landmark: {name}")
                print(f"Location: {lat}, {lng}")
                return name, lat, lng
        else:
            print("No landmarks detected.")
            return None
    else:
        print("Error:", response.json())
        return None

# Test with an image
image_path = "train/0/0/2/002a636eb80ee968.jpg"
#image_path = "images/0/000c9db397ac1e7f.jpg"  # Replace with your image path
detect_landmark(image_path)

# we could utilize this since i got it working, would use gcp funds from other class, but I think we should be fine to use
#the first 1000 requests are free, the next 1000 are 1.50 a month, i have 30.00 left in credits, so should have plenty of flexibility