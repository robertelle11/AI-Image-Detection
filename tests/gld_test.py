import os
import pandas as pd
import glob
import requests

# Load metadata
train_df = pd.read_csv("../spreadsheets/train.csv")  # Image ID → Landmark ID
label_df = pd.read_csv("../spreadsheets/train_label_to_category.csv")  # Landmark ID → Category (Wikimedia URL)

# Create mappings
id_to_landmark = dict(zip(train_df["id"], train_df["landmark_id"]))
landmark_to_category = dict(zip(label_df["landmark_id"], label_df["category"]))

def find_image(image_id, base_dir="../train"):
    """Find the image file path based on the image ID."""
    subdir = os.path.join(base_dir, image_id[0], image_id[1], image_id[2])  # Directory structure
    search_pattern = os.path.join(subdir, f"{image_id}.jpg")

    matching_files = glob.glob(search_pattern)
    return matching_files[0] if matching_files else None

def get_landmark_info(image_id):
    """Return landmark details (Wikimedia category, GPS coordinates) for an image ID."""
    landmark_id = id_to_landmark.get(image_id)
    if not landmark_id:
        return {"error": "No landmark found"}

    category_url = landmark_to_category.get(landmark_id, "Unknown")
    
    # Extract Wikimedia page name
    category_name = category_url.split("/")[-1] if "wikimedia" in category_url else "Unknown"
    
    # Get coordinates from Wikidata API
    coordinates = get_coordinates_from_wikidata(category_name)

    return {
        "landmark_id": landmark_id,
        "category": category_url,
        "category_name": category_name,
        "coordinates": coordinates
    }

def get_coordinates_from_wikidata(category_name):
    """Fetch GPS coordinates from Wikidata API based on the landmark's Wikimedia category."""
    if category_name == "Unknown":
        return None

    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": category_name,
        "language": "en",
        "format": "json"
    }
    
    response = requests.get(wikidata_url, params=params)
    data = response.json()

    if "search" in data and len(data["search"]) > 0:
        entity_id = data["search"][0]["id"]
        
        # Fetch entity details
        details_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        response = requests.get(details_url)
        data = response.json()
        
        try:
            entity_data = data["entities"][entity_id]
            lat = entity_data["claims"]["P625"][0]["mainsnak"]["datavalue"]["value"]["latitude"]
            lon = entity_data["claims"]["P625"][0]["mainsnak"]["datavalue"]["value"]["longitude"]
            return {"latitude": lat, "longitude": lon}
        except KeyError:
            return None
    return None

def locate_image_info(image_id):
    """Find an image and retrieve its landmark information including GPS coordinates."""
    image_path = find_image(image_id)

    if image_path:
        landmark_info = get_landmark_info(image_id)
        print(f"Image found at: {image_path}")
        print(f"Landmark ID: {landmark_info['landmark_id']}")
        print(f"Category: {landmark_info['category']}")
        print(f"Landmark Name: {landmark_info['category_name']}")

        if landmark_info["coordinates"]:
            print(f"Coordinates: {landmark_info['coordinates']['latitude']}, {landmark_info['coordinates']['longitude']}")
        else:
            print("Coordinates not found.")

    else:
        print("Image not found in dataset.")

# Example usage
image_id = "002a196a7a4d4e48"
locate_image_info(image_id)