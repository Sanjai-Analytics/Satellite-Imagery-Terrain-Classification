import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import torch.nn.functional as F
import ee

ee.Authenticate(
    auth_mode='colab',
    force=True
)

print("Authentication completed")

import ee

ee.Initialize(
    project='my_project_id'
)

print("Google Earth Engine initialized successfully")

import ee
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta

# 1. Initialize Google Earth Engine
ee.Initialize(project='my_project_id')

def get_latest_sentinel_image(lat, lon, size_meters=640, max_cloud_cover=10):
    """Fetches the most recent cloud-free Sentinel-2 RGB image for a given location."""

    # 2. Calculate a strict 640x640 meter bounding box (EuroSAT standard)
    # 10m/pixel spatial resolution * 64 pixels = 640 meters
    R = 6378137 # Earth's radius in meters
    d_lat = (size_meters / 2) / R
    d_lon = (size_meters / 2) / (R * np.cos(np.pi * lat / 180))

    roi = ee.Geometry.Polygon([[
        [lon - np.degrees(d_lon), lat - np.degrees(d_lat)],
        [lon + np.degrees(d_lon), lat - np.degrees(d_lat)],
        [lon + np.degrees(d_lon), lat + np.degrees(d_lat)],
        [lon - np.degrees(d_lon), lat + np.degrees(d_lat)],
        [lon - np.degrees(d_lon), lat - np.degrees(d_lat)]
    ]])

    # 3. Define dynamic date range (look back 60 days to ensure a cloud-free pass is found)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    # 4. Query the Sentinel-2 Surface Reflectance (Level-2A) catalog
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                  .sort('system:time_start', False)) # Sort descending to get the absolute latest

    # Extract the single most recent image from the collection
    latest_image = ee.Image(collection.first())

    # 5. Extract RGB bands and clip exactly to our 640m bounding box
    rgb_image = latest_image.select(['B4', 'B3', 'B2']).clip(roi)

    # 6. Generate the download URL
    # Sentinel-2 stores reflectance as 16-bit integers. Setting max: 3000 scales the brightness properly for an 8-bit image.
    vis_params = {
        'min': 0,
        'max': 3000,
        'dimensions': [64, 64], # Force 64x64 pixel dimensions to match EuroSAT model expectations
        'format': 'png'
    }

    url = rgb_image.getThumbURL(vis_params)
    print(f"Fetching image from Earth Engine: {url}")

    # 7. Download directly into memory (No saving to disk required)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    return img

# --- Test the Pipeline ---
# Example coordinates: A farmland region in California
test_lat, test_lon = 36.7783, -119.4179
live_image = get_latest_sentinel_image(test_lat, test_lon)

# Save just to verify visually, but 'live_image' is ready for PyTorch
live_image.save("live_sat_feed.png")
print("Live image loaded into memory! Ready to pass to ResNet50.")


def predict_live_satellite_feed(pil_image, model_path, class_names):
    """
    Takes a PIL image directly from memory, prepares it as a tensor,
    and returns a Pandas DataFrame of predictions.
    """
    # 1. Device routing (CUDA if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Rebuild the ResNet50 architecture and load the trained weights
    model = models.resnet50(weights=None) # No need for ImageNet weights here, we are loading yours
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Load your custom state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() # Freeze batch normalization and dropout layers

    # 3. Define the exact test-time transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Transform the PIL image and add the batch dimension
    # model() expects a 4D tensor [batch_size, channels, height, width]
    # unsqueeze(0) converts our [3, 224, 224] tensor to [1, 3, 224, 224]
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 5. Forward pass without tracking gradients (saves memory & compute)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        _, preds = torch.max(outputs, 1)

    # 6. Structure the output
    prob_list = probabilities.cpu().numpy()
    top_class = class_names[preds[0]]

    df_results = pd.DataFrame({
        'Land Cover Type': class_names,
        'Confidence': prob_list
    })

    # Format the confidence as a percentage for cleaner viewing
    df_results['Confidence'] = df_results['Confidence'].apply(lambda x: f"{x * 100:.2f}%")
    df_results = df_results.sort_values(by='Confidence', ascending=False).reset_index(drop=True)

    return top_class, df_results

# --- Execution Example ---
# class_names must exactly match the alphabetical order of your EuroSAT training folders
eurosat_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                   'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Assuming 'live_image' is the variable holding the PIL Image from the GEE script
top_prediction, prediction_dataframe = predict_live_satellite_feed(
    pil_image=live_image,
    model_path='eurosat_resnet50.pth',
    class_names=eurosat_classes
)

print(f"Top Prediction: {top_prediction}\n")
print(prediction_dataframe)

import matplotlib.pyplot as plt
import numpy as np

def display_prediction_results(pil_image, top_class, df_results):
    """
    Displays the satellite image and a bar chart of the top 5 predictions.
    """
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: The Satellite Image ---
    # Matplotlib can plot PIL images directly, or convert to numpy array
    ax1.imshow(np.array(pil_image))
    ax1.axis('off') # Hide gridlines and axes for a cleaner look
    ax1.set_title(f"Predicted Terrain: {top_class}", fontsize=14, fontweight='bold')

    # --- Plot 2: The Confidence Bar Chart ---
    # The previous script formatted 'Confidence' as a string with a '%' sign (e.g., "98.50%").
    # We must strip the '%' and convert back to a float so Matplotlib can plot the numbers.
    confidence_floats = df_results['Confidence'].str.rstrip('%').astype(float)

    # Isolate just the top 5 predictions to keep the chart uncluttered
    top_5_classes = df_results['Land Cover Type'].head(5)
    top_5_scores = confidence_floats.head(5)

    # Build a horizontal bar chart
    y_pos = np.arange(len(top_5_classes))
    bars = ax2.barh(y_pos, top_5_scores, color='#4C72B0')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_5_classes)
    ax2.invert_yaxis()  # Puts the highest probability at the top
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title('Top 5 Class Probabilities', fontsize=14)
    ax2.set_xlim(0, 110) # Set X-axis to 100% plus a little padding for text

    # Add the exact percentage text to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}%', va='center')

    plt.tight_layout()
    plt.show()

# --- Execution ---
# Assuming 'live_image' is from your Earth Engine script,
# and 'top_prediction', 'prediction_dataframe' are from your PyTorch script:

display_prediction_results(live_image, top_prediction, prediction_dataframe)
