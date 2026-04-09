'''
Course: CEG 4195
Name:   Henry Li

Instructions for running the lab, see lab document.
'''
# ============================================================
#                       Import Statements
# ============================================================
from flask import Flask, request, jsonify  
from PIL import Image
from dotenv import load_dotenv

import torch
import segmentation_models_pytorch as smp
import numpy as np
import io
import base64
import os

# ============================================================
#                        Main Code
# ============================================================

# ===== Get Huggingface Token =====
load_dotenv()
HF_token = os.getenv("HF_TOKEN")

# ===== Create the Flask app =====
app = Flask(__name__)

# ===== Load the Model =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=3, 
    classes=1,
)
model.load_state_dict(torch.load("model/house_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== Image Processing =====
def image_processing(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = torch.tensor(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)
    return image.to(DEVICE)

# ===== Application =====
@app.route("/predict", methods=["POST"])

def predict():                      #Function that does the image analysis
    # === Process the input ===
    JSONData = request.json
    imageBytes = base64.b64decode(JSONData["image"])
    imageTensorInput = image_processing(imageBytes)

    # === Model output ===
    with torch.no_grad():
        outputs = model(imageTensorInput)
        outputs_int = (torch.sigmoid(outputs) > 0.5).squeeze().int()

    # === Create the JSON output ===
    house_pixels = int(outputs_int.sum().item())
    total_pixels = outputs_int.numel()
    house_coverage = round(house_pixels/total_pixels*100,2)

    return jsonify({
        "house_pixels": house_pixels,
        "total_pixels": total_pixels,
        "house_coverage": house_coverage
    })

if __name__ == "__main__":          # App runs only if the script is ran directly.
    app.run(debug=True)
