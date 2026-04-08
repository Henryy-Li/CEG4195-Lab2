# Run this file once before running the model itself.

# ============================================================
#                       Import Statements
# ============================================================
import os
import numpy as np

from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

# ============================================================
#                 Load Huggingface Datset Token 
# ============================================================
print("Dataset preparation starting...")
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
#                 Create Input & Output Directories
# ============================================================
os.makedirs("data/images", exist_ok=True)       # Inputs
os.makedirs("data/masks", exist_ok=True)        # Outputs

# ============================================================
#                      Load the Dataset
# ============================================================
print("Loading dataset...")
dataset = load_dataset("keremberke/satellite-building-segmentation", name="mini", trust_remote_code=True)
print("Dataset loaded!")

# ============================================================
#                       Mask Function
# ============================================================
def mask(bounding_box, image):
    x_min, y_min, width, height = bounding_box
    x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
    mask_array = np.zeros((image.width, image.height))
    
    last_x = x_min + width
    last_y = y_min + height
    mask_array[x_min:last_x, y_min:last_y] = np.ones((width, height))
    return mask_array.T
# ============================================================
#                       Process each image     
# ============================================================
print("Processing images...")

for split in ["train", "validation", "test"]:
    print(f"Processing the \"{split}\" split...")
    os.makedirs(f"data/images/{split}", exist_ok=True)
    os.makedirs(f"data/masks/{split}", exist_ok=True)

    for index, data_sample in enumerate(dataset[split]):
        # Image specific
        print(f"Image {index+1}/{len(dataset[split])}")
        image = data_sample["image"]
        combined_mask = np.zeros((image.height, image.width))

        for house_mask in data_sample["objects"]["bbox"]:
            house_seg = mask(house_mask, image)
            combined_mask = np.logical_or(combined_mask, house_seg).astype(int)

        image.save(f"data/images/{split}/{index}.png")
        mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))
        mask_image.save(f"data/masks/{split}/{index}.png")

print("Processing images complete!")
print("Dataset preparation complete!")
