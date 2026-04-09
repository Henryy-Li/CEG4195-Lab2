# This file is used to train the model using the training and validation sets.
# Then it is used to test the model on the test set.

# ============================================================
#                       Import Statements
# ============================================================
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))
from dataset_preparation import houseDataset

# ============================================================
#            Configuration of the Model & Constants
# ============================================================
EPOCHS = 20
BATCH_SIZE = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILE = "model/house_model.pth"

# ============================================================
#                       Load Dataset Splits
# ============================================================
print("Loading dataset splits...")

# Load the dataset splits
train_dataset = houseDataset("train")
validation_dataset = houseDataset("validation")
test_dataset = houseDataset("test")

# Convert the splits to be stored in batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading dataset splits complete!")

# ============================================================
#                       Load the Model
# ============================================================
print("Loading model...")

model = smp.Unet(
    encoder_name = "resnet34",
    encoder_weights = "imagenet",
    in_channels = 3,
    classes = 1,
)
model.to(DEVICE)

print("Loading model complete!")

# ============================================================
#                       Loss and Optimizer
# ============================================================
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = smp.losses.SoftBCEWithLogitsLoss()

def loss_calculation(outputs, masks):
    return dice_loss(outputs, masks) + bce_loss(outputs, masks)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# ============================================================
#                  Training & Validation Sets
# ============================================================
print("Training model...")

for epoch in range (EPOCHS):
    # ========== Training ==========
    training_loss = 0.0
    model.train()

    # Iterate through each batch
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_calculation(outputs, masks)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    training_loss /= len(train_loader)              # Average loss per epoch
    print("")
    print(f"Training loss for epoch {epoch+1}: {training_loss:.3f}")

    # ========== Validation ==========
    model.eval()
    validation_loss = 0.0
    validation_iou = 0.0
    validation_dice = 0.0

    with torch.no_grad():
        # Iterate through each batch
        for images, masks in validation_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_calculation(outputs, masks)
            validation_loss += loss.item()

            outputs_int = (torch.sigmoid(outputs) > 0.5).int()      # Converts model outputs to probabilities and then to integers for the mask.
            masks_int = masks.int()

            tp, fp, fn, tn = smp.metrics.get_stats(outputs_int, masks_int, mode="binary")               # Metrics
            validation_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            validation_dice += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    
    validation_loss /= len(validation_loader)                               # Average metric values for this epoch.
    validation_iou /= len(validation_loader)
    validation_dice /= len(validation_loader)
    print(f"Validation loss for epoch {epoch+1}: {validation_loss:.3f}")
    print(f"Validation IoU for epoch {epoch+1}: {validation_iou:.3f}")
    print(f"Validation dice for epoch {epoch+1}: {validation_dice:.3f}")

    scheduler.step(validation_loss)

print()
print("Training complete!")

# ============================================================
#                    Save the Trained Model
# ============================================================
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_FILE)
print(f"Model has been saved to {MODEL_FILE}")

# ============================================================
#                    Test the Trained Model
# ============================================================
print("Testing model...")

test_iou = 0.0
test_dice = 0.0
model.eval()

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        outputs_int = (torch.sigmoid(outputs) > 0.5).int()
        masks_int = masks.int()

        tp, fp, fn, tn = smp.metrics.get_stats(outputs_int, masks_int, mode="binary")
        test_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        test_dice += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()

test_iou /= len(test_loader)
test_dice /= len(test_loader)
print(f"Test IoU: {test_iou:.3f}")
print(f"Test dice: {test_dice:.3f}")

print("Testing complete!")

# ============================================================
#                    Visualization
# ============================================================
print("Visualizing predictions...")

model.eval()

num_examples = min(3, len(test_dataset))                                            # Display a max of 3 sets of aerial images, ground truth masks, and predicted masks.
figure, axes = plt.subplots(num_examples, 3, figsize=(12,num_examples*4))

with torch.no_grad():
    # Iterate by image
    for i, (image, mask) in enumerate(test_dataset):
        if i>=num_examples:
            break

        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        
        # Display the prediction
        outputs = model(image.unsqueeze(0))
        outputs_int = (torch.sigmoid(outputs) > 0.5).squeeze().int()
        display_output = outputs_int.cpu().numpy()

        # Display the image
        display_image = image.cpu().permute(1,2,0).numpy()
        display_image = display_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        display_image = np.clip(display_image, 0, 1)

        # Display the mask
        display_mask = mask.cpu().squeeze().numpy()

        # Put images onto the grid
        axes[i][0].imshow(display_image)
        axes[i][0].set_title("Aerial Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(display_mask, cmap="gray")
        axes[i][1].set_title("Ground Truth Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(display_output, cmap="gray")
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")

    # Final adjustments
    plt.tight_layout()
    plt.savefig("visualizations.png")
    print("Visualizations saved!")

