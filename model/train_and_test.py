# ============================================================
#                       Import Statements
# ============================================================
import sys
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))
from dataset_preparation import houseDataset


# ============================================================
#                       Configuration
# ============================================================
EPOCHS = 10
BATCH_SIZE = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILE = "model/house_model.pth"

# ============================================================
#                       Load Dataset Splits
# ============================================================
print("Loading dataset splits...")

train_dataset = houseDataset("train")
validation_dataset = houseDataset("validation")
test_dataset = houseDataset("test")

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

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_calculation(outputs, masks)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    training_loss /= len(train_loader)
    print(f"Training loss for epoch {epoch}: {training_loss}")

    # ========== Validation ==========
    model.eval()
    validation_loss = 0.0
    validation_iou = 0.0
    validation_dice = 0.0

    with torch.no_grad():
        for images, masks in validation_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_calculation(outputs, masks)
            validation_loss += loss.item()

            outputs_int = (torch.sigmoid(outputs) > 0.5).int()
            masks_int = masks.int()
            tp, fp, fn, tn = smp.metrics.get_stats(outputs_int, masks_int, mode="binary")
            validation_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            validation_dice += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    
    validation_loss /= len(validation_loader)
    validation_iou /= len(validation_loader)
    validation_dice /= len(validation_loader)
    print(f"Validation loss for epoch {epoch}: {validation_loss}")
    print(f"Validation IoU for epoch {epoch}: {validation_iou}")
    print(f"Validation dice for epoch {epoch}: {validation_dice}")

    scheduler.step(validation_loss)

print("Training complete!")

# ============================================================
#                    Save the Trained Model
# ============================================================
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_FILE)
print(f"Model has been saved to {MODEL_FILE}")

# ============================================================
#                    Save the Trained Model
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
print(f"Test IoU: {test_iou}")
print(f"Test dice: {test_dice}")

print("Testing complete!")








