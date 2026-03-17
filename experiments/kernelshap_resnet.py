import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import shap
import time
import os

# --- 1. SETTINGS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 200 # Tiny-ImageNet
GRID_SIZE = 7
NUM_PATCHES = GRID_SIZE * GRID_SIZE
TEACHER_PATH = "teacher_models/resnet50_teacher_best.pth"

# --- 2. LOAD DATA ---
print(">>> Loading Data...")
val_hf = load_dataset("zh-plus/tiny-imagenet", split="valid")
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fetch a single image
idx = 0  # Change this to test different images
raw_item = val_hf[idx]
img_tensor = val_transform(raw_item['image']).unsqueeze(0).to(device)
label_idx = raw_item['label']

# --- 3. LOAD TEACHER MODEL ---
print(">>> Loading ResNet-50 Teacher Model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(TEACHER_PATH))
model = model.to(device)
model.eval()

# --- 4. PREDICTION WRAPPER FOR KERNEL SHAP ---
# Kernel SHAP needs a function that takes a binary mask (shape: N x 49),
# applies it to the image, and returns the model probabilities.
def custom_predict(mask_array):
    """
    mask_array: numpy array of shape (N, 49) containing 1s (keep) and 0s (hide)
    """
    N = mask_array.shape[0]
    
    # 1. Convert 1D masks (N, 49) into 2D spatial masks (N, 1, 7, 7)
    masks_tensor = torch.tensor(mask_array, dtype=torch.float32, device=device)
    masks_spatial = masks_tensor.view(N, 1, GRID_SIZE, GRID_SIZE)
    
    # 2. Upsample to image size (N, 1, 224, 224) using Nearest Neighbor to keep hard edges
    masks_upsampled = F.interpolate(masks_spatial, size=(224, 224), mode='nearest')
    
    # 3. Expand the original image to match the batch size N
    img_expanded = img_tensor.expand(N, -1, -1, -1)
    
    # 4. Apply the mask (masked areas become 0, which corresponds to the mean pixel value in normalized space)
    masked_imgs = img_expanded * masks_upsampled
    
    # 5. Run prediction in chunks to avoid Out-Of-Memory on the GPU
    batch_size = 128
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_imgs = masked_imgs[i:i+batch_size]
            with torch.amp.autocast('cuda'):
                logits = model(batch_imgs)
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.vstack(all_probs)

# --- 5. RUN KERNEL SHAP ---
print(f">>> Running Kernel SHAP for Image {idx} (Target Class: {label_idx})...")
start_time = time.time()

# Background and instance
background_mask = np.zeros((1, NUM_PATCHES))
instance_to_explain = np.ones((1, NUM_PATCHES))

explainer = shap.KernelExplainer(custom_predict, background_mask)

# nsamples determines how many random masks are tested
shap_values = explainer.shap_values(instance_to_explain, nsamples=1000)

# We must select the array corresponding to the label_idx.
if isinstance(shap_values, list):
    # This handles the multi-class list output
    target_shap_values = shap_values[label_idx][0]
else:
    # This handles cases where only one class output is returned
    target_shap_values = shap_values[0, :, label_idx]

execution_time = time.time() - start_time
print(f"Kernel SHAP completed in {execution_time:.2f} seconds.")

# --- 6. VISUALISATION ---
shap_map = target_shap_values.reshape(GRID_SIZE, GRID_SIZE)
shap_tensor = torch.tensor(shap_map).unsqueeze(0).unsqueeze(0)
shap_resized = F.interpolate(shap_tensor, size=(224, 224), mode='bicubic').squeeze().numpy()

# Denormalize Image for plotting
img_vis = img_tensor[0].permute(1, 2, 0).cpu().numpy()
img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
img_vis = np.clip(img_vis, 0, 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_vis)
plt.title(f"Original (Class: {label_idx})")
plt.axis('off')

plt.subplot(1, 2, 2)
vmax = np.max(np.abs(shap_resized))
plt.imshow(img_vis, alpha=0.5)
im = plt.imshow(shap_resized, cmap='jet', alpha=0.6, vmin=-vmax, vmax=vmax)
plt.title(f"Kernel SHAP (Time: {execution_time:.1f}s)")
plt.axis('off')
plt.colorbar(im)

plt.savefig("final_paper_results/kernel_shap_baseline.png")
print(">>> Saved Kernel SHAP baseline to final_paper_results/kernel_shap_baseline.png")