import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

# --- 1. SETTINGS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 200
GRID_SIZE = 7
NUM_PATCHES = 49

# We need 8 images for 8 columns. These are arbitrary indices from the validation set.
# You can change these to pick specific birds/dogs/objects that look good!
IMAGE_INDICES = [10, 25, 42, 65, 88, 105, 120, 150]

MODEL_PATHS = {
    "GAM-1 (CE)": "imagenet_gam1.pth", # Update to your actual GAM-1 path
    "GAM-2 (CE)": "final_paper_results/best_gam2_surgical.pth",
    "GAM-3 (CE)": "final_paper_results/best_gam3_surgical.pth",
    "GAM-2 (KD)": "final_paper_results/best_gam2_distilled.pth",
    "GAM-3 (KD)": "final_paper_results/best_gam3_distilled.pth",
}
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

# Gather the 8 images
images = []
labels = []
for idx in IMAGE_INDICES:
    raw_item = val_hf[idx]
    images.append(val_transform(raw_item['image']).unsqueeze(0).to(device))
    labels.append(raw_item['label'])

# --- 3. ARCHITECTURE ---
class InstaSHAP_GAM(nn.Module):
    def __init__(self, order=1, embed_dim=128):
        super().__init__()
        self.order = order
        res50 = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(res50.children())[:-2])
        self.embedding_proj = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head_order1 = nn.Linear(embed_dim, NUM_CLASSES)
        if order >= 2: self.head_order2 = nn.Linear(embed_dim, NUM_CLASSES)
        if order >= 3: self.head_order3 = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x, mask, return_components=False):
        feat = self.backbone(x)
        B, C, H, W = feat.shape
        pts = feat.view(B, C, H*W).permute(0, 2, 1)
        h = self.norm(self.embedding_proj(pts))
        h_masked = h * mask.unsqueeze(-1)
        
        o1_comp = self.head_order1(h_masked)
        logits = o1_comp.sum(dim=1)
        
        if self.order >= 2:
            sum_h = torch.sum(h_masked, dim=1)
            sum_h_sq = torch.sum(h_masked**2, dim=1)
            logits += self.head_order2(0.5 * (sum_h**2 - sum_h_sq))
        if self.order >= 3:
            sum_h_cubed = torch.sum(h_masked**3, dim=1)
            logits += self.head_order3((1/6.0) * (sum_h**3 - 3*(sum_h*sum_h_sq) + 2*sum_h_cubed))
            
        if return_components: return logits, o1_comp
        return logits

# --- 4. EVALUATION LOOP ---
results = { "Original": [], "Kernel SHAP": [] }
for name in MODEL_PATHS.keys(): results[name] = []

print(">>> Computing Kernel SHAP for all 8 images (This will take ~10 seconds)...")
teacher = models.resnet50(weights=None)
teacher.fc = nn.Linear(teacher.fc.in_features, NUM_CLASSES)
teacher.load_state_dict(torch.load(TEACHER_PATH))
teacher.to(device).eval()

def make_predict_fn(img_t):
    def custom_predict(mask_array):
        N = mask_array.shape[0]
        masks_tensor = torch.tensor(mask_array, dtype=torch.float32, device=device)
        masks_spatial = masks_tensor.view(N, 1, GRID_SIZE, GRID_SIZE)
        masks_upsampled = F.interpolate(masks_spatial, size=(224, 224), mode='nearest')
        masked_imgs = img_t.expand(N, -1, -1, -1) * masks_upsampled
        all_probs = []
        with torch.no_grad():
            for i in range(0, N, 128):
                logits = teacher(masked_imgs[i:i+128])
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(all_probs)
    return custom_predict

for i in range(8):
    img_t = images[i]
    lbl = labels[i]
    # Store Original Image
    img_vis = img_t[0].permute(1, 2, 0).cpu().numpy()
    img_vis = np.clip((img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]), 0, 1)
    results["Original"].append(img_vis)
    
    # Run Kernel SHAP
    bg_mask = np.zeros((1, NUM_PATCHES))
    explainer = shap.KernelExplainer(make_predict_fn(img_t), bg_mask)
    shap_vals = explainer.shap_values(np.ones((1, NUM_PATCHES)), nsamples=1000)
    ks_map = shap_vals[lbl][0] if isinstance(shap_vals, list) else shap_vals[0, :, lbl]
    results["Kernel SHAP"].append(ks_map)

print(">>> Running GAM Models...")
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f"  [SKIP] {name} missing at {path}")
        continue
    
    order = int(name[4])
    model = InstaSHAP_GAM(order=order).to(device)
    
    # --- CLEVER LOADING START ---
    state_dict = torch.load(path, map_location=device)
    
    # 1. Remove '_orig_mod.' prefix if it exists (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        name_key = k.replace("_orig_mod.", "") 
        new_state_dict[name_key] = v
        
    # 2. Load with strict=False to ignore 'num_batches_tracked' mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    # print(f"  Loaded {name} (Missing keys: {len(missing)}, Unexpected: {len(unexpected)})")
    # --- CLEVER LOADING END ---

    model.eval()
    
    for i in range(8):
        with torch.no_grad():
            _, o1_comp = model(images[i], torch.ones((1, NUM_PATCHES), device=device), return_components=True)
            results[name].append(o1_comp[0, :, labels[i]].cpu().numpy())


# --- 5. PLOTTING THE 8x8 GRID ---
print(">>> Drawing Grid...")
# Filter out models that were skipped so we don't have empty rows
active_rows = [row for row in results.keys() if len(results[row]) > 0]
num_rows = len(active_rows)
num_cols = 8

fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 2 * num_rows))
fig.subplots_adjust(wspace=0.05, hspace=0.05)

for r, row_name in enumerate(active_rows):
    for c in range(num_cols):
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        
        if row_name == "Original":
            ax.imshow(results[row_name][c])
            # Add class label or image index to the top
            if r == 0: ax.set_title(f"Image {IMAGE_INDICES[c]}", fontsize=12)
        else:
            img_vis = results["Original"][c]
            h_map = results[row_name][c].reshape(7, 7)
            # Use 'nearest' to get the blocky grid look from your photo
            h_resized = F.interpolate(torch.tensor(h_map).unsqueeze(0).unsqueeze(0), 
                                      size=(224, 224), mode='nearest').squeeze().numpy()
            
            vmax = np.max(np.abs(h_resized))
            if vmax == 0: vmax = 1 # Prevent division by zero
            
            ax.imshow(img_vis, alpha=0.4) # Make background image faint
            # Use 'bwr' for Blue-White-Red
            ax.imshow(h_resized, cmap='bwr', alpha=0.7, vmin=-vmax, vmax=vmax)

        # Add Row Labels to the far left
        if c == 0:
            ax.set_ylabel(row_name, fontsize=12, fontweight='bold', rotation=0, labelpad=40, va='center')

# Remove outer borders
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.savefig("final_paper_results/Figure_6_Large_Grid.png", bbox_inches='tight', dpi=400)
print(">>> Grid successfully saved to final_paper_results/Figure_6_Large_Grid.png")