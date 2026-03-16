import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import KFold
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. RTX 4090 SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 512
EPOCHS = 15      
K_FOLDS = 5

# --- 2. DATASET PREPARATION ---
print(">>> Loading Tiny-ImageNet...")
train_hf = load_dataset("zh-plus/tiny-imagenet", split="train") 
val_hf = load_dataset("zh-plus/tiny-imagenet", split="valid")
full_dataset_hf = concatenate_datasets([train_hf, val_hf])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CVDataset(Dataset):
    def __init__(self, hf_data, transform):
        self.hf_data = hf_data
        self.transform = transform
    def __len__(self): return len(self.hf_data)
    def __getitem__(self, idx):
        item = self.hf_data[idx]
        return self.transform(item["image"]), item["label"]

# --- 3. ARCHITECTURE ---
class InstaSHAP_Pro(nn.Module):
    def __init__(self):
        super().__init__()
        res50 = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(res50.children())[:-2])
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone[7].parameters(): p.requires_grad = True # Unfreeze layer 4

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 200)
        )

    def forward(self, x, mask, return_shap=False):
        feat = self.backbone(x) 
        B, C, H, W = feat.shape
        pts = feat.view(B, C, H*W).permute(0, 2, 1)
        m_out = self.head(pts) # SHAP values [B, 49, Classes]
        m_sum = (m_out * mask.unsqueeze(-1)).sum(dim=1)
        if return_shap:
            return m_sum, m_out
        return m_sum

# --- 4. EVALUATION METRICS ---
def faithfulness_deletion_test(model, val_loader):
    """Measures how fast confidence drops when masking top SHAP features."""
    model.eval()
    mask_fractions = [0.0, 0.1, 0.3, 0.5, 0.7]
    confidence_drops = {frac: [] for frac in mask_fractions}
    
    with torch.no_grad():
        # Get one batch for the test
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        
        full_mask = torch.ones((imgs.size(0), 49), device=device)
        base_logits, shap_values = model(imgs, full_mask, return_shap=True)
        base_probs = F.softmax(base_logits, dim=1)
        base_true_probs = base_probs[torch.arange(imgs.size(0)), labels]
        
        for i in range(imgs.size(0)):
            img_shap = shap_values[i, :, labels[i]] 
            sorted_indices = torch.argsort(img_shap, descending=True)
            
            for frac in mask_fractions:
                num_to_drop = int(frac * 49)
                test_mask = torch.ones(49, device=device)
                if num_to_drop > 0:
                    test_mask[sorted_indices[:num_to_drop]] = 0.0
                    
                test_logit = model(imgs[i:i+1], test_mask.unsqueeze(0)) 
                test_prob = F.softmax(test_logit, dim=1)[0, labels[i]]
                confidence_drops[frac].append((base_true_probs[i] - test_prob).item())

    print("\n[Faithfulness] Masking Top-k% features -> Avg Confidence Drop")
    for frac in mask_fractions:
        print(f"  Drop {frac*100:2.0f}% patches : {np.mean(confidence_drops[frac]):+.4f}")

def consistency_test(model_paths, val_loader):
    """Measures standard deviation of SHAP values across the 5-fold ensemble."""
    print("\n>>> Running Consistency Test across Ensemble...")
    models = []
    for path in model_paths:
        m = InstaSHAP_Pro().to(device)
        # Handle torch.compile prefix '_orig_mod.' if present
        state_dict = torch.load(path)
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        m.load_state_dict(clean_dict)
        m.eval()
        models.append(m)

    with torch.no_grad():
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        full_mask = torch.ones((imgs.size(0), 49), device=device)
        
        ensemble_shaps = [] # Will hold [K_models, Batch, 49]
        for m in models:
            _, shap_vals = m(imgs, full_mask, return_shap=True)
            # Extract SHAP values for the true class
            true_class_shaps = shap_vals[torch.arange(imgs.size(0)), :, labels]
            ensemble_shaps.append(true_class_shaps)
            
        # Stack to shape [K_models, Batch, 49]
        ensemble_shaps = torch.stack(ensemble_shaps)
        
        # Calculate standard deviation across the K models (dim=0)
        shap_std = torch.std(ensemble_shaps, dim=0) 
        mean_std = torch.mean(shap_std).item()
        
    
    print(f"[Consistency] Average SHAP Standard Deviation across 5 folds: {mean_std:.4f}")
    print("  (Lower variance = Higher Consistency. Compare this against a baseline GAM!)")

# --- 5. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    saved_models = []

    print(f">>> Starting {K_FOLDS}-Fold Training...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset_hf)))):
        print(f"\n{'='*20} Fold {fold + 1} {'='*20}")
        
        train_loader = DataLoader(Subset(CVDataset(full_dataset_hf, train_transform), train_idx), 
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(Subset(CVDataset(full_dataset_hf, val_transform), val_idx), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        model = InstaSHAP_Pro().to(device)
        compiled_model = torch.compile(model) # Compile for 4090 speed
        
        optimizer = optim.AdamW([
            {'params': compiled_model.backbone[7].parameters(), 'lr': 1e-5},
            {'params': compiled_model.head.parameters(), 'lr': 1e-3}
        ], fused=True)

        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        model_path = f"checkpoints/instashap_fold_{fold+1}.pth"

        for epoch in range(EPOCHS):
            compiled_model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                mask = torch.bernoulli(torch.full((imgs.size(0), 49), 0.5, device=device))
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = criterion(compiled_model(imgs, mask), labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            compiled_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    mask = torch.ones((imgs.size(0), 49), device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = compiled_model(imgs, mask)
                    correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                    total += labels.size(0)

            acc = (correct / total) * 100
            print(f"  Epoch {epoch+1} | Val Acc: {acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(compiled_model.state_dict(), model_path)
                
        saved_models.append(model_path)

    # --- 6. RUN EXPERIMENTS ---
    print("\n" + "="*40)
    print(">>> RUNNING POST-TRAINING A* EXPERIMENTS")
    print("="*40)
    
    # We use the final val_loader for quick testing
    test_loader = DataLoader(Subset(CVDataset(full_dataset_hf, val_transform), val_idx), 
                             batch_size=128, shuffle=True)
    
    # 1. Faithfulness on the Best Model from Fold 1
    best_fold1 = InstaSHAP_Pro().to(device)
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in torch.load(saved_models[0]).items()}
    best_fold1.load_state_dict(clean_dict)
    faithfulness_deletion_test(best_fold1, test_loader)
    
    # 2. Consistency across all 5 Folds
    consistency_test(saved_models, test_loader)

###############################


    
def visualize_ensemble_shaps(model_paths, image_tensor, original_image, target_class=None):
    """
    Visualizes the mean SHAP values across the 5-fold ensemble 
    and the standard deviation (consistency) as a secondary heatmap.
    """
    all_shaps = []
    image_tensor = image_tensor.to(device)
    mask = torch.ones((1, 49), device=device)
    
    # 1. Collect SHAP values from all 5 models
    for path in model_paths:
        m = InstaSHAP_Pro().to(device)
        state_dict = torch.load(path)
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        m.load_state_dict(clean_dict)
        m.eval()
        
        with torch.no_grad():
            logits, shap_vals = m(image_tensor, mask, return_shap=True)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            
            # Extract SHAP for the target class: shape [49]
            all_shaps.append(shap_vals[0, :, target_class].cpu())

    # 2. Compute Mean and Std (Consistency)
    all_shaps = torch.stack(all_shaps) # [5, 49]
    mean_shap = all_shaps.mean(dim=0).view(7, 7)
    std_shap = all_shaps.std(dim=0).view(7, 7)

    # 3. Interpolate for high-res overlay
    def upscale(grid):
        return F.interpolate(grid.unsqueeze(0).unsqueeze(0), 
                             size=(224, 224), mode='bicubic').squeeze().numpy()

    mean_map = upscale(mean_shap)
    std_map = upscale(std_shap)

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Mean SHAP (Importance)
    vmax = np.max(np.abs(mean_map))
    im1 = axes[1].imshow(original_image, alpha=0.3)
    im1 = axes[1].imshow(mean_map, cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.7)
    axes[1].set_title(f"Mean SHAP (Class {target_class})\nRed=Positive, Blue=Negative")
    plt.colorbar(im1, ax=axes[1])
    axes[1].axis('off')

    # Std SHAP (Consistency)
    im2 = axes[2].imshow(std_map, cmap='viridis')
    axes[2].set_title("Inconsistency (Std Dev)\nYellow=High Variance")
    plt.colorbar(im2, ax=axes[2])
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()




    ############################
# This should give the three-panel figure required for the Qualitative Analysis section of the research paper

# Grab a sample image from the test set
sample_img, sample_label = next(iter(test_loader))
# Convert tensor to displayable image
original_img_np = sample_img[0].permute(1, 2, 0).numpy()
# Normalize for display
original_img_np = (original_img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
original_img_np = np.clip(original_img_np, 0, 1)

visualize_ensemble_shaps(saved_models, sample_img[0:1], original_img_np, target_class=sample_label[0].item())



