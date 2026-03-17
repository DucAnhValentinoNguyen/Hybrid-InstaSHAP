import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import time

# --- 1. SETTINGS FROM THE 65% SUCCESS ---
device = torch.device("cuda")
BATCH_SIZE = 256 # Match the 65% script's stability
EPOCHS = 20      # Slightly more time for interaction heads
torch.backends.cudnn.benchmark = True

# --- 2. DATA WITH AUGMENTATION ---
print(">>> Loading Data with 65% Script Augmentations...")
dataset = load_dataset("zh-plus/tiny-imagenet", split="train") 
val_dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")

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

class TinyDataset(Dataset):
    def __init__(self, hf_data, transform):
        self.hf_data, self.transform = hf_data, transform
    def __len__(self): return len(self.hf_data)
    def __getitem__(self, idx):
        return self.transform(self.hf_data[idx]["image"]), self.hf_data[idx]["label"]

train_loader = DataLoader(TinyDataset(dataset, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(TinyDataset(val_dataset, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# --- 3. SURGICAL GAM-3 ARCHITECTURE ---
class GAM3_Surgical(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        res50 = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(res50.children())[:-2])
        
        # SURGICAL LOCK: Freeze 0-6, Unfreeze 7 (Layer 4)
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone[7].parameters(): p.requires_grad = True
        
        self.embedding_proj = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head_order1 = nn.Linear(embed_dim, 200)
        self.head_order2 = nn.Linear(embed_dim, 200)
        self.head_order3 = nn.Linear(embed_dim, 200)

    def forward(self, x, mask):
        feat = self.backbone(x)
        B, C, H, W = feat.shape
        h = self.norm(self.embedding_proj(feat.view(B, C, H*W).permute(0, 2, 1)))
        h_masked = h * mask.unsqueeze(-1)
        
        # Factorized Interactions
        sum_h = torch.sum(h_masked, dim=1)
        sum_h_sq = torch.sum(h_masked**2, dim=1)
        sum_h_cub = torch.sum(h_masked**3, dim=1)
        
        o1 = self.head_order1(h_masked).sum(dim=1)
        o2 = self.head_order2(0.5 * (sum_h**2 - sum_h_sq))
        o3 = self.head_order3((1/6) * (sum_h**3 - 3 * sum_h * sum_h_sq + 2 * sum_h_cub))
        
        return o1 + o2 + o3

model = GAM3_Surgical().to(device)

# --- 4. OPTIMIZER FROM THE 65% SUCCESS ---
optimizer = optim.Adam([
    {'params': model.backbone[7].parameters(), 'lr': 1e-5}, # Surgical update
    {'params': model.embedding_proj.parameters(), 'lr': 1e-3},
    {'params': model.head_order1.parameters(), 'lr': 1e-3},
    {'params': model.head_order2.parameters(), 'lr': 1e-3},
    {'params': model.head_order3.parameters(), 'lr': 1e-3}
])

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Added smoothing for 70% target
scaler = torch.amp.GradScaler('cuda')

# --- 5. TRAINING ---
print(">>> Starting Surgical GAM-3 Run...")
for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        mask = torch.bernoulli(torch.full((imgs.size(0), 49), 0.5, device=device))
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(imgs, mask)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            mask = torch.ones((imgs.size(0), 49), device=device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs, mask)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} | Acc: {100*correct/total:.2f}%")