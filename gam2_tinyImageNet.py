import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import time

# --- 1. SETTINGS FOR SURGICAL GAM-2 ---
device = torch.device("cuda")
BATCH_SIZE = 256 # Optimal for stability and 4090 VRAM
EPOCHS = 25      
torch.backends.cudnn.benchmark = True

# --- 2. AUGMENTED DATA (From your 65% success) ---
print(">>> Loading Data with Augmentations...")
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

def train_transform_fn(examples):
    examples["pixel_values"] = [train_transform(image) for image in examples["image"]]
    return examples

def val_transform_fn(examples):
    examples["pixel_values"] = [val_transform(image) for image in examples["image"]]
    return examples

dataset.set_transform(train_transform_fn)
val_dataset.set_transform(val_transform_fn)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "label": labels}

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

# --- 3. GAM-2 ARCHITECTURE (Pairs Only) ---
class InstaSHAP_GAM2(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        res50 = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(res50.children())[:-2]).to(device)
        
        # SURGICAL LOCK: Freeze layers 0-6, Unfreeze layer 7 (Layer 4)
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone[7].parameters(): p.requires_grad = True

        self.embedding_proj = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head_order1 = nn.Linear(embed_dim, 200)
        self.head_order2 = nn.Linear(embed_dim, 200)

    def forward(self, x, mask):
        feat = self.backbone(x)
        B, C, H, W = feat.shape
        pts = feat.view(B, C, H*W).permute(0, 2, 1)
        h = self.norm(self.embedding_proj(pts))
        h_masked = h * mask.unsqueeze(-1)
        
        # GAM-2 Math: Main Effects + Pairs
        sum_h = torch.sum(h_masked, dim=1)
        sum_h_sq = torch.sum(h_masked**2, dim=1)
        
        o1 = self.head_order1(h_masked).sum(dim=1)
        o2 = self.head_order2(0.5 * (sum_h**2 - sum_h_sq))
        
        return o1 + o2

model = InstaSHAP_GAM2().to(device)

# --- 4. EXACT OPTIMIZER FROM 65% RUN ---
optimizer = optim.Adam([
    {'params': model.backbone[7].parameters(), 'lr': 1e-5}, 
    {'params': model.embedding_proj.parameters(), 'lr': 1e-3},
    {'params': model.norm.parameters(), 'lr': 1e-3},
    {'params': model.head_order1.parameters(), 'lr': 1e-3},
    {'params': model.head_order2.parameters(), 'lr': 1e-3}
])

# Use Plateau scheduler based on accuracy
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

# --- 5. TRAINING LOOP ---
print(">>> Starting GAM-2 Surgical Fine-Tuning...")
best_acc = 0.0
d
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    start_time = time.time()
    
    for batch in train_loader:
        imgs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mask = torch.bernoulli(torch.full((imgs.size(0), 49), 0.5, device=device))
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(imgs, mask)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    # VALIDATION
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            mask = torch.ones((imgs.size(0), 49), device=device)
            
            with torch.amp.autocast('cuda'):
                logits = model(imgs, mask)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = (correct / total) * 100
    epoch_time = time.time() - start_time
    
    # Step scheduler based on Validation Accuracy
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_gam2_surgical.pth")
        
print(f">>> Finished! Best Accuracy: {best_acc:.2f}%")