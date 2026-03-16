import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import time
import os

# --- 1. SETTINGS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 256
EPOCHS = 25
NUM_CLASSES = 200
TEMPERATURE = 2.0  # Softening parameter for Distillation
ALPHA = 0.5        # Weight between CE Loss and Distillation Loss

TEACHER_PATH = "teacher_models/resnet50_teacher_best.pth"

if not os.path.exists(TEACHER_PATH):
    raise FileNotFoundError(f"Teacher model not found at {TEACHER_PATH}. Please run the teacher training script first!")

# --- 2. DATA WITH AUGMENTATION ---
print(">>> Loading Tiny-ImageNet...")
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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

# --- 3. TEACHER MODEL (FROZEN) ---
print(">>> Loading Teacher Model...")
teacher = models.resnet50(weights=None)
teacher.fc = nn.Linear(teacher.fc.in_features, NUM_CLASSES)
teacher.load_state_dict(torch.load(TEACHER_PATH))
teacher = teacher.to(device)
teacher.eval() # Teacher is always in eval mode
for p in teacher.parameters():
    p.requires_grad = False

# --- 4. STUDENT GAM-2 ARCHITECTURE (SURGICAL) ---
print(">>> Initializing Student GAM-2...")
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
        self.head_order1 = nn.Linear(embed_dim, NUM_CLASSES)
        self.head_order2 = nn.Linear(embed_dim, NUM_CLASSES)

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

student = InstaSHAP_GAM2().to(device)

# --- 5. OPTIMIZER & SCHEDULER ---
optimizer = optim.Adam([
    {'params': student.backbone[7].parameters(), 'lr': 1e-5}, 
    {'params': student.embedding_proj.parameters(), 'lr': 1e-3},
    {'params': student.norm.parameters(), 'lr': 1e-3},
    {'params': student.head_order1.parameters(), 'lr': 1e-3},
    {'params': student.head_order2.parameters(), 'lr': 1e-3}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

# --- 6. TRAINING LOOP WITH DISTILLATION ---
print(">>> Starting Knowledge Distillation Training...")
best_acc = 0.0
os.makedirs("final_paper_results", exist_ok=True)

for epoch in range(EPOCHS):
    student.train()
    train_loss, train_ce, train_kl = 0, 0, 0
    start_time = time.time()
    
    for batch in train_loader:
        imgs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        # Student sees MASKED images, Teacher sees FULL images
        mask = torch.bernoulli(torch.full((imgs.size(0), 49), 0.5, device=device))
        
        optimizer.zero_grad(set_to_none=True)
        
        # Get Teacher's soft targets (No gradients needed)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                teacher_logits = teacher(imgs)
        
        # Get Student's predictions
        with torch.amp.autocast('cuda'):
            student_logits = student(imgs, mask)
            
            # 1. Hard Target Loss (Standard CE)
            loss_ce = criterion_ce(student_logits, labels)
            
            # 2. Soft Target Loss (KL Divergence)
            loss_kl = F.kl_div(
                F.log_softmax(student_logits / TEMPERATURE, dim=1),
                F.softmax(teacher_logits / TEMPERATURE, dim=1),
                reduction='batchmean'
            ) * (TEMPERATURE * TEMPERATURE)
            
            # Combine Losses
            loss = (ALPHA * loss_ce) + ((1.0 - ALPHA) * loss_kl)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        train_ce += loss_ce.item()
        train_kl += loss_kl.item()

    # VALIDATION (Student only, unmasked)
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            mask = torch.ones((imgs.size(0), 49), device=device)
            
            with torch.amp.autocast('cuda'):
                logits = student(imgs, mask)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = (correct / total) * 100
    epoch_time = time.time() - start_time
    scheduler.step(val_acc)

    b_count = len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | CE: {train_ce/b_count:.3f} | KL: {train_kl/b_count:.3f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(student.state_dict(), "final_paper_results/best_gam2_distilled.pth")
        
print(f">>> Finished! Best Distilled Accuracy: {best_acc:.2f}%")