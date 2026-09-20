import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import random

from model import EmotionNet

def cutmix(img1, img2, alpha=0.5):
    h, w = img1.shape[:2]
    lam = np.random.beta(alpha, alpha)
    rx, ry = random.randint(0, w-1), random.randint(0, h-1)
    rw, rh = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
    x1, y1 = max(rx - rw // 2, 0), max(ry - rh // 2, 0)
    x2, y2 = min(x1 + rw, w), min(y1 + rh, h)
    img_mix = img1.copy()
    img_mix[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    return img_mix, lam

class FerDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = os.path.join(root, split)
        self.emotion_map = {
            "angry": 0, "disgust": 1, "fear": 2,
            "happy": 3, "sad": 4, "surprise": 5, "neutral": 6
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for emo_name, label in self.emotion_map.items():
            emo_dir = os.path.join(self.root, emo_name)
            if os.path.exists(emo_dir):
                for img_name in os.listdir(emo_dir):
                    data.append((os.path.join(emo_dir, img_name), label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__(np.random.randint(0, len(self.data)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (48, 48))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor, torch.tensor(label)

def main_train():
    device = torch.device("cpu")
    
    dataset_root = r"D:\emotion_exp\dataset\FER2013_aligned"
    train_dataset = FerDataset(dataset_root, split="train")
    val_dataset = FerDataset(dataset_root, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")
    
    model = EmotionNet(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    total_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        
        # CutMix只在训练后期使用（前20轮不使用）
        use_cutmix = epoch >= 20 and np.random.rand() < 0.3
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_cutmix:
                batch_size = images.shape[0]
                perm = torch.randperm(batch_size)
                images_perm = images[perm]
                labels_perm = labels[perm]
                
                for i in range(batch_size):
                    img1 = images[i].permute(1, 2, 0).cpu().numpy()
                    img2 = images_perm[i].permute(1, 2, 0).cpu().numpy()
                    mixed_img, lam = cutmix(img1, img2)
                    images[i] = torch.from_numpy(mixed_img).permute(2, 0, 1)
                
                cls_logits, _ = model(images)
                loss = lam * criterion(cls_logits, labels) + (1 - lam) * criterion(cls_logits, labels_perm)
            else:
                cls_logits, _ = model(images)
                loss = criterion(cls_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred_logits, _ = model(images)
                pred_idx = torch.argmax(pred_logits, dim=1)
                correct += torch.sum(pred_idx == labels).item()
        
        val_acc = correct / len(val_dataset)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), r"D:\emotion_exp\code\emotion_best.pth")
            print(f"★ 保存模型! 验证准确率: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1:02d}/{total_epochs} | 损失: {train_loss/len(train_loader):.4f} | 验证: {val_acc:.4f}")
    
    print(f"\n训练完成! 最佳准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    main_train()