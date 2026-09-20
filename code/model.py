import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        p = torch.exp(-ce_loss)
        focal = self.alpha * torch.pow(1 - p, self.gamma) * ce_loss
        return focal.mean()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x_flat = x3.view(-1, 128 * 6 * 6)
        x_drop = self.dropout(x_flat)
        x_out = self.fc(x_drop)
        return x_out, x_flat

    def get_features(self, x):
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        return [x1, x2, x3], x3

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.cnn = SimpleCNN(num_classes)
    
    def forward(self, x, label=None):
        return self.cnn(x)
    
    def get_features(self, x):
        return self.cnn.get_features(x)

if __name__ == "__main__":
    model = EmotionNet()
    dummy = torch.randn(4, 3, 48, 48)
    pred, _ = model(dummy)
    print("模型测试成功，输出维度：", pred.shape)