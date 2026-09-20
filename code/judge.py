import os
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import EmotionNet
from train import FerDataset

# 重写适配单层文件夹的CKPlusDataset
class CKPlusDataset(Dataset):
    def __init__(self, root):
        self.root = root
        # CK+文件夹名称映射到7分类标签
        self.folder_label_map = {
            "anger": 0,
            "contempt": 1,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "sadness": 4,
            "surprise": 5,
            "neutral": 6
        }
        self.data = self._load()

    def _load(self):
        data = []
        # 遍历每个表情文件夹
        for folder_name, label in self.folder_label_map.items():
            folder_path = os.path.join(self.root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            # 遍历文件夹内所有图片
            img_names = sorted(os.listdir(folder_path))
            for img_name in img_names:
                img_full_path = os.path.join(folder_path, img_name)
                data.append((img_full_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path)
        # 图片损坏容错
        if img is None:
            return self.__getitem__(np.random.randint(0, len(self.data)))
        # 统一转RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (48, 48))
        # 转为模型输入张量
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor, torch.tensor(label)

def evaluate(model, loader, name):
    print(f"\n{'='*50}")
    print(f"【{name}】")
    print(f"{'='*50}")
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out, _ = model(imgs)
            preds.extend(torch.argmax(out, dim=1).numpy())
            labels.extend(lbls.numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    print(f"准确率: {acc*100:.2f}%")
    print(f"F1-score: {f1:.4f}")
    cm = confusion_matrix(labels, preds)
    print("\n混淆矩阵:")
    names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    print(f"{'':>8s}", end="")
    for n in names:
        print(f"{n:>8s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{names[i]:>8s}", end="")
        for val in row:
            print(f"{val:>8d}", end="")
        print()
    return acc, f1

def main():
    model = EmotionNet(num_classes=7)
    model_path = r"D:\emotion_exp\code\emotion_best.pth"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    # 移除weights_only，兼容低版本PyTorch
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"模型加载成功")

    # FER2013测试集评估
    fer_root = r"D:\emotion_exp\dataset\FER2013_aligned"
    fer_test = FerDataset(fer_root, split="test")
    fer_loader = DataLoader(fer_test, batch_size=16)
    print(f"FER-2013测试集样本总数: {len(fer_test)}")
    fer_acc, fer_f1 = evaluate(model, fer_loader, "FER-2013")

    # CK+数据集评估（现在路径匹配你的文件夹）
    ck_root = r"D:\emotion_exp\dataset\CK+"
    ck_acc, ck_f1 = 0, 0
    if os.path.exists(ck_root):
        ck_data = CKPlusDataset(ck_root)
        ck_loader = DataLoader(ck_data, batch_size=16)
        print(f"CK+数据集样本总数: {len(ck_data)}")
        ck_acc, ck_f1 = evaluate(model, ck_loader, "CK+")
    else:
        print("未检测到CK+数据集文件夹")

    # 打印两个数据集对比结果
    print(f"\n{'='*50}")
    print(f"【跨数据集泛化能力对比汇总】")
    print(f"{'='*50}")
    print(f"{'数据集':<15s} {'准确率':<10s} {'F1-score':<10s}")
    print(f"{'FER-2013':<15s} {fer_acc*100:<9.2f}% {fer_f1:<10.4f}")
    print(f"{'CK+':<15s} {ck_acc*100:<9.2f}% {ck_f1:<10.4f}")

    # 保存完整双数据集结果到文件
    os.makedirs(r"D:\emotion_exp\exp_result", exist_ok=True)
    with open(r"D:\emotion_exp\exp_result\deep_results.txt", "w", encoding="utf-8") as f:
        f.write("==== FER2013 测试结果 ====\n")
        f.write(f"Accuracy: {fer_acc*100:.2f}%\n")
        f.write(f"F1-score: {fer_f1:.4f}\n\n")
        f.write("==== CK+ 测试结果 ====\n")
        f.write(f"Accuracy: {ck_acc*100:.2f}%\n")
        f.write(f"F1-score: {ck_f1:.4f}\n")
    print("\n全部评测结果已保存至 exp_result/deep_results.txt")

if __name__ == "__main__":
    main()