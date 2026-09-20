import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

EMOTION_MAP = {"angry":0, "disgust":1, "fear":2, "happy":3, "sad":4, "surprise":5, "neutral":6}

def load_data(root, split="train"):
    features, labels = [], []
    for emo, label in EMOTION_MAP.items():
        emo_dir = os.path.join(root, split, emo)
        if os.path.exists(emo_dir):
            for img_name in os.listdir(emo_dir):
                img = cv2.imread(os.path.join(emo_dir, img_name))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hog = cv2.HOGDescriptor((48,48), (16,16), (8,8), (8,8), 9)
                features.append(hog.compute(gray).flatten())
                labels.append(label)
    return np.array(features), np.array(labels)

def main():
    print("HOG+SVM训练中...")
    root = r"D:\emotion_exp\dataset\FER2013_aligned"
    X_train, y_train = load_data(root, "train")
    X_test, y_test = load_data(root, "test")
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nHOG+SVM结果:")
    print(f"准确率: {acc*100:.2f}%")
    print(f"F1-score: {f1:.4f}")

    with open(r"D:\emotion_exp\exp_result\hog_svm_results.txt", "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"F1-score: {f1:.4f}\n")
    print("结果已保存")

if __name__ == "__main__":
    main()