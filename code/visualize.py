import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import EmotionNet

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def visualize_features(features, save_dir, img_name):
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_idx, feat in enumerate(features):
        num_channels = feat.shape[1]
        cols = 8
        rows = min(8, num_channels // cols)
        if rows == 0:
            rows = 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(rows * cols):
            if i < num_channels:
                ax = axes[i // cols, i % cols]
                channel_data = feat[0, i].cpu().detach().numpy()
                ax.imshow(channel_data, cmap='viridis')
                ax.axis('off')
            else:
                axes[i // cols, i % cols].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{img_name}_layer{layer_idx+1}.png"))
        plt.close()

def generate_cam(model, img_tensor):
    model.eval()
    with torch.no_grad():
        features, last_feat = model.get_features(img_tensor)
    
    last_feat = last_feat[0].cpu().detach().numpy()
    cam = np.mean(last_feat, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (48, 48))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def visualize_cam(cam, original_img, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    original_img = original_img.astype(np.float32) / 255.0
    heatmap = heatmap.astype(np.float32) / 255.0
    
    overlay = 0.6 * heatmap + 0.4 * original_img
    overlay = np.uint8(overlay * 255)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(cam, cmap='jet')
    ax2.set_title("Attention Map")
    ax2.axis('off')
    
    ax3.imshow(overlay)
    ax3.set_title("Attention Overlay")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    model = EmotionNet(num_classes=7)
    model_path = r"D:\emotion_exp\code\emotion_best.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print("模型加载成功")
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    save_dir = r"D:\emotion_exp\exp_result\visualization"
    os.makedirs(save_dir, exist_ok=True)
    
    sample_dir = r"D:\emotion_exp\dataset\FER2013_aligned\val"
    for emo in EMOTIONS:
        emo_dir = os.path.join(sample_dir, emo)
        if os.path.exists(emo_dir):
            img_list = os.listdir(emo_dir)
            if img_list:
                sample_img_path = os.path.join(emo_dir, img_list[0])
                img = cv2.imread(sample_img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (48, 48))
                    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        output, _ = model(img_tensor)
                        pred_idx = torch.argmax(output, dim=1).item()
                    
                    features, _ = model.get_features(img_tensor)
                    visualize_features(features, os.path.join(save_dir, "features"), f"{emo}_features")
                    
                    cam = generate_cam(model, img_tensor)
                    visualize_cam(cam, img_resized, os.path.join(save_dir, f"{emo}_attention.png"))
                    
                    print(f"可视化完成: {emo} → 预测: {EMOTIONS[pred_idx]}")
    
    print(f"\n所有可视化结果已保存到: {save_dir}")

if __name__ == "__main__":
    main()