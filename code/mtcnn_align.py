import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from mtcnn import MTCNN


root_dir = r"D:\emotion_exp\dataset\FER2013"
save_root = r"D:\emotion_exp\dataset\FER2013_after"

# 新版MTCNN支持的参数，调低检测阈值适配极小人脸
detector = MTCNN(min_face_size=8, steps_threshold=[0.4, 0.4, 0.4])

# 创建输出文件夹结构
splits = ["train", "val", "test"]
emotions = ["angry","disgust","fear","happy","sad","surprise","neutral"]
for split in splits:
    for emo in emotions:
        os.makedirs(os.path.join(save_root, split, emo), exist_ok=True)

def align_face(img):
    # 灰度图转3通道BGR，解决MTCNN无法识别灰度图
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 原图48×48过小，放大至96×96再检测，消除张量shape不匹配警告
    img_scaled = cv2.resize(img, (96, 96))
    results = detector.detect_faces(img_scaled)

    # 完全检测不到人脸，直接返回原图缩放结果，不会丢样本
    if len(results) == 0:
        return cv2.resize(img, (48, 48))
    
    # 缩放坐标还原回原图尺寸
    scale_ratio = 48 / 96
    face_info = results[0]
    x, y, w, h = face_info["box"]
    x = int(x * scale_ratio)
    y = int(y * scale_ratio)
    w = int(w * scale_ratio)
    h = int(h * scale_ratio)

    # 还原五官关键点坐标
    kpts = {}
    for name, (px, py) in face_info["keypoints"].items():
        kpts[name] = (int(px * scale_ratio), int(py * scale_ratio))

    left_eye = np.array(kpts["left_eye"])
    right_eye = np.array(kpts["right_eye"])
    eye_center = (left_eye + right_eye) / 2

    # 旋转矫正
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    h_img, w_img = img.shape[:2]
    rotate_mat = cv2.getRotationMatrix2D(eye_center, angle, scale=1.1)
    aligned_img = cv2.warpAffine(img, rotate_mat, (w_img, h_img))

    # 裁剪人脸区域
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x1 + w, w_img)
    y2 = min(y1 + h, h_img)
    crop_face = aligned_img[y1:y2, x1:x2]
    # 统一输出48×48
    crop_face = cv2.resize(crop_face, (48, 48))
    return crop_face

# 批量处理所有图片
total_save = 0
for split in splits:
    for emo in emotions:
        src_dir = os.path.join(root_dir, split, emo)
        dst_dir = os.path.join(save_root, split, emo)
        img_name_list = os.listdir(src_dir)
        for name in img_name_list:
            src_path = os.path.join(src_dir, name)
            img = cv2.imread(src_path)
            if img is None:
                continue
            aligned_out = align_face(img)
            cv2.imwrite(os.path.join(dst_dir, name), aligned_out)
            total_save += 1
print(f"对齐完成，成功保存图片总数：{total_save}")