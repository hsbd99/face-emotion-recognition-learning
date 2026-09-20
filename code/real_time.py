import os
import cv2
import torch
import time
from mtcnn import MTCNN
from model import EmotionNet

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def main():
    detector = MTCNN()
    model = EmotionNet(num_classes=7)
    model_path = r"D:\emotion_exp\code\emotion_best.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        print("模型加载成功")
    else:
        print(f"模型文件不存在: {model_path}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 退出")
    
    fps_list = []
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_faces(frame)
        for result in results:
            x1, y1, w, h = result['box']
            x2, y2 = x1 + w, y1 + h
            
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                out, _ = model(tensor)
                idx = torch.argmax(out, dim=1).item()
                emo = EMOTIONS[idx]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emo, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        fps_list.append(fps)
        if len(fps_list) > 10:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()