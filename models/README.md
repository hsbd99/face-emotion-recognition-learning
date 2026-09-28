# Model files

## YuNet face detector

- File: `face_detection_yunet_2023mar.onnx`
- Source: [OpenCV Zoo face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- License: Apache-2.0
- Purpose: fast CPU-friendly face detection for the real-time demo.

## Facial expression recognition model

- File: `facial_expression_recognition_mobilefacenet_2022july.onnx`
- Source: [OpenCV Zoo facial expression recognition](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition)
- License: Apache-2.0
- Backbone: MobileFaceNet
- Purpose: default pretrained expression classifier for real-time inference.
- Official RAF-DB accuracy: 88.27%
- This project's FER2013 test accuracy: 45.25%
- This project's CK+ accuracy: 59.90%

The custom-trained checkpoint is generated locally by `code/train.py` and is intentionally not stored in Git.
