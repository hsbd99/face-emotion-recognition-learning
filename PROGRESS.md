# Progress Notes

## 2026-09-20

### Completed

- Refactored the training pipeline into reusable `config.py`, `data.py`, `model.py`, and `train.py` modules.
- Added MobileNetV3-Small transfer learning, data augmentation, balanced sampling, label smoothing, early stopping, and checkpoint metadata.
- Added full evaluation metrics, classification reports, and confusion-matrix output.
- Added Grad-CAM visualization.
- Reworked real-time inference to support YuNet, MTCNN/Haar fallback, camera/video/image input, prediction smoothing, FPS display, and output files.
- Added the lightweight YuNet ONNX face detection model and license notes.
- Updated the README documentation structure.

### Current checkpoint

- Best validation accuracy: `31.10%`
- Completed epochs: `8 / 25`
- Local checkpoint: `exp_result/checkpoints/emotion_best.pth`
- Last checkpoint: `exp_result/checkpoints/emotion_last.pth`
- Training history: `exp_result/checkpoints/training_history.json`

The checkpoint files are retained locally but ignored by Git because `*.pth` is in `.gitignore`.

### Observation

The first eight epochs showed a clear train/validation gap:

- Train accuracy rose to about `65%`
- Validation accuracy plateaued around `31%`

This indicates overfitting or an overly aggressive balanced-sampling strategy. Before continuing from the current checkpoint, the next training run should evaluate:

- Disabling or reducing `WeightedRandomSampler`
- Using a lighter class-weight strategy
- Reducing augmentation strength
- Increasing weight decay
- Training only the classifier for more epochs before unfreezing the backbone
- Comparing MobileNetV3-Small with ResNet18

### Next tasks

1. Add resume support to `train.py`.
2. Run an improved training configuration and compare validation macro F1 as well as accuracy.
3. Evaluate the best checkpoint on FER2013 and CK+ in `judge.py`.
4. Run `HOGSVM.py` and `compare.py`.
5. Update the README results table.
6. Smoke-test `real_time.py` with a webcam, image, and video.
7. Push the complete local commit history to GitHub.
