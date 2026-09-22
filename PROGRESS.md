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

## 2026-09-21

### Completed

- Added resume support and configurable class weighting to `train.py`.
- Added a lightweight `fer_cnn` architecture specialized for 48x48 FER2013 inputs.
- Added MixUp, RandomErasing, and a cosine learning-rate schedule.

### Experiments

- `mobilenet_v3_small` transfer learning at 96x96 still overfit or plateaued near 27-31% validation accuracy in the first few epochs.
- `fer_cnn` without class weighting reached a best validation accuracy of `33.40%` and test accuracy `35.01%`.
  - The confusion matrix showed very low recall for `fear`, `sad`, `disgust`, and `angry`.
  - This indicates class imbalance is a major remaining problem.
- A new class-weighted `fer_cnn` run was started with `class-weight-power 0.5`, MixUp `0.1`, cosine schedule, and was stopped after epoch 2 at the user's request.

### Current checkpoint

The current `exp_result/checkpoints/emotion_best.pth` is from the stopped class-weighted run and is not yet representative. Do not use it as a final result.

### Next tasks

1. Resume the class-weighted `fer_cnn` run and let it train until early stopping.
2. Evaluate the best checkpoint with `judge.py` and compare accuracy plus macro F1.
3. Decide whether to keep `fer_cnn` or return to a pretrained backbone with a longer linear-probe/fine-tune schedule.
4. Run the HOG + SVM baseline and `compare.py`.
5. Smoke-test the real-time system with camera, image, and video inputs.
6. Finalize the README results table and push the completed history.

## 2026-09-22

### Progress

- Fixed a critical dimension bug in the MobileNet classification head caused by an earlier global find/replace (`nn.Linear(256, num_classes)` had been changed to `128`). Commit `ce8be99`.
- Retrained MobileNetV3-Small with the fixed classifier. Best validation accuracy remained around `32.47%`.
- Started a ResNet18 transfer-learning experiment and stopped it at epoch 5 at the user's request. Validation accuracy at that point was `30.87%`, with no substantial improvement over previous runs.

### Conclusion so far

All current transfer-learning and from-scratch runs plateau around `30-36%`. This is close to the original project baseline, so the bottleneck is likely not only model architecture. The next session should focus on:

1. Data quality and label inspection (spot-check FER2013_aligned samples and label distribution).
2. A small controlled experiment with one simple model to isolate whether the pipeline, augmentation, or evaluation is limiting accuracy.
3. Comparing against the original raw checkpoint as a legacy reference.
4. If needed, introducing stronger per-class balancing without over-oversampling, or using a face-specific pretrained model.

Current checkpoint is not a final result.
