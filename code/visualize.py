"""Generate Grad-CAM explanations and optional feature-map grids."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import EMOTIONS, OUTPUT_DIR
from data import FERDataset, get_transforms, resolve_fer_root
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "visualization")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-maps", action="store_true")
    return parser.parse_args()


class GradCAM:
    """Minimal Grad-CAM implementation for any convolutional classifier."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        score = logits[0, class_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not receive activations/gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def make_overlay(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(0.55 * heatmap + 0.45 * image_rgb)


def save_gradcam_figure(original: np.ndarray, cam: np.ndarray, overlay: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_feature_maps(model, image: torch.Tensor, path: Path, max_channels: int = 16) -> None:
    with torch.no_grad():
        _, feature_map = model(image, return_features=True)
    features = feature_map[0].cpu().numpy()
    channels = min(max_channels, features.shape[0])
    cols = 4
    rows = int(np.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    for index in range(rows * cols):
        axis = axes[index // cols, index % cols]
        if index < channels:
            axis.imshow(features[index], cmap="viridis")
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent.parent / "exp_result" / "checkpoints" / "emotion_best.pth"
    model, metadata = load_checkpoint(checkpoint, device=device)
    image_size = int(metadata.get("image_size", 96))
    transform = get_transforms(image_size, train=False)
    dataset = FERDataset(resolve_fer_root(args.data_dir), split=args.split, transform=None)

    samples_by_class: dict[str, list[Path]] = {name: [] for name in EMOTIONS}
    for path, label in dataset.samples:
        name = EMOTIONS[label]
        if len(samples_by_class[name]) < args.samples_per_class:
            samples_by_class[name].append(path)

    gradcam = GradCAM(model, model.get_gradcam_layer())
    try:
        for emotion, paths in samples_by_class.items():
            for index, path in enumerate(paths):
                with Image.open(path) as pil_image:
                    rgb = np.asarray(pil_image.convert("RGB").resize((image_size, image_size)))
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
                cam = gradcam(tensor)
                cam_resized = cv2.resize(cam, (image_size, image_size))
                overlay = make_overlay(rgb, cam_resized)
                output_path = args.output_dir / f"gradcam_{emotion}_{index + 1}.png"
                save_gradcam_figure(rgb, cam_resized, overlay, output_path, emotion)
                if args.feature_maps:
                    save_feature_maps(model, tensor, args.output_dir / "features" / f"{emotion}_features.png")
                print(f"Saved {output_path}")
    finally:
        gradcam.close()


if __name__ == "__main__":
    main()
