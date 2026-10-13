"""Read-only integrity and inference checks for Alzheimer's MRI checkpoints.

Run from the project root:
  python verify_checkpoints.py
  python verify_checkpoints.py --image path/to/an_mri_image.jpg

Expected layout:
  alzheimer_research_portal_artifacts/resnet50_best.pth
  alzheimer_research_portal_artifacts/densenet121_best.pth
  alzheimer_research_portal_artifacts/efficientnet_b0_best.pth

This script does not train, save, or modify model weights.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

MODEL_FILES = {
    "resnet50": "resnet50_best.pth",
    "densenet121": "densenet121_best.pth",
    "efficientnet_b0": "efficientnet_b0_best.pth",
}
ARTIFACT_DIR_NAME = "alzheimer_research_portal_artifacts"
EXPECTED_CLASSES = {
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    """Build architecture without downloading pretrained weights."""
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Checkpoints in this project contain tensors and basic Python metadata only.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch versions
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary checkpoint: {path.name}")
    for key in ("model_name", "state_dict", "class_names"):
        if key not in payload:
            raise KeyError(f"{path.name} is missing required key: {key}")
    if not isinstance(payload["state_dict"], dict):
        raise TypeError(f"{path.name}: state_dict is not a dictionary")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(ARTIFACT_DIR_NAME),
        help=f"Artifact directory (default: ./{ARTIFACT_DIR_NAME})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional MRI image path. The same image is passed to all three models.",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("ALZHEIMER MRI CHECKPOINT VERIFICATION (READ-ONLY)")
    print("=" * 68)
    print(f"PyTorch:      {torch.__version__}")
    print(f"TorchVision:  {__import__('torchvision').__version__}")
    print(f"CUDA:         {torch.cuda.is_available()}")
    print(f"Device:       CPU (verification is intentionally run on CPU)")
    print(f"Artifacts:    {args.artifacts.resolve()}")

    results = []
    reference_classes = None
    had_error = False

    for name, filename in MODEL_FILES.items():
        path = args.artifacts / filename
        print("\n" + "-" * 68)
        print(f"Model: {name} | Checkpoint: {filename}")
        try:
            checkpoint = load_checkpoint(path)
            checkpoint_name = str(checkpoint["model_name"]).lower().replace("-", "_")
            aliases = {"efficientnetb0": "efficientnet_b0", "densenet_121": "densenet121"}
            checkpoint_name = aliases.get(checkpoint_name, checkpoint_name)
            if checkpoint_name != name:
                raise ValueError(
                    f"model_name mismatch: checkpoint says {checkpoint['model_name']!r}, expected {name!r}"
                )

            classes = checkpoint["class_names"]
            if not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) for c in classes):
                raise TypeError("class_names must be a list/tuple of strings")
            if len(classes) != 4 or set(classes) != EXPECTED_CLASSES:
                raise ValueError(f"Unexpected class_names: {classes!r}")
            if reference_classes is None:
                reference_classes = list(classes)
            elif list(classes) != reference_classes:
                raise ValueError(
                    f"Class order differs from previous model: {list(classes)!r} vs {reference_classes!r}"
                )

            model = build_model(name, num_classes=len(classes))
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
            # strict=True should already reject missing/unexpected keys.
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"State-dict mismatch: {incompatible}")
            model.eval()

            tensor_count = sum(1 for value in checkpoint["state_dict"].values() if torch.is_tensor(value))
            val_loss = checkpoint.get("best_validation_loss", "not recorded")
            epoch = checkpoint.get("best_epoch_in_history", "not recorded")
            print("PASS: strict state-dictionary loading")
            print(f"  Tensor entries: {tensor_count}")
            print(f"  Classes/order:  {list(classes)}")
            print(f"  Best val loss:  {val_loss}")
            print(f"  Best epoch:     {epoch}")

            results.append((name, model, list(classes)))
        except Exception as exc:
            had_error = True
            print(f"FAIL: {type(exc).__name__}: {exc}")

    if len(results) != len(MODEL_FILES):
        print("\nAt least one model failed integrity checks; inference comparison skipped.")
        return 1

    if args.image is not None:
        print("\n" + "=" * 68)
        print(f"INFERENCE SMOKE TEST: {args.image.resolve()}")
        print("=" * 68)
        try:
            with Image.open(args.image) as opened:
                image = opened.convert("RGB")
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            batch = preprocess(image).unsqueeze(0)
            print(f"Input: RGB, resized to 224x224; tensor shape={tuple(batch.shape)}")
            print("Normalization: ImageNet mean/std (must match the training pipeline).")
            with torch.inference_mode():
                for name, model, classes in results:
                    logits = model(batch)
                    if tuple(logits.shape) != (1, 4):
                        raise RuntimeError(f"{name}: expected logits shape (1, 4), got {tuple(logits.shape)}")
                    probs = torch.softmax(logits, dim=1)[0]
                    if not torch.isfinite(probs).all():
                        raise RuntimeError(f"{name}: probabilities contain NaN or infinity")
                    total = float(probs.sum().item())
                    if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                        raise RuntimeError(f"{name}: probabilities sum to {total}, not 1")
                    ranked = sorted(zip(classes, probs.tolist()), key=lambda pair: pair[1], reverse=True)
                    print(f"\n{name}: PASS | probability sum={total:.6f}")
                    for label, probability in ranked:
                        print(f"  {label:<20} {probability:.4f} ({probability * 100:6.2f}%)")
                    print(f"  Top class: {ranked[0][0]}")
            print("\nInference plumbing passed. These outputs are not a clinical diagnosis.")
        except (FileNotFoundError, UnidentifiedImageError, OSError, RuntimeError, ValueError) as exc:
            print(f"INFERENCE FAIL: {type(exc).__name__}: {exc}")
            had_error = True
    else:
        print("\nAll three checkpoints passed strict loading.")
        print("No image was supplied, so no MRI prediction was attempted.")
        print("Next: rerun with --image path/to/an_mri_image.jpg for a shared-input smoke test.")

    print("\n" + "=" * 68)
    print("FINAL STATUS:", "CHECKS FAILED" if had_error else "CHECKS PASSED")
    print("No training was performed; no checkpoint files were written or modified.")
    print("=" * 68)
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
