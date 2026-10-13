import torch
import torchvision
from pathlib import Path

root = Path("alzheimer_research_portal_artifacts")

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print("\n" + "=" * 60)
print("CHECKPOINT INSPECTION")
print("=" * 60)

for name in [
    "resnet50_best.pth",
    "densenet121_best.pth",
    "efficientnet_b0_best.pth",
]:
    path = root / name
    print(f"\n{name}")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(checkpoint, dict):
            print("Top-level keys:", list(checkpoint.keys())[:15])

            # Support either a raw state_dict or a wrapped checkpoint.
            state = checkpoint
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    break

            tensors = [
                (key, value)
                for key, value in state.items()
                if isinstance(value, torch.Tensor)
            ]

            print("Tensor count:", len(tensors))
            print("First parameter names and shapes:")
            for key, value in tensors[:5]:
                print(f"  {key}: {tuple(value.shape)}")
        else:
            print("Unexpected checkpoint type:", type(checkpoint))

    except Exception as exc:
        print("Inspection error:", type(exc).__name__, str(exc))

print("\nInspection complete. No weights were changed.")