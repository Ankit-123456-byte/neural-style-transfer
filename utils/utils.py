import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path


_to_tensor = transforms.ToTensor()
_to_pil    = transforms.ToPILImage()


def load_image(path: str, image_size: int = 512, device: torch.device = None) -> torch.Tensor:
    """Load a single image as a (1, C, H, W) tensor."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, C, H, W) or (C, H, W) tensor in [0,1] to a PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1).cpu()
    return _to_pil(tensor)


def save_tensor(tensor: torch.Tensor, path: str):
    """Save a batch or single image tensor to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0, 1), str(path))


def save_checkpoint(state: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path: str, device: torch.device = None) -> dict:
    map_location = device if device is not None else "cpu"
    return torch.load(path, map_location=map_location)
