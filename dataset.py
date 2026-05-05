import random
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # skip corrupt/truncated files gracefully
Image.MAX_IMAGE_PIXELS = None           # disable decompression bomb limit for large style images


def build_transform(image_size: int = 512, augment: bool = True):
    """Build an image transform pipeline with optional augmentation.

    When `augment=True` a richer set of spatial and color transforms is applied
    to improve generalisation during training.  When `augment=False` a clean
    center-crop is used (suitable for validation / inference).
    """
    ops = [transforms.Resize(int(image_size * 1.1))]
    if augment:
        ops += [
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    else:
        ops.append(transforms.CenterCrop(image_size))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class StyleTransferDataset(Dataset):
    """
    Yields (content_image, style_image) pairs.

    Content images are iterated in order; style images are sampled randomly so
    every content/style combination can appear during training.
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, content_dir: str, style_dir: str,
                 image_size: int = 512, augment: bool = True):
        self.content_paths = self._gather(content_dir)
        self.style_paths   = self._gather(style_dir)
        if not self.content_paths:
            raise FileNotFoundError(f"No images found in {content_dir}")
        if not self.style_paths:
            raise FileNotFoundError(f"No images found in {style_dir}")
        self.transform = build_transform(image_size, augment=augment)

    MAX_FILE_BYTES = 5 * 1024 * 1024  # skip images > 5 MB (huge files stall PIL)

    def _gather(self, directory: str):
        p = Path(directory)
        return sorted(
            f for f in p.rglob("*")
            if f.is_file()
            and f.suffix.lower() in self.EXTENSIONS
            and f.stat().st_size <= self.MAX_FILE_BYTES
        )

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content = Image.open(self.content_paths[idx]).convert("RGB")
        # Sample style randomly so every content image sees varied styles
        style_idx = random.randint(0, len(self.style_paths) - 1)
        style = Image.open(self.style_paths[style_idx]).convert("RGB")
        return self.transform(content), self.transform(style)
