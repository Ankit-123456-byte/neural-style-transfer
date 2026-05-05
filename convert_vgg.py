"""
Build vgg_normalised.pth from torchvision's pretrained VGG19.

The normalised VGG embeds ImageNet mean/std subtraction as a learned 1x1 conv
so the rest of the pipeline can feed raw [0,1] tensors without a separate
transforms.Normalize step.

Extended to include conv4_2 through conv5_1 to support WCT and richer style
loss computation up to relu5_1.
"""
import torch
import torch.nn as nn
from torchvision import models
from model import vgg_arch

# ImageNet stats
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

def build_norm_conv():
    """1x1 conv that maps [0,1] pixels → ImageNet-normalised space."""
    conv = nn.Conv2d(3, 3, (1, 1))
    conv.weight.data = torch.diag(1.0 / STD).view(3, 3, 1, 1)
    conv.bias.data   = -MEAN / STD
    return conv

def convert():
    print("Loading pretrained VGG19 from torchvision...")
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

    # VGG19 conv layer indices → our vgg_arch conv indices
    vgg19_to_ours = {
        0:  2,   # conv1_1
        2:  5,   # conv1_2
        5:  9,   # conv2_1
        7:  12,  # conv2_2
        10: 16,  # conv3_1
        12: 19,  # conv3_2
        14: 22,  # conv3_3
        16: 25,  # conv3_4
        19: 29,  # conv4_1
        21: 32,  # conv4_2
        23: 35,  # conv4_3
        25: 38,  # conv4_4
        28: 42,  # conv5_1
    }

    state = vgg_arch.state_dict()

    # Fill normalisation conv (index 0)
    norm = build_norm_conv()
    state["0.weight"] = norm.weight.data
    state["0.bias"]   = norm.bias.data

    # Copy VGG19 conv weights into our architecture
    for vgg_idx, our_idx in vgg19_to_ours.items():
        state[f"{our_idx}.weight"] = vgg19[vgg_idx].weight.data.clone()
        state[f"{our_idx}.bias"]   = vgg19[vgg_idx].bias.data.clone()

    vgg_arch.load_state_dict(state)
    torch.save(state, "vgg_normalised.pth")
    print("Saved vgg_normalised.pth")

if __name__ == "__main__":
    convert()
