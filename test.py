"""
Inference script — stylize a single content image with a single style image.

Usage:
    python test.py \
        --content content_data/000000000069.jpg \
        --style   style_data/30.jpg \
        --vgg     vgg_normalised.pth \
        --decoder experiment/experiment1/decoder_final.pth \
        --output  output.jpg \
        --alpha   1.0
"""
import argparse
import torch

from model import VGGEncoder, Decoder, adain
from utils.utils import load_image, save_tensor


def parse_arguments():
    parser = argparse.ArgumentParser(description="AdaIN Neural Style Transfer — Inference")
    parser.add_argument("--content",  type=str, required=True)
    parser.add_argument("--style",    type=str, required=True)
    parser.add_argument("--vgg",      type=str, default="vgg_normalised.pth")
    parser.add_argument("--decoder",  type=str, required=True,
                        help="Path to trained decoder weights")
    parser.add_argument("--output",   type=str, default="output.jpg")
    parser.add_argument("--alpha",    type=float, default=1.0,
                        help="Style strength 0→1 (0=content only, 1=full style)")
    parser.add_argument("--image_size", type=int, default=512)
    return parser.parse_args()


@torch.no_grad()
def stylize():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = VGGEncoder(args.vgg).to(device).eval()
    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))

    content = load_image(args.content, args.image_size, device)
    style   = load_image(args.style,   args.image_size, device)

    content_feat = encoder(content)
    style_feat   = encoder(style)

    t = adain(content_feat, style_feat)
    t = args.alpha * t + (1 - args.alpha) * content_feat

    generated = decoder(t)
    save_tensor(generated, args.output)
    print(f"Stylized image saved to: {args.output}")


if __name__ == "__main__":
    stylize()
