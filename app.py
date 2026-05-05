"""
Neural Style Transfer — Production Gradio App

Three modes:
  • Classic (Gatys)  — optimization-based; works with VGG only, quality preset controls steps
  • WCT              — Whitening and Coloring Transform; requires trained decoder
  • AdaIN (Fast)     — instant feed-forward style transfer; requires trained decoder

Post-processing pipeline (optional):
  • Histogram matching   — shift output color distribution toward style image
  • Color preservation   — keep content colors, apply only style luminance/texture
  • Sharpening           — unsharp mask to recover fine detail
"""
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import transforms

from model import VGGEncoder, Decoder, adain, wct
from post_process import apply_pipeline

# ─────────────────────────────────────────────────────────────────── config ──
VGG_PATH     = os.getenv("VGG_PATH",     "vgg_normalised.pth")
DECODER_PATH = os.getenv("DECODER_PATH", "experiment/experiment1/decoder_final.pth")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTENT_EXAMPLES = sorted(Path("content_data").glob("*.jpg"))[:4]
STYLE_EXAMPLES   = sorted(Path("style_data").glob("*.jpg"))[:4]

# Quality preset → optimization steps for Classic (Gatys) mode
QUALITY_STEPS = {
    "Draft (~15s)":    50,
    "Standard (~45s)": 150,
    "Premium (~90s)":  300,
}


# ──────────────────────────────────────────────────────────────── model load ──
class ModelRegistry:
    _encoder = None
    _decoder = None
    _vgg_ok  = False

    @classmethod
    def load_encoder(cls):
        if cls._vgg_ok:
            return True, ""
        if not Path(VGG_PATH).exists():
            return False, f"VGG weights not found at `{VGG_PATH}`."
        try:
            cls._encoder = VGGEncoder(VGG_PATH).to(DEVICE).eval()
            cls._vgg_ok  = True
            return True, ""
        except Exception as e:
            return False, f"Failed to load VGG: {e}"

    @classmethod
    def load_decoder(cls):
        if cls._decoder is not None:
            return True
        if not Path(DECODER_PATH).exists():
            return False
        try:
            dec = Decoder().to(DEVICE).eval()
            dec.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
            cls._decoder = dec
            return True
        except Exception:
            return False

    @classmethod
    def encoder(cls):
        return cls._encoder

    @classmethod
    def decoder(cls):
        return cls._decoder


# ─────────────────────────────────────────── image pre/post processing ──
def _preprocess(pil_img: Image.Image, size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)


def _postprocess(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.squeeze(0).clamp(0, 1).cpu())


# ─────────────────────────────────────────── MODE 1: fast AdaIN inference ──
@torch.no_grad()
def run_adain(content_img: Image.Image, style_img: Image.Image,
              alpha: float, image_size: int) -> Image.Image:
    c = _preprocess(content_img, image_size)
    s = _preprocess(style_img,   image_size)

    enc = ModelRegistry.encoder()
    cf  = enc(c)
    sf  = enc(s, return_intermediates=True)

    t = adain(cf, sf[-1])
    t = alpha * t + (1 - alpha) * cf
    return _postprocess(ModelRegistry.decoder()(t))


# ─────────────────────────────────────────── MODE 2: WCT inference ──
@torch.no_grad()
def run_wct(content_img: Image.Image, style_img: Image.Image,
            alpha: float, image_size: int) -> Image.Image:
    """
    Whitening and Coloring Transform style transfer.

    Applies WCT at relu4_1 (the deepest shared feature space the decoder
    accepts) and reconstructs with the trained decoder.  Falls back to AdaIN
    if the decoder is unavailable.
    """
    enc = ModelRegistry.encoder()
    c   = _preprocess(content_img, image_size)
    s   = _preprocess(style_img,   image_size)

    cf = enc(c)                              # relu4_1 features
    sf = enc(s, return_intermediates=True)   # (h1, h2, h3, h4, h5)

    # Apply WCT at relu4_1
    t = wct(cf, sf[3], alpha=alpha)

    dec = ModelRegistry.decoder()
    if dec is not None:
        return _postprocess(dec(t))

    # Fallback: no decoder available — use AdaIN result instead
    t_adain = adain(cf, sf[3])
    t_adain = alpha * t_adain + (1 - alpha) * cf
    return _postprocess(t_adain)


# ──────────────────────────────────────── MODE 3: classic Gatys optimization ──
def _gram(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def _tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Total Variation loss — penalises noise and high-frequency artifacts."""
    return (
        torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
        torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    )


def _optimize(enc, gen_init, content_targets, s_grams,
              style_weight, content_weight, tv_weight,
              layer_weights, steps):
    """Run LBFGS optimization for one scale."""
    gen = gen_init.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([gen], lr=1.0, max_iter=20)
    run = [0]

    while run[0] < steps:
        def closure():
            run[0] += 1
            opt.zero_grad()
            x = gen.clamp(0, 1)
            g_feats = enc(x, return_intermediates=True)

            # Content: relu3_1 + relu4_1 for structure + detail
            loss_c = content_weight * (
                0.5 * nn.functional.mse_loss(g_feats[2], content_targets[0]) +
                0.5 * nn.functional.mse_loss(g_feats[3], content_targets[1])
            )

            # Style: all five relu layers with graduated weights
            loss_s = 0.0
            for gf, sg, lw in zip(g_feats, s_grams, layer_weights):
                loss_s += style_weight * lw * nn.functional.mse_loss(_gram(gf), sg)

            # TV: keeps output smooth and artifact-free
            loss_tv = tv_weight * _tv_loss(x)

            loss = loss_c + loss_s + loss_tv
            loss.backward()
            return loss

        opt.step(closure)

    return gen.detach().clamp(0, 1)


def run_gatys(content_img: Image.Image, style_img: Image.Image,
              alpha: float, image_size: int, steps: int = 150) -> Image.Image:
    """
    Multi-scale LBFGS optimization with TV loss.
    Scale 1 (half-res): gets overall colors and composition right.
    Scale 2 (full-res): adds sharp fine details.
    """
    enc = ModelRegistry.encoder()

    half = max(image_size // 2, 128)
    scales = [half, image_size] if image_size > 256 else [image_size]

    # Pre-compute style Gram matrices at full style resolution
    s_full = _preprocess(style_img, image_size)
    with torch.no_grad():
        s_feats = enc(s_full, return_intermediates=True)
        s_grams = [_gram(f).detach() for f in s_feats]

    style_weight   = 1e7 * alpha
    content_weight = 5.0 * (1 - alpha + 0.1)
    tv_weight      = 5e-4

    # Graduated layer weights — early layers capture fine texture
    layer_weights = [1.0, 0.8, 0.5, 0.2, 0.1]

    gen = None
    for scale in scales:
        c = _preprocess(content_img, scale)
        with torch.no_grad():
            c_feats = enc(c, return_intermediates=True)
        content_targets = [c_feats[2].detach(), c_feats[3].detach()]  # relu3_1, relu4_1

        # Warm-start from upscaled previous result
        if gen is None:
            gen_init = c.clone()
        else:
            gen_init = nn.functional.interpolate(
                gen.unsqueeze(0) if gen.dim() == 3 else gen,
                size=(scale, scale), mode="bilinear", align_corners=False
            )

        # Split total steps across scales (60% low-res, 40% high-res)
        scale_steps = int(steps * 0.6) if scale == scales[0] and len(scales) > 1 else steps
        if scale == scales[-1] and len(scales) > 1:
            scale_steps = steps - int(steps * 0.6)

        gen = _optimize(enc, gen_init, content_targets, s_grams,
                        style_weight, content_weight, tv_weight,
                        layer_weights, scale_steps)

    return _postprocess(gen)


# ─────────────────────────────────────────────────────── gradio callbacks ──
def stylize(
    content_img,
    style_img,
    alpha,
    image_size,
    mode,
    quality,
    color_pres,
    hist_match,
    sharpen_str,
    progress=gr.Progress(),
):
    if content_img is None:
        raise gr.Error("Please upload a content image.")
    if style_img is None:
        raise gr.Error("Please upload a style image.")

    progress(0.05, desc="Loading VGG encoder...")
    ok, err = ModelRegistry.load_encoder()
    if not ok:
        raise gr.Error(err)

    has_decoder = ModelRegistry.load_decoder()
    size = int(image_size)
    t0   = time.time()

    if mode == "AdaIN (Fast)":
        if not has_decoder:
            raise gr.Error(
                "AdaIN (Fast) mode requires a trained decoder. "
                "Train one with train.py or switch to Classic mode."
            )
        progress(0.4, desc="Running fast AdaIN style transfer...")
        result = run_adain(content_img, style_img, alpha, size)
        mode_label = "AdaIN (Fast)"

    elif mode == "WCT":
        if not has_decoder:
            raise gr.Error(
                "WCT mode requires a trained decoder. "
                "Train one with train.py or switch to Classic mode."
            )
        progress(0.4, desc="Running WCT style transfer...")
        result = run_wct(content_img, style_img, alpha, size)
        mode_label = "WCT"

    else:  # Classic (Gatys)
        steps = QUALITY_STEPS.get(quality, 150)
        progress(0.2, desc=f"Running Classic (Gatys) optimization ({steps} steps)...")
        result = run_gatys(content_img, style_img, alpha, size, steps=steps)
        mode_label = f"Classic (Gatys) [{quality}]"

    # Post-processing pipeline
    if hist_match or color_pres or sharpen_str > 1.0:
        progress(0.92, desc="Applying post-processing...")
        result = apply_pipeline(
            result,
            content_img,
            style_img,
            do_color_preserve=color_pres,
            do_histogram_match=hist_match,
            sharpen_strength=sharpen_str,
        )

    elapsed = time.time() - t0
    progress(1.0, desc="Done!")
    info = (
        f"Mode: {mode_label}  |  "
        f"Done in {elapsed:.1f}s  |  "
        f"Device: {DEVICE}  |  "
        f"Size: {size}px"
    )
    return result, gr.update(value=info, visible=True)


def get_status():
    enc_ok, _ = ModelRegistry.load_encoder()
    dec_ok    = Path(DECODER_PATH).exists()
    if not enc_ok:
        msg = "VGG weights missing — run convert_vgg.py first"
    elif not dec_ok:
        msg = (
            "No trained decoder found. "
            "Classic (Gatys) mode available. "
            "Train a decoder for AdaIN and WCT modes."
        )
    else:
        msg = "All modes ready: Classic (Gatys), WCT, and AdaIN (Fast)."
    return gr.update(value=msg, visible=True)


def _toggle_quality_visibility(mode):
    """Show the quality preset only for Classic (Gatys) mode."""
    return gr.update(visible=(mode == "Classic (Gatys)"))


# ──────────────────────────────────────────────────────────────── gradio UI ──
CSS = """
#title    { text-align: center; }
#subtitle { text-align: center; color: #888; margin-top: -8px; }
#status   { font-size: 0.82em; }
footer    { display: none !important; }
"""

def build_ui():
    with gr.Blocks(title="Neural Style Transfer") as demo:

        gr.Markdown("# Neural Style Transfer", elem_id="title")
        gr.Markdown(
            "Combine the **content** of one image with the **style** of another.",
            elem_id="subtitle",
        )

        status_box = gr.Textbox(
            label="Status", interactive=False, visible=False, elem_id="status"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                content_input = gr.Image(label="Content Image", type="pil", height=270)
                style_input   = gr.Image(label="Style Image",   type="pil", height=270)

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_image = gr.Image(label="Stylized Image", type="pil", height=560)

        with gr.Row():
            with gr.Column(scale=2):
                alpha_slider = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Style Strength",
                    info="0 = content only  |  1 = full style",
                )
            with gr.Column(scale=1):
                size_drop = gr.Dropdown(
                    choices=[256, 384, 512, 768, 1024], value=512,
                    label="Output Size (px)",
                    info="512+ recommended for best quality",
                )

        with gr.Row():
            with gr.Column(scale=1):
                mode_radio = gr.Radio(
                    ["Classic (Gatys)", "WCT", "AdaIN (Fast)"],
                    value="Classic (Gatys)",
                    label="Transfer Mode",
                    info=(
                        "Classic: optimization-based, no decoder needed.  "
                        "WCT/AdaIN: instant, requires trained decoder."
                    ),
                )
            with gr.Column(scale=1):
                quality_radio = gr.Radio(
                    ["Draft (~15s)", "Standard (~45s)", "Premium (~90s)"],
                    value="Standard (~45s)",
                    label="Quality Preset",
                    info="Controls optimization steps (Classic mode only)",
                    visible=True,
                )

        with gr.Row():
            with gr.Column(scale=1):
                color_preserve_cb = gr.Checkbox(
                    label="Preserve Content Colors",
                    value=False,
                    info="Keep content palette; apply only style luminance/texture",
                )
                histogram_match_cb = gr.Checkbox(
                    label="Histogram Matching",
                    value=False,
                    info="Match output color distribution to style image",
                )
            with gr.Column(scale=1):
                sharpen_slider = gr.Slider(
                    1.0, 3.0, value=1.0, step=0.1,
                    label="Sharpening",
                    info="1.0 = no sharpening  |  higher = stronger unsharp mask",
                )

        with gr.Row():
            run_btn   = gr.Button("Stylize", variant="primary", scale=3)
            clear_btn = gr.Button("Clear", scale=1)

        info_text = gr.Textbox(label="Info", interactive=False, visible=False)

        if CONTENT_EXAMPLES and STYLE_EXAMPLES:
            gr.Markdown("### Examples")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Content**")
                    gr.Examples([[str(p)] for p in CONTENT_EXAMPLES], inputs=content_input, label="")
                with gr.Column():
                    gr.Markdown("**Style**")
                    gr.Examples([[str(p)] for p in STYLE_EXAMPLES], inputs=style_input, label="")

        with gr.Accordion("How it works", open=False):
            gr.Markdown("""
**Classic mode (Gatys)** — works right now, no decoder needed:
1. VGG encodes both images to extract content and style features
2. Content loss + Gram-matrix style loss guide pixel optimization
3. Quality preset controls step count (Draft / Standard / Premium)

**WCT mode** — requires a trained decoder:
1. VGG encodes both images
2. Whitening and Coloring Transform matches full style covariance (richer than AdaIN)
3. Trained decoder reconstructs the stylized image

**AdaIN (Fast)** — requires a trained decoder, instant results:
1. VGG encodes both images
2. AdaIN transfers per-channel mean & std from style to content features
3. Trained decoder reconstructs the stylized image

**Alpha slider** controls style strength in all modes.

**Post-processing options:**
- **Histogram Matching** shifts output color distribution toward the style image
- **Preserve Content Colors** retains content chrominance, applies only style luminance
- **Sharpening** applies unsharp mask to recover fine detail lost during style transfer
""")

        # ---------------------------------------------------------------- wiring
        mode_radio.change(
            fn=_toggle_quality_visibility,
            inputs=[mode_radio],
            outputs=[quality_radio],
        )

        run_btn.click(
            fn=stylize,
            inputs=[
                content_input,
                style_input,
                alpha_slider,
                size_drop,
                mode_radio,
                quality_radio,
                color_preserve_cb,
                histogram_match_cb,
                sharpen_slider,
            ],
            outputs=[output_image, info_text],
        )

        clear_btn.click(
            fn=lambda: (None, None, None, gr.update(visible=False)),
            outputs=[content_input, style_input, output_image, info_text],
        )

        demo.load(fn=get_status, outputs=status_box)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
    )
