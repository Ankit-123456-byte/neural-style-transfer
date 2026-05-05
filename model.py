import torch
import torch.nn as nn


# VGG architecture with reflection padding and no BN (matches vgg_normalised.pth)
# Extended to relu5_1 for WCT and richer style loss computation.
vgg_arch = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),                   # 0  — normalisation conv
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 1
    nn.Conv2d(3, 64, (3, 3)),                   # 2
    nn.ReLU(),                                  # 3  relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 4
    nn.Conv2d(64, 64, (3, 3)),                  # 5
    nn.ReLU(),                                  # 6  relu1_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 7
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 8
    nn.Conv2d(64, 128, (3, 3)),                 # 9
    nn.ReLU(),                                  # 10 relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 11
    nn.Conv2d(128, 128, (3, 3)),                # 12
    nn.ReLU(),                                  # 13 relu2_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 14
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 15
    nn.Conv2d(128, 256, (3, 3)),                # 16
    nn.ReLU(),                                  # 17 relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 18
    nn.Conv2d(256, 256, (3, 3)),                # 19
    nn.ReLU(),                                  # 20
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 21
    nn.Conv2d(256, 256, (3, 3)),                # 22
    nn.ReLU(),                                  # 23
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 24
    nn.Conv2d(256, 256, (3, 3)),                # 25
    nn.ReLU(),                                  # 26
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 27
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 28
    nn.Conv2d(256, 512, (3, 3)),                # 29
    nn.ReLU(),                                  # 30 relu4_1
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 31
    nn.Conv2d(512, 512, (3, 3)),                # 32
    nn.ReLU(),                                  # 33 relu4_2
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 34
    nn.Conv2d(512, 512, (3, 3)),                # 35
    nn.ReLU(),                                  # 36 relu4_3
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 37
    nn.Conv2d(512, 512, (3, 3)),                # 38
    nn.ReLU(),                                  # 39 relu4_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 40
    nn.ReflectionPad2d((1, 1, 1, 1)),           # 41
    nn.Conv2d(512, 512, (3, 3)),                # 42
    nn.ReLU(),                                  # 43 relu5_1
)

STYLE_LAYERS = [3, 10, 17, 30, 43]   # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
CONTENT_LAYER = 30                    # relu4_1


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path: str):
        super().__init__()
        vgg = vgg_arch
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))
        # Freeze encoder — only the decoder is trained
        for p in vgg.parameters():
            p.requires_grad = False
        # Store slices so we can grab intermediate outputs cheaply
        self.slice1 = vgg[:4]    # up to relu1_1
        self.slice2 = vgg[4:11]  # relu1_1 → relu2_1
        self.slice3 = vgg[11:18] # relu2_1 → relu3_1
        self.slice4 = vgg[18:31] # relu3_1 → relu4_1
        self.slice5 = vgg[31:44] # relu4_1 → relu5_1

    def forward(self, x, return_intermediates: bool = False):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if return_intermediates:
            h5 = self.slice5(h4)
            return h1, h2, h3, h4, h5
        return h4


class Decoder(nn.Module):
    """Mirror of VGG encoder, trained from scratch."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        return self.net(x)


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Adaptive Instance Normalization: transfer style statistics to content."""
    assert content_feat.shape[:2] == style_feat.shape[:2], \
        "content and style must have same batch size and channels"
    size = content_feat.shape

    # Compute per-channel mean and std over spatial dims
    style_mean = style_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
    style_std  = style_feat.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-5

    c_mean = content_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
    c_std  = content_feat.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-5

    normalized = (content_feat - c_mean) / c_std
    return normalized * style_std + style_mean


def wct(content_feat: torch.Tensor, style_feat: torch.Tensor,
        alpha: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Whitening and Coloring Transform (WCT).

    Matches the full covariance structure of the style features, not just the
    per-channel mean and std like AdaIN.  This produces richer, more accurate
    style transfer at the cost of an SVD computation per call.
    Processes each batch element independently (SVD is not batchable).
    """
    b, c, h, w = content_feat.shape
    results = []
    for i in range(b):
        cf = content_feat[i].view(c, h * w)   # (c, hw)
        sf = style_feat[i].view(c, -1)         # (c, hw)

        # --- Whiten content features ---
        c_mean = cf.mean(dim=1, keepdim=True)
        cf_c   = cf - c_mean
        c_cov  = cf_c @ cf_c.t() / (h * w - 1) + eps * torch.eye(c, device=cf.device)
        Uc, Sc, _ = torch.linalg.svd(c_cov)
        Dc = torch.diag(Sc.clamp(min=0).sqrt().reciprocal().clamp(max=1e4))
        whitened = Uc @ Dc @ Uc.t() @ cf_c    # full whitening: U Λ^{-1/2} Uᵀ x

        # --- Color with style covariance ---
        s_mean = sf.mean(dim=1, keepdim=True)
        sf_c   = sf - s_mean
        s_cov  = sf_c @ sf_c.t() / (sf.shape[1] - 1) + eps * torch.eye(c, device=sf.device)
        Us, Ss, _ = torch.linalg.svd(s_cov)
        Ds = torch.diag(Ss.clamp(min=0).sqrt())
        colored = Us @ Ds @ Us.t() @ whitened + s_mean

        result = alpha * colored + (1 - alpha) * cf
        results.append(result.view(c, h, w))

    return torch.stack(results)  # (b, c, h, w)


class NSTModel(nn.Module):
    def __init__(self, encoder: VGGEncoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # ------------------------------------------------------------------ losses
    def _calc_content_loss(self, generated_feat: torch.Tensor, target: torch.Tensor):
        return nn.functional.mse_loss(generated_feat, target)

    def _calc_style_loss(self, gen_feats, style_feats):
        loss = 0.0
        for gf, sf in zip(gen_feats, style_feats):
            gm, gs = self._mean_std(gf)
            sm, ss = self._mean_std(sf)
            loss += nn.functional.mse_loss(gm, sm) + nn.functional.mse_loss(gs, ss)
        return loss

    @staticmethod
    def _mean_std(feat: torch.Tensor):
        b, c = feat.shape[:2]
        flat = feat.view(b, c, -1)
        mean = flat.mean(dim=2).view(b, c, 1, 1)
        std  = flat.std(dim=2).view(b, c, 1, 1) + 1e-5
        return mean, std

    # ---------------------------------------------------------------- forward
    def forward(self, content: torch.Tensor, style: torch.Tensor,
                alpha: float = 1.0, content_weight: float = 1.0,
                style_weight: float = 10.0, mode: str = "adain"):
        """
        Args:
            mode: 'adain' — fast per-channel statistics matching (default)
                  'wct'   — full covariance matching via SVD (richer, slower)
        Returns:
            generated  – stylized image tensor
            loss_c     – content loss
            loss_s     – style loss
        """
        style_feats  = self.encoder(style,   return_intermediates=True)
        content_feat = self.encoder(content, return_intermediates=False)  # h4 only

        if mode == "adain":
            t = adain(content_feat, style_feats[3])
            t = alpha * t + (1 - alpha) * content_feat
        elif mode == "wct":
            t = wct(content_feat, style_feats[3], alpha=alpha)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'adain' or 'wct'.")

        generated = self.decoder(t)

        gen_feats = self.encoder(generated, return_intermediates=True)

        loss_c = self._calc_content_loss(gen_feats[3], t)
        loss_s = self._calc_style_loss(gen_feats, style_feats)

        return generated, content_weight * loss_c, style_weight * loss_s
