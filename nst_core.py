"""
nst_core.py
───────────
Your original NST code (neural_style_transfer_colab.py) — ZERO logic changes.
One addition only: run_job() entry-point + optional progress_cb hook so the
FastAPI layer can stream progress. Every algorithm, loss, layer, optimizer
decision is identical to your Colab notebook.
"""

import os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Image utilities  (identical to CELL 3)
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path: str, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    new_size = (size // 8) * 8
    tfm = transforms.Compose([
        transforms.Resize((new_size, new_size),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return tfm(img).unsqueeze(0).to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1,3,1,1)
    img  = (tensor * std + mean).squeeze(0).clamp(0,1)
    img  = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    return Image.fromarray(img)


def save_tensor(tensor: torch.Tensor, path: str) -> None:
    tensor_to_pil(tensor).save(path, quality=95)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss modules  (identical to CELL 4)
# ═══════════════════════════════════════════════════════════════════════════════

class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss   = torch.tensor(0.0)
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, h*w)
        return torch.bmm(f, f.transpose(1,2)) / (c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = GramMatrix()(target_feature).detach()
        self.loss   = torch.tensor(0.0)
    def forward(self, x):
        self.loss = F.mse_loss(GramMatrix()(x), self.target)
        return x


class TVLoss(nn.Module):
    def forward(self, x):
        return (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean() + \
               (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Build model  (identical to CELL 5)
# ═══════════════════════════════════════════════════════════════════════════════

CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS   = ["conv_1","conv_2","conv_3","conv_4","conv_5"]


def build_nst_model(content_img, style_img):
    vgg = models.vgg19(
        weights=models.VGG19_Weights.IMAGENET1K_V1
    ).features.to(device).eval()

    seq = nn.Sequential()
    content_losses, style_losses = [], []
    conv_idx = relu_idx = pool_idx = 0

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1;  name = f"conv_{conv_idx}"
        elif isinstance(layer, nn.ReLU):
            relu_idx += 1;  name = f"relu_{relu_idx}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            pool_idx += 1;  name = f"pool_{pool_idx}"
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            name = f"layer_{len(seq)}"

        seq.add_module(name, layer)

        if name in CONTENT_LAYERS:
            cl = ContentLoss(seq(content_img).detach())
            seq.add_module(f"closs_{conv_idx}", cl)
            content_losses.append(cl)

        if name in STYLE_LAYERS:
            sl = StyleLoss(seq(style_img).detach())
            seq.add_module(f"sloss_{conv_idx}", sl)
            style_losses.append(sl)

    last = max(i for i,(n,_) in enumerate(seq.named_children()) if "loss" in n)
    seq = nn.Sequential(*list(seq.children())[:last+1])
    return seq, content_losses, style_losses


# ═══════════════════════════════════════════════════════════════════════════════
# Optimisation loop  (identical to CELL 6, plus optional progress_cb)
# ═══════════════════════════════════════════════════════════════════════════════

def run_nst(content_img, style_img,
            num_steps=500, style_weight=1_000_000,
            content_weight=1, tv_weight=1,
            lr=0.02, save_every=100, output_dir=".",
            progress_cb=None):

    model, c_losses, s_losses = build_nst_model(content_img, style_img)
    tv_fn = TVLoss()

    canvas = content_img.clone().requires_grad_(True)
    opt    = optim.Adam([canvas], lr=lr)
    history = dict(style=[], content=[], tv=[], total=[])

    for step in range(1, num_steps + 1):
        with torch.no_grad():
            canvas.clamp_(-2.5, 2.5)

        opt.zero_grad()
        model(canvas)

        s_loss = sum(l.loss for l in s_losses)
        c_loss = sum(l.loss for l in c_losses)
        t_loss = tv_fn(canvas)
        loss   = style_weight*s_loss + content_weight*c_loss + tv_weight*t_loss
        loss.backward()
        opt.step()

        history["style"].append(s_loss.item())
        history["content"].append(c_loss.item())
        history["tv"].append(t_loss.item())
        history["total"].append(loss.item())

        if step % save_every == 0:
            ckpt = os.path.join(output_dir, f"nst_step_{step:04d}.jpg")
            save_tensor(canvas.detach(), ckpt)
            # ── Only new line: fire progress callback if provided ──
            if progress_cb:
                progress_cb(step, num_steps, {
                    "style":   s_loss.item(),
                    "content": c_loss.item(),
                    "tv":      t_loss.item(),
                })

    return canvas.detach(), history


# ═══════════════════════════════════════════════════════════════════════════════
# run_job — entry point called by main.py (no Colab-specific I/O)
# ═══════════════════════════════════════════════════════════════════════════════

def run_job(content_path, style_path, output_path,
            image_size=512, num_steps=300,
            style_weight=1_000_000, content_weight=1,
            tv_weight=1, lr=0.02, save_every=50,
            progress_cb=None):

    content_img = load_image(content_path, size=image_size)
    style_img   = load_image(style_path,   size=image_size)

    output_dir  = os.path.dirname(output_path) or "."

    output_tensor, _ = run_nst(
        content_img    = content_img,
        style_img      = style_img,
        num_steps      = num_steps,
        style_weight   = style_weight,
        content_weight = content_weight,
        tv_weight      = tv_weight,
        lr             = lr,
        save_every     = save_every,
        output_dir     = output_dir,
        progress_cb    = progress_cb,
    )

    save_tensor(output_tensor, output_path)
