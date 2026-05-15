"""
XY KSampler Plot — Custom ComfyUI node
Generates an XY grid of images by varying two parameters.
Designed for Cosmos Predict 2 / Anima (Qwen encoder).

Supported axes:
  Seed · LoRA · LoRA Weight · LoRA Enc Weight ·
  CFG  · Steps · Denoise · Sampler · Scheduler ·
  Prompt+ · Prompt-

Drop the whole folder into ComfyUI/custom_nodes/ and restart.
"""

import os
import traceback

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

AXIS_TYPES = [
    "Seed",
    "LoRA",
    "LoRA Weight",
    "LoRA Enc Weight",   # encoder (Qwen / text encoder) LoRA strength
    "CFG",
    "Steps",
    "Denoise",
    "Sampler",
    "Scheduler",
    "Prompt+",
    "Prompt-",
]
Y_AXIS_TYPES = ["None"] + AXIS_TYPES


# ──────────────────────────────────────────────
# Helpers: tensor normalization
# ──────────────────────────────────────────────

def normalize_to_3d(t: torch.Tensor) -> torch.Tensor:
    """
    Force any image tensor to exactly [H, W, C] (3D).
    VAE decode returns [B, H, W, C]; tiled-VAE may return higher.
    Peels leading dims one at a time, always taking index [0].
    """
    if t.ndim == 3:
        return t
    if t.ndim < 3:
        raise ValueError(
            f"[XYKSamplerPlot] normalize_to_3d: tensor has only {t.ndim} dims, "
            f"expected >= 3.  shape={tuple(t.shape)}"
        )
    while t.ndim > 3:
        if t.shape[0] != 1:
            print(
                f"[XYKSamplerPlot] WARNING: batch dim > 1 ({t.shape[0]}) "
                f"taking first item.  Full shape: {tuple(t.shape)}"
            )
        t = t[0]
    return t


# ──────────────────────────────────────────────
# Helpers: value parsing
# ──────────────────────────────────────────────

def parse_axis_values(raw: str, axis_type: str):
    """
    Parse axis values from user text.

    Prompt+/Prompt- use ||| as separator (prompts contain commas).
    Everything else: comma- or newline-separated, cast to the right type.
    """
    if axis_type in ("Prompt+", "Prompt-"):
        if "|||" in raw:
            return [t.strip() for t in raw.split("|||") if t.strip()]
        return [t.strip() for t in raw.split("\n") if t.strip()]

    raw = raw.replace("\n", ",")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return []
    try:
        if axis_type == "Seed":
            return [int(t) for t in tokens]
        elif axis_type in ("LoRA Weight", "LoRA Enc Weight", "CFG", "Denoise"):
            return [float(t) for t in tokens]
        elif axis_type == "Steps":
            return [int(t) for t in tokens]
        else:
            return tokens
    except ValueError as exc:
        print(f"[XYKSamplerPlot] Value parse error for axis '{axis_type}': {exc}")
        return tokens


# ──────────────────────────────────────────────
# Helpers: prompt encoding
# ──────────────────────────────────────────────

def encode_prompt(clip, text: str) -> list:
    """
    Encode a text string into a ComfyUI conditioning list.
    Mirrors exactly what ComfyUI's built-in CLIPTextEncode node does.
    Required formatting for SD3, Flux, and Cosmos (Qwen) encoders.
    """
    tokens = clip.tokenize(text)
    try:
        # Modern ComfyUI API (crucial for models with extra dict conditioning like Cosmos 2)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]
    except TypeError:
        # Older ComfyUI fallback
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]


# ──────────────────────────────────────────────
# Helpers: label rendering
# ──────────────────────────────────────────────

def _get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _shorten(text: str, max_chars: int = 28) -> str:
    _MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".bin", ".pth"}
    norm = text.replace("\\", "/")
    base = norm.split("/")[-1]
    ext  = os.path.splitext(base)[1].lower()
    display = os.path.splitext(base)[0] if ext in _MODEL_EXTS else base
    if len(display) > max_chars:
        display = "..." + display[-(max_chars - 1):]
    return display


def make_top_label(text: str, width: int, height: int, font_size: int) -> torch.Tensor:
    img = Image.new("RGB", (width, height), (28, 28, 36))
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    label = _shorten(text)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), label, fill=(210, 210, 230), font=font)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)


def make_side_label(text: str, height: int, width: int, font_size: int) -> torch.Tensor:
    img = Image.new("RGB", (width, height), (28, 28, 36))
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    label = _shorten(text, 60)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = width - tw - 8
    y = (height - th) // 2
    draw.text((x, y), label, fill=(210, 210, 230), font=font)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)


def make_gutter(dim: int, thickness: int, axis: str = "h") -> torch.Tensor:
    if axis == "h":
        return torch.full((thickness, dim, 3), 0.1)
    else:
        return torch.full((dim, thickness, 3), 0.1)


# ──────────────────────────────────────────────
# Helpers: LoRA loading
# ──────────────────────────────────────────────

def apply_lora(model, clip, lora_name: str, model_strength: float, clip_strength: float):
    if not lora_name or lora_name == "None":
        return model, clip

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        print(f"[XYKSamplerPlot] LoRA not found on disk: {lora_name!r}  — skipping.")
        return model, clip

    try:
        lora_weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
        patched_model, patched_clip = comfy.sd.load_lora_for_models(
            model, clip, lora_weights, model_strength, clip_strength
        )
        return patched_model, patched_clip
    except Exception as exc:
        print(f"[XYKSamplerPlot] Failed to load LoRA {lora_name!r}: {exc}")
        return model, clip


# ──────────────────────────────────────────────
# Helpers: sampling
# ──────────────────────────────────────────────

def run_sampler(model, positive, negative, latent: dict,
                seed: int, steps: int, cfg: float,
                sampler_name: str, scheduler: str, denoise: float) -> dict:
    
    latent_samples = latent["samples"].clone() 

    try:
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
    except AttributeError:
        pass

    batch_inds = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
    noise_mask = latent.get("noise_mask")

    samples_out = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_samples,
        denoise=denoise,
        disable_noise=False,
        force_full_denoise=False,  # Fixed: Match standard ComfyUI KSampler instead of forcing to True
        noise_mask=noise_mask,
        seed=seed,
    )

    out = latent.copy()
    out["samples"] = samples_out
    return out


# ──────────────────────────────────────────────
# Grid assembly
# ──────────────────────────────────────────────

def assemble_grid(cells, x_labels, y_labels, add_labels,
                  cell_padding=4, label_font_size=0) -> torch.Tensor:
    n_rows = len(cells)
    n_cols = len(cells[0])

    cells = [[normalize_to_3d(img) for img in row] for row in cells]

    img_h = cells[0][0].shape[0]
    img_w = cells[0][0].shape[1]

    font_sz   = label_font_size if label_font_size > 0 else max(14, min(48, min(img_h, img_w) // 18))
    top_strip = max(font_sz + 16, 50)

    font_probe  = _get_font(font_sz)
    _probe_img  = Image.new("RGB", (1, 1))
    _probe_draw = ImageDraw.Draw(_probe_img)
    side_strip  = max(
        (_probe_draw.textbbox((0, 0), _shorten(yl, 60), font=font_probe)[2] + 24
         for yl in y_labels if yl),
        default=top_strip,
    )

    gutter     = max(0, cell_padding)
    has_side   = add_labels and any(y_labels)
    has_header = add_labels

    row_tensors = []

    if has_header:
        header_cells = []
        if has_side:
            header_cells.append(torch.zeros(top_strip, side_strip, 3))
            if gutter:
                header_cells.append(make_gutter(top_strip, gutter, axis="w"))
        for ci, xl in enumerate(x_labels):
            header_cells.append(make_top_label(xl, img_w, top_strip, font_sz))
            if ci < n_cols - 1 and gutter:
                header_cells.append(make_gutter(top_strip, gutter, axis="w"))
        row_tensors.append(torch.cat(header_cells, dim=1))
        if gutter:
            row_tensors.append(make_gutter(row_tensors[-1].shape[1], gutter))

    for ri, row in enumerate(cells):
        row_cells = []
        if has_side:
            yl = y_labels[ri] if ri < len(y_labels) else ""
            row_cells.append(make_side_label(yl, img_h, side_strip, font_sz))
            if gutter:
                row_cells.append(make_gutter(img_h, gutter, axis="w"))
        for ci, img in enumerate(row):
            row_cells.append(img)
            if ci < n_cols - 1 and gutter:
                row_cells.append(make_gutter(img_h, gutter, axis="w"))
        row_tensors.append(torch.cat(row_cells, dim=1))
        if ri < n_rows - 1 and gutter:
            row_tensors.append(make_gutter(row_tensors[-1].shape[1], gutter))

    return torch.cat(row_tensors, dim=0)


def error_cell(h: int, w: int, message: str = "ERROR") -> torch.Tensor:
    img = Image.new("RGB", (w, h), (60, 20, 20))
    draw = ImageDraw.Draw(img)
    font = _get_font(16)
    bbox = draw.textbbox((0, 0), message, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), message, fill=(220, 80, 80), font=font)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)


# ──────────────────────────────────────────────
# The Node
# ──────────────────────────────────────────────

class XYKSamplerPlot:

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("grid_image", "all_images")
    FUNCTION      = "generate"
    CATEGORY      = "sampling/xy_plot"
    OUTPUT_NODE   = False

    @classmethod
    def INPUT_TYPES(cls):
        loras      = ["None"] + folder_paths.get_filename_list("loras")
        samplers   = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS

        return {
            "required": {
                "model":        ("MODEL",),
                "clip":         ("CLIP",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "vae":          ("VAE",),
                
                "seed":         ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":        ("INT",   {"default": 20,  "min": 1,   "max": 200}),
                "cfg":          ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0,  "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler":    (schedulers,),
                "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,   "step": 0.01}),
                
                "base_lora":           (loras,),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength":  ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Encoder (Qwen / text encoder) LoRA strength.",
                }),
                
                "prompt_prefix": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prepended to every Prompt+ axis value.",
                }),
                "negative_prefix": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prepended to every Prompt- axis value.",
                }),
                
                "x_axis":  (AXIS_TYPES,),
                "x_values": ("STRING", {
                    "multiline": True,
                    "default": "0, 1, 2, 3",
                }),
                
                "y_axis":  (Y_AXIS_TYPES,),
                "y_values": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Same format as x_values. Leave empty when y_axis = None.",
                }),
                
                "add_labels":      ("BOOLEAN", {"default": True}),
                "cell_padding":    ("INT",     {"default": 4,  "min": 0, "max": 64, "step": 1}),
                "label_font_size": ("INT",     {"default": 0,  "min": 0, "max": 72, "step": 1,
                                               "tooltip": "0 = auto-scale with image size."}),
                "no_lora_baseline": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When a LoRA axis is active, prepend a baseline cell "
                        "with no LoRA so you can compare against the base model."
                    ),
                }),
            }
        }

    def generate(
        self,
        model, clip, positive, negative, latent_image, vae,
        seed, steps, cfg, sampler_name, scheduler, denoise,
        base_lora, lora_model_strength, lora_clip_strength,
        prompt_prefix, negative_prefix,
        x_axis, x_values,
        y_axis, y_values,
        add_labels, cell_padding, label_font_size,
        no_lora_baseline,
    ):
        cell_padding    = int(cell_padding)    if cell_padding    is not None else 4
        label_font_size = int(label_font_size) if label_font_size is not None else 0

        x_vals = parse_axis_values(x_values, x_axis)
        if not x_vals:
            raise ValueError(f"[XYKSamplerPlot] x_values is empty or unparseable: {x_values!r}")

        if y_axis == "None":
            y_vals   = [None]
            y_labels = [""]
        else:
            y_vals = parse_axis_values(y_values, y_axis)
            if not y_vals:
                raise ValueError(f"[XYKSamplerPlot] y_values is empty for axis '{y_axis}': {y_values!r}")
            y_labels = [str(v) for v in y_vals]

        x_labels = [str(v) for v in x_vals]

        if no_lora_baseline:
            if x_axis == "LoRA":
                x_vals   = ["None"] + list(x_vals)
                x_labels = ["NO LORA"] + x_labels
            if y_axis == "LoRA":
                y_vals   = ["None"] + list(y_vals)
                y_labels = ["NO LORA"] + y_labels

        latent_h = latent_image["samples"].shape[2] * 8
        latent_w = latent_image["samples"].shape[3] * 8

        grid_cells = []
        all_imgs   = []

        total = len(y_vals) * len(x_vals)
        done  = 0

        for y_val in y_vals:
            row_cells = []

            for x_val in x_vals:
                done += 1
                print(f"[XYKSamplerPlot] Cell {done}/{total}  |  x={x_val}  y={y_val}")

                cur_seed   = seed
                cur_steps  = steps
                cur_cfg    = cfg
                cur_lora   = base_lora
                cur_lora_m = lora_model_strength
                cur_lora_c = lora_clip_strength

                cell_sampler   = sampler_name
                cell_scheduler = scheduler
                cell_denoise   = denoise

                # Resolve LoRA axes first!
                def _apply_lora(axis_type, value):
                    nonlocal cur_lora, cur_lora_m, cur_lora_c
                    if axis_type == "LoRA":
                        cur_lora = str(value)
                    elif axis_type == "LoRA Weight":
                        cur_lora_m = float(value)
                    elif axis_type == "LoRA Enc Weight":
                        cur_lora_c = float(value)

                _apply_lora(x_axis, x_val)
                if y_axis != "None" and y_val is not None:
                    _apply_lora(y_axis, y_val)

                # ── Load LoRA (fresh from original weights each cell) ────────
                cell_model, cell_clip = apply_lora(
                    model, clip, cur_lora, cur_lora_m, cur_lora_c
                )

                # Resolve standard and Prompt axes using the patched `cell_clip`
                cell_positive  = positive
                cell_negative  = negative

                def _apply_other(axis_type, value):
                    nonlocal cur_seed, cur_steps, cur_cfg
                    nonlocal cell_positive, cell_negative
                    nonlocal cell_sampler, cell_scheduler, cell_denoise

                    if axis_type == "Seed":
                        cur_seed = int(value)
                    elif axis_type == "Steps":
                        cur_steps = int(value)
                    elif axis_type == "CFG":
                        cur_cfg = float(value)
                    elif axis_type == "Denoise":
                        cell_denoise = float(value)
                    elif axis_type == "Sampler":
                        cell_sampler = str(value)
                    elif axis_type == "Scheduler":
                        cell_scheduler = str(value)
                    elif axis_type == "Prompt+":
                        prefix = (prompt_prefix or "").strip()
                        text   = f"{prefix}, {str(value).strip()}" if prefix else str(value).strip()
                        cell_positive = encode_prompt(cell_clip, text)
                    elif axis_type == "Prompt-":
                        prefix = (negative_prefix or "").strip()
                        text   = f"{prefix}, {str(value).strip()}" if prefix else str(value).strip()
                        cell_negative = encode_prompt(cell_clip, text)

                _apply_other(x_axis, x_val)
                if y_axis != "None" and y_val is not None:
                    _apply_other(y_axis, y_val)

                # ── Sample & decode ──────────────────────────────────────────
                try:
                    result_latent = run_sampler(
                        cell_model, cell_positive, cell_negative, latent_image,
                        cur_seed, cur_steps, cur_cfg,
                        cell_sampler, cell_scheduler, cell_denoise,
                    )
                    decoded = vae.decode(result_latent["samples"])
                    img = normalize_to_3d(decoded).clamp(0.0, 1.0)
                except Exception:
                    traceback.print_exc()
                    img = error_cell(latent_h, latent_w, f"ERR\nx={x_val}\ny={y_val}")

                row_cells.append(img)
                all_imgs.append(img.unsqueeze(0))

                comfy.model_management.soft_empty_cache()

            grid_cells.append(row_cells)

        grid_hw  = assemble_grid(
            grid_cells,
            x_labels=x_labels,
            y_labels=y_labels,
            add_labels=add_labels,
            cell_padding=cell_padding,
            label_font_size=label_font_size,
        )
        grid_out     = grid_hw.unsqueeze(0)            # [1, H, W, 3]
        images_batch = torch.cat(all_imgs, dim=0)      # [N, H, W, 3]

        return (grid_out, images_batch)


# ──────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "XYKSamplerPlot": XYKSamplerPlot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XYKSamplerPlot": "XY KSampler Plot",
}