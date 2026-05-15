# XY KSampler Plot

A custom ComfyUI node that generates a **full XY grid** of images by independently varying two parameters — LoRA selection, seed, LoRA weight, CFG, or steps.
---

## Installation

```
ComfyUI/
└── custom_nodes/
    └── xy_ksampler_plot/   ← drop this whole folder here
        ├── __init__.py
        ├── nodes.py
        └── web/
            └── xy_ksampler.js
```

Restart ComfyUI. The node appears under **sampling → xy_plot** as **"XY KSampler Plot 🎛️"**.

---

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| model / clip / vae | standard | from your checkpoint loader |
| positive / negative | CONDITIONING | from your text encoder |
| latent_image | LATENT | from Empty Latent Image |
| seed | INT | base seed (overridden when axis = Seed) |
| steps | INT | base step count |
| cfg | FLOAT | base CFG |
| sampler_name | COMBO | e.g. euler, dpmpp_2m |
| scheduler | COMBO | e.g. karras, simple |
| denoise | FLOAT | 1.0 = full denoise |
| base_lora | COMBO | LoRA applied when axis ≠ LoRA |
| lora_model_strength | FLOAT | model weight for base/axis LoRA |
| lora_clip_strength | FLOAT | clip weight for base/axis LoRA |
| x_axis | COMBO | Seed · LoRA · LoRA Weight · CFG · Steps |
| x_values | STRING | comma-separated values for X axis |
| y_axis | COMBO | None · Seed · LoRA · LoRA Weight · CFG · Steps |
| y_values | STRING | comma-separated values for Y axis |
| add_labels | BOOL | draw axis labels on the grid |

---

## Value format

| Axis type | Format | Example |
|-----------|--------|---------|
| Seed | integers | `0, 42, 1337` |
| Steps | integers | `10, 20, 30` |
| CFG | floats | `4.0, 7.0, 9.0` |
| LoRA Weight | floats | `0.5, 0.75, 1.0` |
| LoRA | filenames | `anima_v1.safetensors, anima_v2.safetensors` |

For LoRA axes, use the **📂 Pick X/Y LoRAs** buttons that appear on the node — they open a searchable checkbox picker that fills the field automatically.

---

## Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| grid_image | [1, H, W, 3] | assembled grid with optional labels |
| all_images | [N, H, W, 3] | every cell as a batch |

---

## Typical use: LoRA comparison × seed

```
x_axis  = LoRA
x_values = anima_char_eliza_v2_s10.safetensors, anima_char_eliza_v2_s40.safetensors, anima_char_eliza_v2_s70.safetensors

y_axis  = Seed
y_values = 0, 42, 1337
```

This produces a 3×3 grid: 3 LoRA checkpoints across columns, 3 seeds down rows.
