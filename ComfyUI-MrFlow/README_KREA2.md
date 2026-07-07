# MrFlow Krea2 ComfyUI Nodes

This folder includes an experimental Krea2-oriented MrFlow ComfyUI node set.

## What the Krea2 nodes add

The Krea2 nodes mirror the repository's Qwen-oriented MrFlow flow, but keep separate node IDs and defaults:

- `MrFlow Krea2 Preset`
- `MrFlow Krea2 Upscale + Encode`
- `MrFlow Krea2 Refine`
- `MrFlow Krea2 Save Image`

The implementation is model-loader agnostic. It expects the normal ComfyUI sockets:

- `MODEL`
- `VAE`
- `CONDITIONING`
- `LATENT`
- `IMAGE`
- `UPSCALE_MODEL`

Use whatever Krea2 loader is available in your ComfyUI environment, then wire the low-resolution Krea2 generation output into the MrFlow Krea2 upscale/encode/refine stage.

## Suggested starting workflow

1. Generate the first pass at low resolution.
   - Use `MrFlow Krea2 Preset` to calculate `low_width`, `low_height`, and `stage1_steps`.
   - For Krea2 Turbo, start with `turbo_8plus1` and CFG `0.0`.
   - For Krea2 Raw/Base, start with `raw_12plus1` and CFG around `3.5`.

2. Decode the low-resolution latent to image.

3. Run `MrFlow Krea2 Upscale + Encode`.
   - Connect the decoded low-resolution image.
   - Connect your VAE.
   - Connect a 2x upscale model, such as the Real-ESRGAN x2 model used by this repository.
   - Connect the preset node's `target_width` and `target_height`.

4. Run `MrFlow Krea2 Refine`.
   - Connect the same Krea2 `MODEL`, `VAE`, positive/negative conditioning, and the `prepared_latent`.
   - Connect `refine_steps`, `refine_denoise`, and `suggested_cfg` from the preset node.
   - Start with an Euler-family sampler if your ComfyUI build exposes multiple sampler choices.

5. Save the refined image with `MrFlow Krea2 Save Image`.

## Presets

| Preset | Stage-1 steps | Refine steps | Direct sigma | Suggested CFG |
| --- | ---: | ---: | ---: | ---: |
| `turbo_8plus1` | 8 | 1 | 0.16 | 0.0 |
| `turbo_12plus1` | 12 | 1 | 0.16 | 0.0 |
| `raw_12plus1` | 12 | 1 | 0.12 | 3.5 |
| `raw_20plus1` | 20 | 1 | 0.15 | 3.5 |

## Notes

- These nodes do not include a Krea2 model loader. They are the MrFlow staging/refinement pieces.
- The refine node uses explicit direct-sigma nodes, matching the MrFlow design used by the Qwen plugin.
- Enable `print_schedule` on `MrFlow Krea2 Refine` to print the exact refinement sigma path in the ComfyUI console.
- Krea2 Turbo is distilled, so CFG `0.0` is the recommended starting point.
- Raw/Base variants generally need normal CFG values.
