# MrFlow Tiled Upscale / Refine Nodes

This is an experimental memory-smart tiled add-on for very large MrFlow/Krea2-style upscaling.

## Why this exists

A normal full-frame high-resolution refine can run out of VRAM at 2048, 3072, 4096, or larger. These nodes split the work into overlapping tiles, process one tile at a time, feather the overlaps, and clear cache between tiles.

## Training-free policy

The core path is training-free and does not require a learned latent upscaler.

The intended workflow is:

```text
pixel-space upscale
→ VAE re-encode
→ controlled latent noise
→ real model refine pass
```

A learned latent upscaler such as NNLatentUpscale can still be used by someone who wants it, but it is not part of the recommended MrFlow path because it uses separate trained weights.

## Added nodes

- `MrFlow Tiled Plan`
  - Calculates target size, source tile size, source overlap, latent tile size, and latent overlap.
- `MrFlow Tiled Pixel Upscale`
  - Splits an `IMAGE` into source tiles.
  - Runs the ComfyUI `UPSCALE_MODEL` on each tile.
  - Stitches the tiles in pixel space with overlap feathering.
- `MrFlow Tiled VAE Encode`
  - Uses `vae.encode_tiled` when available, with fallback to normal encode.
- `MrFlow Tiled VAE Decode`
  - Uses `vae.decode_tiled` when available, with fallback to normal decode.
- `MrFlow Latent Upscale + Noise`
  - Lightweight interpolation-based latent resize fallback plus optional Gaussian noise.
  - This is not a learned latent super-resolution model.
- `MrFlow Latent Noise Inject`
  - Adds controlled latent noise before a refine pass.
- `MrFlow Tiled Latent Refine`
  - Runs the MrFlow direct-sigma refine on overlapping latent tiles.
  - Stitches the refined latent tiles back together.
- `MrFlow Tiled Save Image`
  - Save helper for tiled outputs.

## Recommended training-free high-quality chain

```text
Krea2 low-res generation
→ VAE Decode
→ MrFlow Tiled Pixel Upscale
→ MrFlow Tiled VAE Encode
→ MrFlow Latent Noise Inject
→ MrFlow Tiled Latent Refine
→ MrFlow Tiled Save Image
```

## Suggested starting values

For 1024 → 2048:

```text
source_tile_size: 512
source_overlap: 64
latent_tile_size: 128
latent_overlap: 16
denoise: 0.08 - 0.16
steps: 1 - 2
```

For 2048 → 4096:

```text
source_tile_size: 512 or 768
source_overlap: 96 - 128
latent_tile_size: 128 or 160
latent_overlap: 24 - 32
denoise: 0.06 - 0.12
steps: 1 - 2
```

For 4096+:

```text
source_tile_size: 512
source_overlap: 128
latent_tile_size: 96 - 128
latent_overlap: 32
denoise: 0.04 - 0.10
steps: 1
```

## Quality notes

- Pixel tiled upscale is safest for huge resolution growth.
- Latent tiled refine adds real model detail, but too much denoise can change tile content and create seams.
- More overlap improves seams but costs more time.
- For tile refine, keep denoise lower than normal full-frame refine.
- 2x per pass is usually safer than one giant 4x or 8x jump.

## Limitations

- The tiled latent refine is experimental.
- It does not yet add model-specific positional conditioning per tile.
- Very strong tile denoise may create inconsistent objects across tiles.
- Best use is final detail/refinement, not full semantic regeneration per tile.
