from __future__ import annotations

import math

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import torch
from comfy import model_management
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_api.latest import ComfyExtension, io, ui
from nodes import MAX_RESOLUTION
from typing_extensions import override


def _resize_image(image: torch.Tensor, width: int, height: int, method: str = "bicubic") -> torch.Tensor:
    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return resized.movedim(1, -1)


def _resize_latent_samples(samples: torch.Tensor, width: int, height: int, method: str = "bicubic") -> torch.Tensor:
    return comfy.utils.common_upscale(samples, width, height, method, "disabled")


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if tile >= length:
        return [0]

    stride = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_coords(width: int, height: int, tile_w: int, tile_h: int, overlap: int) -> list[tuple[int, int, int, int]]:
    xs = _tile_starts(width, min(tile_w, width), overlap)
    ys = _tile_starts(height, min(tile_h, height), overlap)
    coords: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(width, x0 + tile_w)
            y1 = min(height, y0 + tile_h)
            coords.append((x0, y0, x1, y1))
    return coords


def _feather_1d(length: int, left: bool, right: bool, overlap: int, device, dtype) -> torch.Tensor:
    weights = torch.ones((length,), device=device, dtype=dtype)
    if overlap <= 0 or length <= 1:
        return weights

    ramp = min(overlap, max(1, length // 2))
    if left and ramp > 1:
        weights[:ramp] *= torch.linspace(0.0, 1.0, ramp, device=device, dtype=dtype).clamp_min(1.0e-4)
    if right and ramp > 1:
        weights[-ramp:] *= torch.linspace(1.0, 0.0, ramp, device=device, dtype=dtype).clamp_min(1.0e-4)
    return weights


def _feather_mask(
    height: int,
    width: int,
    blend_top: bool,
    blend_bottom: bool,
    blend_left: bool,
    blend_right: bool,
    overlap_y: int,
    overlap_x: int,
    device,
    dtype,
) -> torch.Tensor:
    wy = _feather_1d(height, blend_top, blend_bottom, overlap_y, device, dtype).view(1, height, 1, 1)
    wx = _feather_1d(width, blend_left, blend_right, overlap_x, device, dtype).view(1, 1, width, 1)
    return wy * wx


def _latent_feather_mask(
    height: int,
    width: int,
    blend_top: bool,
    blend_bottom: bool,
    blend_left: bool,
    blend_right: bool,
    overlap_y: int,
    overlap_x: int,
    device,
    dtype,
) -> torch.Tensor:
    wy = _feather_1d(height, blend_top, blend_bottom, overlap_y, device, dtype).view(1, 1, height, 1)
    wx = _feather_1d(width, blend_left, blend_right, overlap_x, device, dtype).view(1, 1, 1, width)
    return wy * wx


def _flowmatch_shift(t: torch.Tensor, mu: float, sigma: float = 1.0) -> torch.Tensor:
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0) ** sigma)


def _direct_sigma_shift_mu(steps: int) -> float:
    if steps <= 1:
        return 0.0
    return 0.25 * float(steps - 1)


def _direct_sigma_nodes(first_sigma: float, steps: int, device: torch.device) -> torch.Tensor:
    if not 0.0 < first_sigma < 1.0:
        raise ValueError(f"first_sigma must be in (0, 1), got {first_sigma}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps == 1:
        return torch.tensor([float(first_sigma), 0.0], dtype=torch.float32, device=device)

    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device=device)
    mu = _direct_sigma_shift_mu(steps)
    shifted = _flowmatch_shift(base.clamp(1.0e-6, 1.0 - 1.0e-6), mu=mu)
    shifted = shifted - shifted[-1]
    shifted = shifted / shifted[0]
    shifted = shifted * float(first_sigma)
    shifted[0] = float(first_sigma)
    shifted[-1] = 0.0
    return shifted


def _offload_and_empty_cache() -> None:
    try:
        model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _copy_latent_metadata(src: dict, samples: torch.Tensor) -> dict:
    out = src.copy()
    out["samples"] = samples
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    return out


class MrFlowTiledPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "source_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "source_tile_size": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "source_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("target_width", "target_height", "source_tile_size", "source_overlap", "latent_tile_size", "latent_overlap")
    FUNCTION = "build"
    CATEGORY = "MrFlow/Tiled"

    def build(self, source_width: int, source_height: int, upscale_factor: float, source_tile_size: int, source_overlap: int):
        target_width = max(16, int(round(source_width * upscale_factor / 16.0)) * 16)
        target_height = max(16, int(round(source_height * upscale_factor / 16.0)) * 16)
        latent_tile_size = max(8, int(round(source_tile_size / 8.0)))
        latent_overlap = max(0, int(round(source_overlap / 8.0)))
        return (target_width, target_height, source_tile_size, source_overlap, latent_tile_size, latent_overlap)


class MrFlowTiledPixelUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "target_width": ("INT", {"default": 2048, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "target_height": ("INT", {"default": 2048, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "source_tile_size": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "source_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "resize_method": (["bicubic", "bilinear", "area", "nearest-exact"], {"default": "bicubic"}),
                "clear_cache_per_tile": ("BOOLEAN", {"default": True}),
                "print_tiles": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("upscaled_image", "tile_count")
    FUNCTION = "upscale"
    CATEGORY = "MrFlow/Tiled"

    def upscale(
        self,
        image,
        upscale_model,
        target_width: int,
        target_height: int,
        source_tile_size: int,
        source_overlap: int,
        resize_method: str,
        clear_cache_per_tile: bool,
        print_tiles: bool,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(image.shape)}")

        batch, source_height, source_width, channels = image.shape
        scale_x = float(target_width) / float(source_width)
        scale_y = float(target_height) / float(source_height)
        coords = _tile_coords(source_width, source_height, source_tile_size, source_tile_size, source_overlap)

        canvas = torch.zeros((batch, target_height, target_width, channels), dtype=torch.float32, device="cpu")
        weight_sum = torch.zeros((1, target_height, target_width, 1), dtype=torch.float32, device="cpu")

        if print_tiles:
            print(f"[MrFlow Tiled Pixel Upscale] {len(coords)} tiles, source={source_width}x{source_height}, target={target_width}x{target_height}")

        for index, (x0, y0, x1, y1) in enumerate(coords):
            tile = image[:, y0:y1, x0:x1, :]
            up_tile = ImageUpscaleWithModel.execute(upscale_model, tile)[0]
            tx0 = int(round(x0 * scale_x))
            ty0 = int(round(y0 * scale_y))
            tx1 = target_width if x1 == source_width else int(round(x1 * scale_x))
            ty1 = target_height if y1 == source_height else int(round(y1 * scale_y))
            tw = max(1, tx1 - tx0)
            th = max(1, ty1 - ty0)

            if up_tile.shape[2] != tw or up_tile.shape[1] != th:
                up_tile = _resize_image(up_tile, tw, th, method=resize_method)

            overlap_x = int(round(source_overlap * scale_x))
            overlap_y = int(round(source_overlap * scale_y))
            mask = _feather_mask(
                th,
                tw,
                blend_top=y0 > 0,
                blend_bottom=y1 < source_height,
                blend_left=x0 > 0,
                blend_right=x1 < source_width,
                overlap_y=overlap_y,
                overlap_x=overlap_x,
                device=up_tile.device,
                dtype=up_tile.dtype,
            )

            canvas[:, ty0:ty1, tx0:tx1, :] += (up_tile * mask).detach().cpu().float()
            weight_sum[:, ty0:ty1, tx0:tx1, :] += mask.detach().cpu().float()

            if print_tiles:
                print(f"[MrFlow Tiled Pixel Upscale] tile {index + 1}/{len(coords)} src=({x0},{y0})-({x1},{y1}) dst=({tx0},{ty0})-({tx1},{ty1})")

            del tile, up_tile, mask
            if clear_cache_per_tile:
                _offload_and_empty_cache()

        canvas = canvas / weight_sum.clamp_min(1.0e-6)
        return (canvas.clamp(0.0, 1.0), len(coords))


class MrFlowTiledVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 16}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "MrFlow/Tiled"

    def encode(self, image, vae, tile_size: int, overlap: int):
        if hasattr(vae, "encode_tiled"):
            samples = vae.encode_tiled(image, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
        else:
            print("[MrFlow Tiled VAE Encode] vae.encode_tiled was not found; falling back to normal vae.encode.")
            samples = vae.encode(image)
        return ({"samples": samples},)


class MrFlowTiledVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 16}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "MrFlow/Tiled"

    def decode(self, latent, vae, tile_size: int, overlap: int):
        samples = latent["samples"]
        if hasattr(vae, "decode_tiled"):
            image = vae.decode_tiled(samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
        else:
            print("[MrFlow Tiled VAE Decode] vae.decode_tiled was not found; falling back to normal vae.decode.")
            image = vae.decode(samples)
        if len(image.shape) == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        return (image,)


class MrFlowLatentUpscaleAndNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "area"], {"default": "bicubic"}),
                "noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "seed": ("INT", {"default": 2026, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = "MrFlow/Tiled"

    def upscale(self, latent, upscale_factor: float, upscale_method: str, noise_strength: float, seed: int):
        samples = latent["samples"]
        target_h = max(1, int(round(samples.shape[-2] * upscale_factor)))
        target_w = max(1, int(round(samples.shape[-1] * upscale_factor)))
        upscaled = _resize_latent_samples(samples, target_w, target_h, method=upscale_method)

        if noise_strength > 0.0:
            generator = torch.Generator(device=upscaled.device).manual_seed(seed)
            noise = torch.randn(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
            upscaled = upscaled + noise * float(noise_strength)

        return (_copy_latent_metadata(latent, upscaled),)


class MrFlowLatentNoiseInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 2026, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "noise_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mode": (["add", "lerp"], {"default": "add"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "inject"
    CATEGORY = "MrFlow/Tiled"

    def inject(self, latent, seed: int, noise_strength: float, mode: str):
        samples = latent["samples"]
        generator = torch.Generator(device=samples.device).manual_seed(seed)
        noise = torch.randn(samples.shape, generator=generator, device=samples.device, dtype=samples.dtype)

        if mode == "lerp":
            mixed = samples * (1.0 - float(noise_strength)) + noise * float(noise_strength)
        else:
            mixed = samples + noise * float(noise_strength)

        return (_copy_latent_metadata(latent, mixed),)


class MrFlowTiledLatentRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 2026, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "denoise": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "latent_tile_size": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 8}),
                "latent_overlap": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 4}),
                "decode_tiled": ("BOOLEAN", {"default": True}),
                "vae_tile_size": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 16}),
                "vae_overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8}),
                "clear_cache_per_tile": ("BOOLEAN", {"default": True}),
                "print_tiles": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT")
    RETURN_NAMES = ("refined_latent", "refined_image", "tile_count")
    FUNCTION = "refine"
    CATEGORY = "MrFlow/Tiled"

    def refine(
        self,
        model,
        vae,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        denoise: float,
        latent_tile_size: int,
        latent_overlap: int,
        decode_tiled: bool,
        vae_tile_size: int,
        vae_overlap: int,
        clear_cache_per_tile: bool,
        print_tiles: bool,
    ):
        latent = latent_image.copy()
        samples = latent["samples"]
        samples = comfy.sample.fix_empty_latent_channels(
            model,
            samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )

        batch, channels, latent_h, latent_w = samples.shape
        coords = _tile_coords(latent_w, latent_h, latent_tile_size, latent_tile_size, latent_overlap)
        refined = torch.zeros_like(samples, device="cpu", dtype=torch.float32)
        weight_sum = torch.zeros((1, 1, latent_h, latent_w), device="cpu", dtype=torch.float32)

        sigmas = _direct_sigma_nodes(denoise, steps, device=model.load_device)
        sampler = comfy.samplers.sampler_object(sampler_name)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        if print_tiles:
            print(f"[MrFlow Tiled Latent Refine] {len(coords)} latent tiles, latent={latent_w}x{latent_h}, sigmas={[round(float(x), 6) for x in sigmas.detach().cpu()]}")

        for index, (x0, y0, x1, y1) in enumerate(coords):
            tile_samples = samples[:, :, y0:y1, x0:x1]
            tile_latent = latent.copy()
            tile_latent["samples"] = tile_samples

            batch_inds = tile_latent.get("batch_index", None)
            tile_seed = int(seed) + index
            noise = comfy.sample.prepare_noise(tile_samples, tile_seed, batch_inds)
            noise_mask = tile_latent.get("noise_mask", None)
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)

            refined_tile = comfy.sample.sample_custom(
                model,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                tile_samples,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=tile_seed,
            )

            th = y1 - y0
            tw = x1 - x0
            mask = _latent_feather_mask(
                th,
                tw,
                blend_top=y0 > 0,
                blend_bottom=y1 < latent_h,
                blend_left=x0 > 0,
                blend_right=x1 < latent_w,
                overlap_y=latent_overlap,
                overlap_x=latent_overlap,
                device=refined_tile.device,
                dtype=refined_tile.dtype,
            )

            refined[:, :, y0:y1, x0:x1] += (refined_tile * mask).detach().cpu().float()
            weight_sum[:, :, y0:y1, x0:x1] += mask.detach().cpu().float()

            if print_tiles:
                print(f"[MrFlow Tiled Latent Refine] tile {index + 1}/{len(coords)} latent=({x0},{y0})-({x1},{y1}) seed={tile_seed}")

            del tile_samples, refined_tile, mask
            if clear_cache_per_tile:
                _offload_and_empty_cache()

        refined = refined / weight_sum.clamp_min(1.0e-6)
        refined_latent = _copy_latent_metadata(latent, refined.to(samples.device, dtype=samples.dtype))

        if decode_tiled and hasattr(vae, "decode_tiled"):
            refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=vae_tile_size, tile_y=vae_tile_size, overlap=vae_overlap)
        else:
            refined_image = vae.decode(refined_latent["samples"])
        if len(refined_image.shape) == 5:
            refined_image = refined_image.reshape(-1, refined_image.shape[-3], refined_image.shape[-2], refined_image.shape[-1])

        return (refined_latent, refined_image, len(coords))


class MrFlowTiledSaveImage:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "MrFlow/Tiled/output"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "MrFlow/Tiled"

    def save_images(self, images, filename_prefix="MrFlow/Tiled/output", prompt=None, extra_pnginfo=None):
        saved = ui.ImageSaveHelper.get_save_images_ui(
            images,
            filename_prefix=filename_prefix,
            cls=None,
            compress_level=self.compress_level,
        )
        return {"ui": saved.as_dict(), "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "MrFlowTiledPlan": MrFlowTiledPlan,
    "MrFlowTiledPixelUpscale": MrFlowTiledPixelUpscale,
    "MrFlowTiledVAEEncode": MrFlowTiledVAEEncode,
    "MrFlowTiledVAEDecode": MrFlowTiledVAEDecode,
    "MrFlowLatentUpscaleAndNoise": MrFlowLatentUpscaleAndNoise,
    "MrFlowLatentNoiseInject": MrFlowLatentNoiseInject,
    "MrFlowTiledLatentRefine": MrFlowTiledLatentRefine,
    "MrFlowTiledSaveImage": MrFlowTiledSaveImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MrFlowTiledPlan": "MrFlow Tiled Plan",
    "MrFlowTiledPixelUpscale": "MrFlow Tiled Pixel Upscale",
    "MrFlowTiledVAEEncode": "MrFlow Tiled VAE Encode",
    "MrFlowTiledVAEDecode": "MrFlow Tiled VAE Decode",
    "MrFlowLatentUpscaleAndNoise": "MrFlow Latent Upscale + Noise",
    "MrFlowLatentNoiseInject": "MrFlow Latent Noise Inject",
    "MrFlowTiledLatentRefine": "MrFlow Tiled Latent Refine",
    "MrFlowTiledSaveImage": "MrFlow Tiled Save Image",
}


class MrFlowTiledPlanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowTiledPlan",
            display_name="MrFlow Tiled Plan",
            category="MrFlow/Tiled",
            inputs=[
                io.Int.Input("source_width", default=1024, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("source_height", default=1024, min=64, max=MAX_RESOLUTION, step=16),
                io.Float.Input("upscale_factor", default=2.0, min=1.0, max=16.0, step=0.05),
                io.Int.Input("source_tile_size", default=512, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("source_overlap", default=64, min=0, max=1024, step=8),
            ],
            outputs=[
                io.Int.Output(display_name="target_width"),
                io.Int.Output(display_name="target_height"),
                io.Int.Output(display_name="source_tile_size"),
                io.Int.Output(display_name="source_overlap"),
                io.Int.Output(display_name="latent_tile_size"),
                io.Int.Output(display_name="latent_overlap"),
            ],
        )

    @classmethod
    def execute(cls, source_width, source_height, upscale_factor, source_tile_size, source_overlap):
        return io.NodeOutput(*MrFlowTiledPlan().build(source_width, source_height, upscale_factor, source_tile_size, source_overlap))


class MrFlowTiledPixelUpscaleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowTiledPixelUpscale",
            display_name="MrFlow Tiled Pixel Upscale",
            category="MrFlow/Tiled",
            inputs=[
                io.Image.Input("image"),
                io.UpscaleModel.Input("upscale_model"),
                io.Int.Input("target_width", default=2048, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("target_height", default=2048, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("source_tile_size", default=512, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("source_overlap", default=64, min=0, max=1024, step=8),
                io.Combo.Input("resize_method", options=["bicubic", "bilinear", "area", "nearest-exact"], default="bicubic"),
                io.Boolean.Input("clear_cache_per_tile", default=True),
                io.Boolean.Input("print_tiles", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="upscaled_image"),
                io.Int.Output(display_name="tile_count"),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_model, target_width, target_height, source_tile_size, source_overlap, resize_method, clear_cache_per_tile, print_tiles):
        return io.NodeOutput(*MrFlowTiledPixelUpscale().upscale(image, upscale_model, target_width, target_height, source_tile_size, source_overlap, resize_method, clear_cache_per_tile, print_tiles))


class MrFlowTiledVAEEncodeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowTiledVAEEncode",
            display_name="MrFlow Tiled VAE Encode",
            category="MrFlow/Tiled",
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Int.Input("tile_size", default=1024, min=128, max=MAX_RESOLUTION, step=16),
                io.Int.Input("overlap", default=128, min=0, max=1024, step=8),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, image, vae, tile_size, overlap):
        return io.NodeOutput(*MrFlowTiledVAEEncode().encode(image, vae, tile_size, overlap))


class MrFlowTiledVAEDecodeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowTiledVAEDecode",
            display_name="MrFlow Tiled VAE Decode",
            category="MrFlow/Tiled",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Int.Input("tile_size", default=1024, min=128, max=MAX_RESOLUTION, step=16),
                io.Int.Input("overlap", default=128, min=0, max=1024, step=8),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, latent, vae, tile_size, overlap):
        return io.NodeOutput(*MrFlowTiledVAEDecode().decode(latent, vae, tile_size, overlap))


class MrFlowLatentUpscaleAndNoiseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowLatentUpscaleAndNoise",
            display_name="MrFlow Latent Upscale + Noise",
            category="MrFlow/Tiled",
            inputs=[
                io.Latent.Input("latent"),
                io.Float.Input("upscale_factor", default=1.5, min=1.0, max=4.0, step=0.05),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "bicubic", "area"], default="bicubic"),
                io.Float.Input("noise_strength", default=0.0, min=0.0, max=2.0, step=0.01, round=0.001),
                io.Int.Input("seed", default=2026, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, upscale_factor, upscale_method, noise_strength, seed):
        return io.NodeOutput(*MrFlowLatentUpscaleAndNoise().upscale(latent, upscale_factor, upscale_method, noise_strength, seed))


class MrFlowLatentNoiseInjectNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowLatentNoiseInject",
            display_name="MrFlow Latent Noise Inject",
            category="MrFlow/Tiled",
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input("seed", default=2026, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Float.Input("noise_strength", default=0.08, min=0.0, max=2.0, step=0.01, round=0.001),
                io.Combo.Input("mode", options=["add", "lerp"], default="add"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, seed, noise_strength, mode):
        return io.NodeOutput(*MrFlowLatentNoiseInject().inject(latent, seed, noise_strength, mode))


class MrFlowTiledLatentRefineNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowTiledLatentRefine",
            display_name="MrFlow Tiled Latent Refine",
            category="MrFlow/Tiled",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("seed", default=2026, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=1, min=1, max=10000),
                io.Float.Input("cfg", default=0.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default=comfy.samplers.KSampler.SAMPLERS[0]),
                io.Float.Input("denoise", default=0.12, min=0.0, max=1.0, step=0.01, round=0.001),
                io.Int.Input("latent_tile_size", default=128, min=16, max=4096, step=8),
                io.Int.Input("latent_overlap", default=16, min=0, max=1024, step=4),
                io.Boolean.Input("decode_tiled", default=True),
                io.Int.Input("vae_tile_size", default=1024, min=128, max=MAX_RESOLUTION, step=16),
                io.Int.Input("vae_overlap", default=128, min=0, max=1024, step=8),
                io.Boolean.Input("clear_cache_per_tile", default=True),
                io.Boolean.Input("print_tiles", default=False),
            ],
            outputs=[
                io.Latent.Output(display_name="refined_latent"),
                io.Image.Output(display_name="refined_image"),
                io.Int.Output(display_name="tile_count"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, denoise, latent_tile_size, latent_overlap, decode_tiled, vae_tile_size, vae_overlap, clear_cache_per_tile, print_tiles):
        return io.NodeOutput(*MrFlowTiledLatentRefine().refine(model, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, denoise, latent_tile_size, latent_overlap, decode_tiled, vae_tile_size, vae_overlap, clear_cache_per_tile, print_tiles))


class MrFlowTiledSaveImageNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowTiledSaveImage",
            display_name="MrFlow Tiled Save Image",
            category="MrFlow/Tiled",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="MrFlow/Tiled/output"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix, prompt=None, extra_pnginfo=None):
        result = MrFlowTiledSaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return io.NodeOutput(result["result"][0], ui=result["ui"])


class MrFlowTiledExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MrFlowTiledPlanNode,
            MrFlowTiledPixelUpscaleNode,
            MrFlowTiledVAEEncodeNode,
            MrFlowTiledVAEDecodeNode,
            MrFlowLatentUpscaleAndNoiseNode,
            MrFlowLatentNoiseInjectNode,
            MrFlowTiledLatentRefineNode,
            MrFlowTiledSaveImageNode,
        ]


async def comfy_entrypoint() -> MrFlowTiledExtension:
    return MrFlowTiledExtension()
