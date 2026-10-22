"""Opt-in real CPU weight-loading smoke test. No server mutations or downloads.
Usage: ComfyUI-python tests/native_moss_load.py CORE_ROOT SHARED_MODELS
"""
import importlib.util
from pathlib import Path
import sys

infer = "--infer" in sys.argv[3:]
core_root, models = map(Path, sys.argv[1:3])
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, str(core_root))
import comfy.options
comfy.options.enable_args_parsing()
import torch
torch.set_num_threads(4)
import folder_paths
import nodes as core
for kind in ("diffusion_models", "text_encoders", "vae"):
    folder_paths.add_model_folder_path(kind, str(models / kind))
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uml_native", root / "__init__.py", submodule_search_locations=[str(root)])
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
loader = package.NODE_CLASS_MAPPINGS["UniversalModelLoaderV2"]()
result = loader.load_model(
    model_type="diffusion_model", checkpoint_name="unused",
    diffusion_model_name="MI/MOSS Image (int8 ConvRot).safetensors",
    diffusers_model_path="unused", gguf_unet_name="unused", weight_dtype="default",
    gguf_dequant_dtype="default", gguf_patch_dtype="default", gguf_patch_on_device=False,
    clip_name="MI/qwen35ForMossImage_v10_int8.safetensors", clip_type="auto", clip_device="cpu",
    vae_name="MI/moss-image-vae.safetensors", clip2_name="none", vae2_name="none",
    clip2_type="auto", clip2_device="cpu")
assert type(result[0].model).__name__ == "MOSSImage"
assert type(result[2].cond_stage_model).__name__ == "MOSSImageTEModel_"
assert type(result[3].first_stage_model).__name__ == "AutoencoderKL"
assert result[1] == "MOSS Image (int8 ConvRot)"
assert result[5:] == (None, None)
assert "(moss_image)" in result[4]
print("PASS: real CPU MOSS MODEL/CLIP/VAE weight loading through UniversalModelLoaderV2", flush=True)
print("Classes:", type(result[0].model).__name__, type(result[2].cond_stage_model).__name__, type(result[3].first_stage_model).__name__, flush=True)
if infer:
    with torch.no_grad():
        positive = core.CLIPTextEncode().encode(result[2], "A red cube on a white table.")[0]
        print("PASS: native MOSS text encoding", flush=True)
        latent = {"samples": torch.zeros(1, 128, 8, 8)}
        sampled = core.KSampler().sample(result[0], 1234, 1, 1.0, "euler", "simple", positive, positive, latent)[0]
        image = core.VAEDecode().decode(result[3], sampled)[0]
        assert torch.isfinite(image).all()
        assert tuple(image.shape) == (1, 128, 128, 3)
        from PIL import Image
        import os
        output = Path(os.environ["TMPDIR"]) / "uml-native-moss-smoke.png"
        Image.fromarray((image[0].clamp(0, 1).numpy() * 255).astype("uint8")).save(output)
        print(f"PASS: 1-step CPU sampling + VAE decode, finite {tuple(image.shape)}; artifact: {output}", flush=True)
# Dispose patchers before interpreter module teardown.
import gc
del result
gc.collect()
