import importlib.util
import inspect
import os
import sys

import torch

import comfy.sd
import folder_paths
import nodes


NONE_ITEM = "— none found —"
MODEL_TYPES = ["checkpoint", "diffusers", "diffusion_model", "unet", "gguf_unet"]
MODEL_ONLY_TYPES = {"diffusion_model", "unet", "gguf_unet"}
WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
GGUF_DTYPES = ["default", "target", "float32", "float16", "bfloat16"]


def _filename_list(folder_key, extra_extensions=None):
    try:
        names = list(folder_paths.get_filename_list(folder_key))
    except Exception:
        names = []

    if extra_extensions:
        existing = set(names)
        for base in folder_paths.get_folder_paths(folder_key):
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base, followlinks=True):
                for file_name in files:
                    if os.path.splitext(file_name)[1].lower() not in extra_extensions:
                        continue
                    rel = os.path.relpath(os.path.join(root, file_name), base)
                    existing.add(rel)
        names = sorted(existing)

    return names or [NONE_ITEM]


def _diffusers_list():
    paths = []
    try:
        search_paths = folder_paths.get_folder_paths("diffusers")
    except Exception:
        search_paths = []

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, _dirs, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
    return sorted(paths) or [NONE_ITEM]


def _clip_types():
    try:
        clip_types = list(nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
    except Exception:
        clip_types = ["stable_diffusion"]
    if "auto" not in clip_types:
        clip_types.insert(0, "auto")
    return (clip_types, {"default": "auto"})


def _is_krea2_hint(hint):
    hint = hint.lower()
    tokens = hint.replace("\\", "/").replace("_", "-")
    return (
        "krea2" in hint
        or "krea-2" in hint
        or "krea_2" in hint
        or tokens.startswith("kr2/")
        or "/kr2/" in tokens
        or "[kr2]" in hint
        or " kr2" in hint
    )


def _effective_clip_type(model_hint, clip_type):
    hint = model_hint.lower()
    if clip_type != "auto" and not (clip_type == "stable_diffusion" and _is_krea2_hint(model_hint)):
        return clip_type

    if _is_krea2_hint(model_hint):
        return "krea2"
    if "qwen_image" in hint or "qwen-image" in hint or "qwenimage" in hint:
        return "qwen_image"
    if "wan" in hint:
        return "wan"
    if "hidream" in hint:
        return "hidream"
    if "flux2" in hint or "flux-2" in hint:
        return "flux2"
    if "chroma" in hint:
        return "chroma"
    return "stable_diffusion"


def _vae_names():
    try:
        return (nodes.VAELoader.vae_list(nodes.VAELoader),)
    except Exception:
        return (_filename_list("vae"),)


def _require_choice(kind, value):
    if not value or value == NONE_ITEM:
        raise ValueError(f"No {kind} was selected/found. Put a compatible model in ComfyUI/models/{kind} and Refresh the browser.")


def _load_diffusers(model_path):
    try:
        import comfy.diffusers_load
    except Exception as exc:
        raise RuntimeError("This ComfyUI build does not expose comfy.diffusers_load.") from exc

    _require_choice("diffusers", model_path)
    resolved = model_path
    for search_path in folder_paths.get_folder_paths("diffusers"):
        path = os.path.join(search_path, model_path)
        if os.path.exists(path):
            resolved = path
            break
    return comfy.diffusers_load.load_diffusers(
        resolved,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )[:3]


def _load_clip(model_hint, clip_name, clip_type, clip_device):
    _require_choice("text_encoders", clip_name)
    device = "cpu" if clip_device == "cpu" else "default"
    resolved_clip_type = _effective_clip_type(model_hint, clip_type)
    return nodes.CLIPLoader().load_clip(clip_name, resolved_clip_type, device)[0], resolved_clip_type


def _load_vae(vae_name):
    _require_choice("vae", vae_name)
    return nodes.VAELoader().load_vae(vae_name)[0]


def _resolve_aux_models(model_type, model_hint, clip_name, clip_type, clip_device, vae_name):
    if model_type not in MODEL_ONLY_TYPES:
        return None, None, []

    clip, resolved_clip_type = _load_clip(model_hint, clip_name, clip_type, clip_device)
    vae = _load_vae(vae_name)
    loaded = [f"clip: {clip_name} ({resolved_clip_type})", f"vae: {vae_name}"]
    return clip, vae, loaded


def _gguf_nodes_module():
    package_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-GGUF")
    init_file = os.path.join(package_dir, "__init__.py")
    if not os.path.exists(init_file):
        raise RuntimeError("GGUF loading requires the ComfyUI-GGUF custom node pack to be installed.")

    package_name = "comfyui_gguf_universal_loader_bridge"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(package_name, init_file, submodule_search_locations=[package_dir])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import ComfyUI-GGUF from {package_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(package_name, None)
            raise RuntimeError(f"Could not import ComfyUI-GGUF from {package_dir}: {exc}") from exc

    return importlib.import_module(f"{package_name}.nodes")


def _load_gguf_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device):
    _require_choice("diffusion_models", unet_name)
    gguf_nodes = _gguf_nodes_module()

    ops = gguf_nodes.GGMLOps()
    if dequant_dtype in ("default", None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype == "target":
        ops.Linear.dequant_dtype = dequant_dtype
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

    if patch_dtype in ("default", None):
        ops.Linear.patch_dtype = None
    elif patch_dtype == "target":
        ops.Linear.patch_dtype = patch_dtype
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype)

    unet_path = folder_paths.get_full_path("unet_gguf", unet_name) or folder_paths.get_full_path("diffusion_models", unet_name)
    if unet_path is None:
        raise FileNotFoundError(f"Could not resolve GGUF diffusion model: {unet_name}")

    sd, extra = gguf_nodes.gguf_sd_loader(unet_path)
    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops}, **kwargs)
    if model is None:
        raise RuntimeError(f"Could not detect model type of GGUF diffusion model: {unet_path}")
    model = gguf_nodes.GGUFModelPatcher.clone(model)
    model.patch_on_device = patch_on_device
    return model


class UniversalModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (MODEL_TYPES, {"default": "checkpoint", "tooltip": "Choose which backend to load. The browser extension hides irrelevant fields after this selection."}),
                "checkpoint_name": (_filename_list("checkpoints"), {"tooltip": "Used when model_type = checkpoint. Returns MODEL, CLIP and VAE."}),
                "diffusion_model_name": (_filename_list("diffusion_models"), {"tooltip": "Used when model_type = diffusion_model or unet. Returns MODEL, CLIP and VAE."}),
                "diffusers_model_path": (_diffusers_list(), {"tooltip": "Used when model_type = diffusers. Folder under models/diffusers containing model_index.json."}),
                "gguf_unet_name": (_filename_list("diffusion_models", {".gguf"}), {"tooltip": "Used when model_type = gguf_unet. Requires ComfyUI-GGUF."}),
                "weight_dtype": (WEIGHT_DTYPES, {"default": "default", "tooltip": "Only used for diffusion_model/unet loading."}),
                "gguf_dequant_dtype": (GGUF_DTYPES, {"default": "default", "tooltip": "Only used for gguf_unet."}),
                "gguf_patch_dtype": (GGUF_DTYPES, {"default": "default", "tooltip": "Only used for gguf_unet."}),
                "gguf_patch_on_device": ("BOOLEAN", {"default": False, "tooltip": "Only used for gguf_unet."}),
                "clip_name": (_filename_list("text_encoders"), {"tooltip": "CLIP/text encoder loaded for diffusion_model, unet, and gguf_unet."}),
                "clip_type": _clip_types(),
                "clip_device": (["default", "cpu"], {"default": "default", "advanced": True, "tooltip": "Device for the loaded CLIP/text encoder."}),
                "vae_name": _vae_names(),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "loaded_info")
    FUNCTION = "load_model"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Universal front-end model loader: checkpoint, diffusers, diffusion/unet, GGUF unet, optional CLIP, and optional VAE from one node."
    SEARCH_ALIASES = ["universal model loader", "checkpoint loader", "unet loader", "gguf loader", "diffusers loader", "clip loader", "vae loader"]

    def load_model(
        self,
        model_type,
        checkpoint_name,
        diffusion_model_name,
        diffusers_model_path,
        gguf_unet_name,
        weight_dtype,
        gguf_dequant_dtype,
        gguf_patch_dtype,
        gguf_patch_on_device,
        clip_name,
        clip_type,
        clip_device,
        vae_name,
    ):
        if model_type == "checkpoint":
            _require_choice("checkpoints", checkpoint_name)
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_name)
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )[:3]
            return (*out, f"checkpoint: {checkpoint_name}")

        if model_type == "diffusers":
            model, clip, vae = _load_diffusers(diffusers_model_path)
            return (model, clip, vae, f"diffusers: {diffusers_model_path}")

        if model_type in ("diffusion_model", "unet"):
            _require_choice("diffusion_models", diffusion_model_name)
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", diffusion_model_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            clip, vae, aux_info = _resolve_aux_models(model_type, diffusion_model_name, clip_name, clip_type, clip_device, vae_name)
            info = f"{model_type}: {diffusion_model_name}"
            if aux_info:
                info += " | " + " | ".join(aux_info)
            return (model, clip, vae, info)

        if model_type == "gguf_unet":
            model = _load_gguf_unet(gguf_unet_name, gguf_dequant_dtype, gguf_patch_dtype, gguf_patch_on_device)
            clip, vae, aux_info = _resolve_aux_models(model_type, gguf_unet_name, clip_name, clip_type, clip_device, vae_name)
            info = f"gguf_unet: {gguf_unet_name}"
            if aux_info:
                info += " | " + " | ".join(aux_info)
            return (model, clip, vae, info)

        raise ValueError(f"Unsupported model_type: {model_type}")


NODE_CLASS_MAPPINGS = {
    "UniversalModelLoader": UniversalModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalModelLoader": "Universal MODEL Loader",
}
