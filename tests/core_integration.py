"""Real-core contract smoke test, no weights/inference. Pass ComfyUI root as argv[1]."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

core_root = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, str(core_root))
import comfy.options
comfy.options.enable_args_parsing()
import nodes as core
import comfy.sd
import comfy.supported_models as supported

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uml_integration", root / "__init__.py", submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
loader = sys.modules["uml_integration.nodes"]
auto = sys.modules["uml_integration.clip_auto"]
assert loader.COMFY_NODES is core
schema = loader.UniversalModelLoader.INPUT_TYPES()["required"]
assert list(schema) == ["model_type", "checkpoint_name", "diffusion_model_name", "diffusers_model_path", "gguf_unet_name", "weight_dtype", "gguf_dequant_dtype", "gguf_patch_dtype", "gguf_patch_on_device", "clip_name", "clip_type", "clip_device", "vae_name", "clip2_name", "vae2_name", "clip2_type", "clip2_device"]
core_types = set(core.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
assert set(schema["clip_type"][0]) == core_types | {"auto"}
assert schema["clip2_name"][1]["default"] == "none"
assert schema["vae2_name"][1]["default"] == "none"
assert package.__version__ == "1.2.0"
assert loader.UniversalModelLoader.RETURN_TYPES == ("MODEL", "CLIP", "VAE", "STRING", "CLIP", "VAE")
count = 0
for name, expected in auto.RUNTIME_FAMILIES.items():
    cls = getattr(supported, name)
    config = cls.__new__(cls)  # Real class/MRO without allocating a diffusion model.
    model = SimpleNamespace(model=SimpleNamespace(model_config=config))
    assert auto.runtime_family(model) == expected, name
    assert expected in core_types, (name, expected)
    count += 1
with patch.object(loader.folder_paths, "get_full_path_or_raise", return_value="/contract/encoder.safetensors"), patch.object(comfy.sd, "load_clip", return_value="CLIP sentinel") as load:
    model = SimpleNamespace(model=SimpleNamespace(model_config=supported.Krea2.__new__(supported.Krea2)))
    clip, family = loader._load_clip("renamed_model", "qwen3.safetensors", "auto", "cpu", model)
    assert (clip, family) == ("CLIP sentinel", "krea2")
    assert load.call_args.kwargs["clip_type"] == comfy.sd.CLIPType.KREA2
    assert str(load.call_args.kwargs["model_options"]["load_device"]) == "cpu"
print(f"PASS: real core import/schema, {count} supported config classes, real CLIPLoader enum/device forwarding (weight I/O mocked; no inference)")
