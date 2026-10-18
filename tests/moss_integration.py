"""Real MOSS component-loader validation; no mocks/downloads/inference.

Usage: ComfyUI/.venv/bin/python tests/moss_integration.py CORE_ROOT SHARED_MODELS
This invokes the dependency's mandatory integrity checks (including hashes).
The model_name output itself remains path-only and performs no I/O.
"""
import importlib.util
from pathlib import Path
import sys

core_root, models = map(lambda p: Path(p).resolve(), sys.argv[1:3])
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, str(core_root))
import comfy.options
comfy.options.enable_args_parsing()
import nodes as core
import folder_paths


def load_package(name, root):
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for folder in ("diffusion_models", "text_encoders", "vae"):
    folder_paths.add_model_folder_path(folder, str(models / folder))
moss = load_package("moss_real_integration", core_root / "custom_nodes/moss-image-t8")
core.NODE_CLASS_MAPPINGS.update(moss.NODE_CLASS_MAPPINGS)
uml = load_package("uml_real_moss", Path(__file__).resolve().parents[1])
node = uml.NODE_CLASS_MAPPINGS["UniversalMOSSLoader"]()
schema = node.INPUT_TYPES()["required"]
assert "moss" not in core.CLIPLoader.INPUT_TYPES()["required"]["type"][0]
print("Core has no moss CLIP type; adapter uses exact MOSS_IMAGE_* contracts", flush=True)
# Reproduce the reported universal-node route against real core (no weight mocks).
core_loader = uml.NODE_CLASS_MAPPINGS["UniversalModelLoaderV2"]()
kwargs = {}
for key, entry in core_loader.INPUT_TYPES()["required"].items():
    options = entry[1] if len(entry) > 1 else {}
    kwargs[key] = options.get("default", entry[0][0] if isinstance(entry[0], list) else False)
kwargs.update(model_type="diffusion_model", diffusion_model_name="MI/mossImage_sft_pruned_int8.safetensors",
              clip_name="MI/moss-image-qwen35-bf16.safetensors", vae_name="MI/moss-image-vae.safetensors")
try:
    core_loader.load_model(**kwargs)
except ValueError as exc:
    assert "Add MOSS-Image loader" in str(exc)
    print("PASS reported universal route rejected with typed-routing guidance before core loading", flush=True)
else:
    raise AssertionError("MOSS incorrectly routed through core")
for dit in ("MI/moss-image-sft-bf16.safetensors", "MIB/moss-image-base-bf16.safetensors"):
    te, vae = "MI/moss-image-qwen35-bf16.safetensors", "MI/moss-image-vae.safetensors"
    for key, name in (("diffusion_model_name", dit), ("text_encoder_name", te), ("vae_name", vae)):
        assert name in schema[key][0], (key, name)
    result = node.load_model(dit, te, vae, device="cpu", precision="bf16")
    assert [result[i].kind for i in (0, 2, 3)] == ["transformer", "text_encoder", "vae"]
    assert result[1] == Path(dit).stem
    assert all(result[i].device == "cpu" for i in (0, 2, 3))
    assert len(node.IS_CHANGED(dit, te, vae)) == 3
    print(f"PASS actual MOSS loaders: {dit} + {te} + {vae}; model_name={result[1]}", flush=True)
try:
    node.load_model("MI/mossImage_sft_pruned_int8.safetensors", te, vae, device="cpu")
except ValueError as exc:
    assert "independently exported" in str(exc)
    print("PASS incompatible INT8 rejected by real dependency:", exc, flush=True)
else:
    raise AssertionError("Unsupported raw INT8 unexpectedly accepted")
print("PASS real local component validation; no tensor allocation or generation", flush=True)
