"""Run with python -m unittest discover -s tests -v (no ComfyUI required)."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clip_auto", ROOT / "clip_auto.py")
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)
AVAILABLE = set(auto.ALIASES) | {"stable_diffusion"}


def model_for(name):
    return types.SimpleNamespace(model=types.SimpleNamespace(model_config=type(name, (), {})()))


class AutoTests(unittest.TestCase):
    def test_all_aliases_and_windows_paths(self):
        for family, aliases in auto.ALIASES.items():
            for alias in aliases:
                with self.subTest(alias=alias):
                    self.assertEqual(auto.family_hint(f"C:\\models\\{alias}\\weights.safetensors"), family)
                    self.assertEqual(auto.family_hint(f"[{alias}]_v1.safetensors"), family)

    def test_false_positives_and_ambiguous_names(self):
        for name in ["image", "landscape", "swan", "hunyuan_video", "qwen_2.5_vl", "gemma", "t5", "lensflare"]:
            self.assertIsNone(auto.family_hint(name), name)
        with self.assertRaisesRegex(ValueError, "Ambiguous Qwen"):
            auto.resolve_clip_type("unknown", "auto", "qwen3.safetensors", AVAILABLE)

    def test_model_switch_and_manual_override(self):
        for model, expected in [("KR2/model", "krea2"), ("QWN2/model", "qwen_image"), ("wan_2.2", "wan")]:
            self.assertEqual(auto.resolve_clip_type(model, "auto", "encoder", AVAILABLE), expected)
            self.assertEqual(auto.resolve_clip_type(model, "stable_diffusion", "encoder", AVAILABLE), "stable_diffusion")

    def test_runtime_context_and_encoder_specificity(self):
        model = model_for("Krea2")
        self.assertEqual(auto.resolve_clip_type("QWN2/misnamed", "auto", "qwen3", AVAILABLE, model), "krea2")
        self.assertEqual(auto.resolve_clip_type("KR2/model", "auto", "qwen_rvq", AVAILABLE, model), "minimax")
        self.assertEqual(auto.resolve_clip_type("unknown", "auto", "qwen3", AVAILABLE, model_for("ZImage")), "stable_diffusion")

    def test_unsupported_family_does_not_silently_fallback(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            auto.resolve_clip_type("KR2/model", "auto", "encoder", {"stable_diffusion"})


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.core = types.ModuleType("nodes")
        self.core.CLIPLoader = Mock()
        self.core.CLIPLoader.INPUT_TYPES.return_value = {"required": {"type": (sorted(AVAILABLE),)}}
        self.core.CLIPLoader.return_value.load_clip.side_effect = lambda *args: (args,)
        self.core.VAELoader = Mock()
        self.core.VAELoader.return_value.load_vae.side_effect = lambda name: (name,)
        self.core.UNETLoader = Mock()
        self.model = model_for("Krea2")
        self.core.UNETLoader.return_value.load_unet.return_value = (self.model,)
        folder = types.ModuleType("folder_paths")
        folder.get_full_path_or_raise = lambda kind, name: name
        folder.get_folder_paths = lambda kind: []
        comfy = types.ModuleType("comfy")
        comfy.sd = types.ModuleType("comfy.sd")
        comfy.sd.load_checkpoint_guess_config = Mock(return_value=(self.model, "bundled_clip", "bundled_vae"))
        package = types.ModuleType("uml_test")
        package.__path__ = [str(ROOT)]
        self.modules = patch.dict(sys.modules, {"nodes": self.core, "folder_paths": folder, "comfy": comfy, "comfy.sd": comfy.sd, "torch": types.ModuleType("torch"), "uml_test": package})
        self.modules.start()
        spec = importlib.util.spec_from_file_location("uml_test.nodes", ROOT / "nodes.py")
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)
        self.addCleanup(self.modules.stop)

    def test_aux_preserves_vae_no_discovery_and_encoder_choice(self):
        clip, vae, info = self.loader._resolve_aux_models("unet", "KR2/model", "qwen3", "auto", "cpu", "my_vae", self.model)
        self.assertEqual(clip, ("qwen3", "krea2", "cpu"))
        self.assertEqual(vae, "my_vae")
        self.assertIn("vae: my_vae", info)
        self.core.VAELoader.vae_list.assert_not_called()
        # Cached schema lookup avoids repeated encoder-directory discovery.
        self.loader._load_clip("KR2/model", "qwen3", "auto", "cpu", self.model)
        self.core.CLIPLoader.INPUT_TYPES.assert_called_once()
        secondary, _, _ = self.loader._resolve_optional_aux_models("KR2/model", "qwen_rvq", "auto", "default", "none", self.model)
        self.assertEqual(secondary[1], "minimax")

    def test_optional_none_does_not_load(self):
        self.assertEqual(self.loader._resolve_optional_aux_models("", "none", "auto", "cpu", "none"), (None, None, []))
        self.core.CLIPLoader.return_value.load_clip.assert_not_called()
        self.core.VAELoader.return_value.load_vae.assert_not_called()

    def test_routes_and_output_compatibility(self):
        kwargs = dict(checkpoint_name="checkpoint", diffusion_model_name="renamed", diffusers_model_path="diffusers", gguf_unet_name="gguf", weight_dtype="fp8_e4m3fn", gguf_dequant_dtype="default", gguf_patch_dtype="default", gguf_patch_on_device=False, clip_name="qwen3", clip_type="auto", clip_device="cpu", vae_name="explicit_vae", clip2_name="none", vae2_name="none", clip2_type="auto", clip2_device="default")
        self.loader._load_diffusers = Mock(return_value=(self.model, "bundled_clip", "bundled_vae"))
        self.loader._load_gguf_unet = Mock(return_value=self.model)
        self.assertEqual(self.loader.UniversalModelLoader.RETURN_NAMES, ("model", "clip", "vae", "loaded_info", "clip2", "vae2"))
        for mode in self.loader.MODEL_TYPES:
            with self.subTest(mode=mode):
                result = self.loader.UniversalModelLoader().load_model(model_type=mode, **kwargs)
                self.assertEqual(len(result), 6)
                self.assertIs(result[0], self.model)
                self.assertEqual(result[4:], (None, None))
                if mode in self.loader.MODEL_ONLY_TYPES:
                    self.assertEqual(result[1], ("qwen3", "krea2", "cpu"))
                    self.assertEqual(result[2], "explicit_vae")
        self.core.UNETLoader.return_value.load_unet.assert_called_with("renamed", "fp8_e4m3fn")


if __name__ == "__main__":
    unittest.main()
