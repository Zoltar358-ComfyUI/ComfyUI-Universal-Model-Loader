"""Typed adapter for moss-image-t8's independent official component loaders.

These descriptors are NOT ComfyUI MODEL/CLIP/VAE objects. Never disguise them
with wildcard sockets or route them through core UNETLoader/CLIPLoader.
"""
import importlib

from .model_name import model_name

COMPONENTS = (
    ("MOSSOfficialDiffusionLoader", "MOSS_IMAGE_DIT", "diffusion_models"),
    ("MOSSOfficialTextEncoderLoader", "MOSS_IMAGE_TE", "text_encoders"),
    ("MOSSOfficialVAELoader", "MOSS_IMAGE_VAE", "vae"),
)


def _loaders():
    core = importlib.import_module("nodes")
    mapping = getattr(core, "NODE_CLASS_MAPPINGS", {})
    classes = []
    for name, socket, _folder in COMPONENTS:
        cls = mapping.get(name)
        if cls is None:
            raise RuntimeError("Universal MOSS-Image Loader requires the moss-image-t8 "
                               f"integration ({name}). Install/enable it and restart ComfyUI.")
        if tuple(cls.RETURN_TYPES) != (socket,) or cls.FUNCTION != "load":
            raise RuntimeError(f"Incompatible {name} schema; expected {socket} and load().")
        classes.append(cls)
    return classes


class UniversalMOSSLoader:
    RETURN_TYPES = ("MOSS_IMAGE_DIT", "STRING", "MOSS_IMAGE_TE", "MOSS_IMAGE_VAE")
    RETURN_NAMES = ("model", "model_name", "text_encoder", "vae")
    FUNCTION = "load_model"
    CATEGORY = "model/loaders"
    DESCRIPTION = ("Loads independently exported MOSS-Image components using moss-image-t8. "
                   "Connect to its Official Text Encode, Sampler and VAE Decode, not KSampler. "
                   "No automatic downloads, conversion or VAE substitution.")

    @classmethod
    def INPUT_TYPES(cls):
        try:
            schemas = [loader.INPUT_TYPES()["required"] for loader in _loaders()]
            return {"required": {
                "diffusion_model_name": schemas[0]["model_name"],
                "text_encoder_name": schemas[1]["model_name"],
                "vae_name": schemas[2]["model_name"],
                "device": schemas[0]["device"],
                "precision": schemas[0]["precision"],
            }}
        except RuntimeError:
            # Keep registry parsing/core startup safe without this optional pack.
            import folder_paths
            def names(folder):
                try:
                    return (folder_paths.get_filename_list(folder) or ["— none found —"],)
                except Exception:
                    return (["— none found —"],)
            return {"required": {
                "diffusion_model_name": names("diffusion_models"),
                "text_encoder_name": names("text_encoders"), "vae_name": names("vae"),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
            }}

    @classmethod
    def IS_CHANGED(cls, diffusion_model_name, text_encoder_name, vae_name, **kwargs):
        return tuple(loader.IS_CHANGED(model_name=name) for loader, name in
                     zip(_loaders(), (diffusion_model_name, text_encoder_name, vae_name)))

    def load_model(self, diffusion_model_name, text_encoder_name, vae_name,
                   device="auto", precision="bf16"):
        dit, te, vae = _loaders()
        model = dit().load(diffusion_model_name, device=device, precision=precision)[0]
        encoder = te().load(text_encoder_name, device=device)[0]
        decoder = vae().load(vae_name, device=device)[0]
        return (model, model_name(diffusion_model_name), encoder, decoder)
