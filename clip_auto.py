"""Conservative CLIP family resolution without loading encoder weights."""
import re

# Specific names only: generic Qwen/Gemma/T5 encoders serve several families.
ALIASES = {
    "krea2": ("krea2", "krea-2", "kr2"),
    "qwen_image": ("qwen-image", "qwenimage", "qwen-image21", "qwenimage21", "qwen2.1", "qwen-2.1", "qwn2"),
    "hunyuan_image": ("hunyuan-image",),
    "ideogram4": ("ideogram4", "ideogram-4", "ideo4", "ideo"),
    "boogu": ("boogu", "boog"), "joyimage": ("joyimage", "joy-image", "joy"),
    "mage": ("mage", "mageflow"),
    "minimax": ("minimax", "mini-max", "music3", "music-3", "mm3", "rvq"),
    "longcat_image": ("longcat-image", "longcatimage", "long-cat"),
    "pixeldit": ("pixeldit", "pixel-dit"), "omnigen2": ("omnigen2", "omnigen-2"),
    "flux2": ("flux2", "flux-2", "flux.2", "fk9"),
    "wan": ("wan", "wan2", "wan-2", "wan2.1", "wan2.2", "wan21", "wan22", "umt5"),
    "hidream": ("hidream", "hi-dream"), "chroma": ("chroma",),
    "ovis": ("ovis",), "lens": ("lens",),
    "cogvideox": ("cogvideox", "cogvideo-x"), "cosmos": ("cosmos",),
    "ltxv": ("ltxv", "ltx-video", "ltx-2", "ltx2"), "mochi": ("mochi",),
    "pixart": ("pixart", "pix-art"), "lumina2": ("lumina2", "lumina-2"),
    "ace": ("ace", "ace-step", "acestep"),
    "sd3": ("sd3", "sd3.5", "sd-3", "stable-diffusion-3"),
    "stable_audio": ("stable-audio",), "stable_cascade": ("stable-cascade",),
    "yue2": ("yue2", "yue-2"),
}

def _normalize(value):
    return str(value or "").lower().replace("\\", "/").replace("_", "-").replace(" ", "-")

PATTERNS = tuple((family, re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, aliases)) + r")(?![a-z0-9])")) for family, aliases in ALIASES.items())

# Names of installed ComfyUI supported_models configs; MRO handles derivatives.
RUNTIME_FAMILIES = {
    "Krea2": "krea2", "QwenImage": "qwen_image", "QwenImage21": "qwen_image",
    "HunyuanImage21": "hunyuan_image", "HunyuanImage21Refiner": "hunyuan_image",
    "Ideogram4": "ideogram4", "Boogu": "boogu", "JoyImage": "joyimage",
    "MageFlow": "mage", "MiniMaxH3": "minimax", "MiniMaxMusic3": "minimax",
    "LongCatImage": "longcat_image", "PixelDiTT2I": "pixeldit", "Omnigen2": "omnigen2",
    "Flux2": "flux2", "WAN21_T2V": "wan", "HiDream": "hidream", "Chroma": "chroma",
    "Lens": "lens", "CogVideoX_T2V": "cogvideox", "CosmosT2V": "cosmos",
    "CosmosT2IPredict2": "cosmos", "LTXV": "ltxv", "GenmoMochi": "mochi",
    "PixArtAlpha": "pixart", "Lumina2": "lumina2", "ACEStep": "ace", "ACEStep15": "ace",
    "SD3": "sd3", "StableAudio": "stable_audio", "Stable_Cascade_C": "stable_cascade",
    "YuE2": "yue2", "ZImage": "stable_diffusion", "Flux": "stable_diffusion",
    "SD15": "stable_diffusion", "SD20": "stable_diffusion", "SDXL": "stable_diffusion",
}

def family_hint(value):
    normalized = _normalize(value)
    return next((family for family, pattern in PATTERNS if pattern.search(normalized)), None)


def runtime_family(model):
    config = getattr(getattr(model, "model", None), "model_config", None)
    for cls in type(config).__mro__:
        if cls.__name__ in RUNTIME_FAMILIES:
            return RUNTIME_FAMILIES[cls.__name__]
    return None


def resolve_clip_type(model_hint, requested, encoder_name, available, model=None):
    if requested != "auto":
        return requested  # Never reinterpret an explicit user selection.
    family = family_hint(encoder_name) or runtime_family(model) or family_hint(model_hint)
    if family is None:
        if "qwen" in _normalize(encoder_name):
            raise ValueError("Ambiguous Qwen encoder: choose clip_type explicitly or use a recognized model family.")
        family = "stable_diffusion"
    if family not in available:
        raise ValueError(f"Auto CLIP type '{family}' is unavailable in this ComfyUI build; update ComfyUI or select clip_type explicitly.")
    return family
