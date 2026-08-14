# ComfyUI Universal Model Loader

![Universal MODEL Loader node screenshot](assets/universal-model-loader.png)

Author: Zoltar358

Version: 1.0.6

A local ComfyUI custom node pack that provides one loader node with a first-step `model_type` selector. The browser UI hides irrelevant parameters after you choose the type, while optional second CLIP and VAE selectors support workflows that need additional encoders or VAEs.

## Installation

### ComfyUI Manager

This nodepack has been submitted for ComfyUI Manager listing. After it is accepted, install it from:

1. Open ComfyUI.
2. Click **Manager**.
3. Click **Install Custom Nodes**.
4. Search for **ComfyUI-Universal-Model-Loader**.
5. Click **Install** and restart ComfyUI.

### Comfy Registry / comfy-cli

After the registry package is published, install it with:

```bash
comfy node install universal-model-loader
```

### Manual install

From your `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/Zoltar358-ComfyUI/ComfyUI-Universal-Model-Loader.git
```

Then restart ComfyUI and hard-refresh the browser page.

### Updating

From the installed nodepack directory:

```bash
git pull
```

Then restart ComfyUI and hard-refresh the browser page.

### Optional dependency for GGUF mode

`gguf_unet` mode requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
```

## Node

**Universal MODEL Loader** (`model/loaders`)

Outputs:

1. `model` — `MODEL`
2. `clip` — `CLIP`
3. `vae` — `VAE`
4. `loaded_info` — `STRING`
5. `clip2` — `CLIP` optional second CLIP/text encoder, or `none`
6. `vae2` — `VAE` optional second VAE, or `none`

Supported modes:

- `checkpoint` — loads from `models/checkpoints`, returns MODEL/CLIP/VAE.
- `diffusers` — loads a folder under `models/diffusers` containing `model_index.json`, returns MODEL/CLIP/VAE.
- `diffusion_model` / `unet` — loads from `models/unet` or `models/diffusion_models`, returns MODEL, CLIP and VAE.
- `gguf_unet` — loads `.gguf` diffusion models using ComfyUI-GGUF, returns MODEL, CLIP and VAE.

The node uses a purple/dark-teal theme matching the screenshot you provided. The frontend extension collapses every widget not relevant to the selected `model_type`; MODEL-only types always load CLIP and VAE from this node.

Optional `clip2` and `vae2` selectors default to `none`, so existing one-CLIP/one-VAE workflows remain simple. Select a second text encoder or VAE only for model families/workflows that require one.

Auto CLIP type selection recognizes current ComfyUI model-family hints including Krea2/KR2, Qwen Image, Hunyuan Image, Ideogram 4, Boogu, JoyImage, Mage, MiniMax/MiniMax Music3/MM3/RVQ, LongCat Image, PixelDiT, Omnigen2, Flux.2/FK9, Wan, HiDream, Chroma, Ovis, Lens, CogVideoX, Cosmos, LTXV, Mochi, PixArt, Lumina2, ACE, SD3, Stable Audio, and Stable Cascade when those CLIP types are available in the installed ComfyUI build.

## 1.0.2 Node Pack Info update

Version 1.0.2 adds richer Comfy Registry metadata for ComfyUI Extensions' **Node Pack Info** panel:

- Registry icon and banner assets.
- README, author, keyword and project URL metadata in `pyproject.toml`.
- Hardened node metadata extraction so the Registry can parse and render the node preview card even in lean extraction environments.
- A clearer node description for the preview card in the **NODES** section.

Version 1.0.3 hardens the Registry extraction path further by avoiding ambiguity between this package's `nodes.py` and ComfyUI core `nodes.py` during node schema discovery.

Version 1.0.4 adds optional second CLIP and VAE loading with new `clip2` and `vae2` outputs. The extra selectors default to `none` for workflows that only need the original primary CLIP/VAE.

Version 1.0.5 makes the optional VAE selector easier to see by placing `vae2_name` directly after `clip2_name`, keeping it visible for every mode, and hiding `clip2_type`/`clip2_device` until a real second CLIP is selected.

Version 1.0.6 improves compatibility with latest MiniMax Music3 naming by auto-selecting ComfyUI CLIP type `minimax` for `MM3`, `Music3`, `RVQ`, and `qwen-rvq` file/folder hints. For Music3 workflows, pair the loaded MODEL/CLIP/VAE with ComfyUI's built-in **MiniMax Music3 Text Encode** and **Empty MiniMax Music3 Latent Audio** nodes.

## Notes

For MODEL-only modes, choose the CLIP/text encoder and VAE directly in the Universal MODEL Loader.

GGUF mode requires `custom_nodes/ComfyUI-GGUF` to be installed.
