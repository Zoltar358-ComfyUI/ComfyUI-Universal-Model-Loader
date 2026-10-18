# Universal Model Loader 1.3.0

- Add `model_name` directly after MODEL in the new `UniversalModelLoaderV2` output contract. Uses the selected filename without its final extension, or the Diffusers folder basename; no extra hashing, downloads or file I/O.
- Preserve original `UniversalModelLoader` API indices. Recognized legacy UI graphs migrate their class, sockets and outgoing links together before configuration; unknown schemas remain on the legacy contract.
- Add `UniversalMOSSLoader`, a typed adapter for the installed `moss-image-t8` official independent component loaders. Outputs MOSS_IMAGE_DIT, model_name, MOSS_IMAGE_TE and MOSS_IMAGE_VAE. Requires the corresponding optional pack; no fabricated native MOSS CLIP type or wildcard socket.
- Preserve explicit component/device/precision selections and dependency integrity checks. No automatic VAE substitutions or model conversions. Raw INT8/ConvRot/AIO compatibility is not claimed.

## Verified

- Python unit/contract regressions and JavaScript frontend regressions.
- Installed ComfyUI core schema/config/CLIP enum forwarding (weight I/O mocked for core contract checks).
- Actual local MOSS Base and SFT exported DiT components, Qwen3.5 text encoder and VAE validated by real dependency loaders, including manifests and weight integrity. Incompatible raw INT8 correctly rejected. This is component-descriptor loading, not inference.
- `comfy node validate`.

## Upgrade

Restart ComfyUI when convenient, hard-refresh the frontend, then reopen saved UI workflows. No running server or workflow is modified by the release itself. For new API prompts use `UniversalModelLoaderV2` and indices 0=model, 1=model_name, 2=clip, 3=vae, 4=loaded_info, 5=clip2, 6=vae2. Old API prompts may retain `UniversalModelLoader` unchanged.
