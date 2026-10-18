# Universal Model Loader 1.3.1

## Fixed MOSS routing and guidance

- Recognized MOSS model/encoder/VAE filenames now stop before core weight loading with a precise explanation and typed-loader routing instructions. Inactive hidden selections are ignored.
- Existing Universal MODEL Loader nodes now offer **Add MOSS-Image loader (typed)** in their right-click menu. It adds the typed adapter beside the original and copies filenames verbatim; existing nodes, links and output contracts stay unchanged.
- No invented `moss` CLIP enum, wildcard sockets, implicit model conversion or VAE substitution.

The reported `mossImage_sft_pruned_int8.safetensors` failure occurred in core UNETLoader before CLIP loading. That raw/pruned INT8 file remains unsupported by the installed MOSS integration. Select independently exported MOSS Base/SFT, Qwen3.5 text encoder and VAE components explicitly and connect to moss-image-t8 Official Text Encode / Sampler / VAE Decode, not standard KSampler.

## Validation

- Python and frontend regression suites, real installed core schema/config checks, and Comfy Registry metadata validation.
- Real MOSS Base/SFT component metadata, manifests and weight-integrity checks; explicit unsupported INT8 rejection and universal-node early routing check.
- These are loader/descriptor checks, not end-to-end image generation. No inference claim.

Restart ComfyUI and refresh the browser after installing the update.
