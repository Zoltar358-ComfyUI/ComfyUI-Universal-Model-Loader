# Universal Model Loader 1.3.2

- Fix native MOSS-Image loading on current ComfyUI: allow the standard MODEL/CLIP/VAE route when core exposes `moss_image` / `CLIPType.MOSS_IMAGE`.
- Resolve automatic CLIP selection from MOSS-specific filenames, MOSS-specific Qwen encoder names, and the native `MOSSImage` model config. Generic Qwen3.5 filenames alone remain ambiguous.
- Preserve explicit CLIP types, VAE selections, model_name output, socket order, and independent secondary outputs. Core remains responsible for validating weight formats; no downloads, conversions, or substitutions are performed.
- Keep actionable update-or-typed-loader errors on older cores and retain the optional typed MOSS integration.
- Add native MOSS routing regression coverage, real-core enum forwarding coverage, and an opt-in CPU weight-loading smoke test.

Validation: 19 Python unit tests, 7 frontend tests, and real-core contracts across 38 supported model config classes. Real CPU MOSS MODEL/CLIP/VAE loading and text encoding were verified during fix validation. End-to-end sampling/image inference was not verified (sampling failed/timed out); this release makes no image-inference claim.
