import { app } from "/scripts/app.js";

const NODE_NAME = "UniversalModelLoader";
const INITIAL_WIDTH = 580;
const MIN_HEIGHT = 90;

// Palette sampled from the provided screenshots.
const THEME = {
  header: "#5140d4",
  headerDot: "#9aa45d",
  body: "#29263d",
  widget: "#003f46",
  widgetBorder: "#87c7bc",
  widgetText: "#f7fff8",
  widgetLabel: "#8fa3a6",
  widgetArrow: "#fff7df",
  muted: "#6f8489",
  info: "#ffd21a",
  close: "#ff4a3d",
};

const BASE_GROUPS = {
  checkpoint: new Set(["model_type", "checkpoint_name", "clip2_name", "vae2_name"]),
  diffusers: new Set(["model_type", "diffusers_model_path", "clip2_name", "vae2_name"]),
  diffusion_model: new Set(["model_type", "diffusion_model_name", "weight_dtype", "clip_name", "clip_type", "clip_device", "vae_name", "clip2_name", "vae2_name"]),
  unet: new Set(["model_type", "diffusion_model_name", "weight_dtype", "clip_name", "clip_type", "clip_device", "vae_name", "clip2_name", "vae2_name"]),
  gguf_unet: new Set(["model_type", "gguf_unet_name", "gguf_dequant_dtype", "gguf_patch_dtype", "gguf_patch_on_device", "clip_name", "clip_type", "clip_device", "vae_name", "clip2_name", "vae2_name"]),
};

const MODEL_SELECTOR_BY_TYPE = {
  checkpoint: "checkpoint_name",
  diffusers: "diffusers_model_path",
  diffusion_model: "diffusion_model_name",
  unet: "diffusion_model_name",
  gguf_unet: "gguf_unet_name",
};

const MODEL_SELECTOR_WIDGETS = new Set(Object.values(MODEL_SELECTOR_BY_TYPE));

function selected(node, name, fallback) {
  return String(node.widgets?.find((w) => w.name === name)?.value ?? fallback);
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function normalizeOptionalSelector(node, name) {
  const w = widget(node, name);
  if (!w) return;
  const values = w.options?.values || w.options?.items || w.values;
  if (!Array.isArray(values) || !values.includes("none")) return;
  if (w.value === undefined || w.value === null || w.value === "" || w.value === "— none found —") {
    w.value = "none";
  }
}

function visibleWidgets(node) {
  const modelType = selected(node, "model_type", "checkpoint");
  const visible = new Set(BASE_GROUPS[modelType] || BASE_GROUPS.checkpoint);
  if (selected(node, "clip2_name", "none") !== "none") {
    visible.add("clip2_type");
    visible.add("clip2_device");
  }
  return visible;
}

function graphToScreen(node, x, y) {
  const canvas = app.canvas?.canvas;
  const rect = canvas?.getBoundingClientRect?.() || { left: 0, top: 0 };
  const ds = app.canvas?.ds;
  const scale = ds?.scale || 1;
  const offset = ds?.offset || [0, 0];
  return {
    x: rect.left + (node.pos[0] + x + offset[0]) * scale,
    y: rect.top + (node.pos[1] + y + offset[1]) * scale,
  };
}

function showUsageInfo(node) {
  const old = document.getElementById("uml-usage-popup");
  if (old) old.remove();

  const p = graphToScreen(node, node.size?.[0] || INITIAL_WIDTH, -32);
  const panel = document.createElement("div");
  panel.id = "uml-usage-popup";
  panel.innerHTML = `
    <button title="Close">×</button>
    <div><b>Universal MODEL Loader</b></div>
    <p>Select <code>model_type</code>; only matching fields are shown.</p>
    <p><b>Checkpoint/Diffusers</b>: load MODEL, CLIP and VAE from the selected package.</p>
    <p><b>UNet/GGUF</b>: load diffusion MODEL plus CLIP and VAE from this node.</p>
    <p><b>Extra outputs</b>: optional <code>clip2</code> and <code>vae2</code> can be set to <code>none</code>.</p>
    <p>Leave CLIP type on <code>auto</code> to resolve it at load time. Explicit CLIP types and VAE selections are preserved.</p>
  `;
  Object.assign(panel.style, {
    position: "fixed",
    left: `${Math.round(p.x + 12)}px`,
    top: `${Math.round(p.y)}px`,
    width: "370px",
    padding: "18px 22px 18px 14px",
    color: THEME.widgetText,
    background: THEME.widget,
    border: `3px solid ${THEME.widgetBorder}`,
    borderRadius: "8px",
    font: "13px monospace",
    lineHeight: "1.35",
    zIndex: 1000000,
    boxShadow: "0 10px 26px rgba(0,0,0,.55)",
  });
  const close = panel.querySelector("button");
  Object.assign(close.style, {
    position: "absolute",
    top: "4px",
    right: "6px",
    border: "0",
    background: "transparent",
    color: THEME.close,
    font: "bold 24px sans-serif",
    cursor: "pointer",
  });
  close.onclick = () => panel.remove();
  document.body.appendChild(panel);
}

function patchWidget(widget) {
  if (widget._umlPatched) return;
  widget._umlPatched = true;
  widget._umlComputeSize = widget.computeSize;
  widget.computeSize = function(width) {
    if (this._umlHidden) return [width, 0];
    if (this._umlComputeSize) return this._umlComputeSize.call(this, width);
    return [width, 20];
  };
}

function styleWidget(widget) {
  widget.options ??= {};
  widget.options.color = THEME.widget;
  widget.options.bg_color = THEME.widget;
  widget.options.text_color = widget._umlHidden ? THEME.muted : THEME.widgetText;
  widget.options.secondary_text_color = THEME.widgetLabel;
  widget.options.border_color = THEME.widgetBorder;
}

function setWidgetVisible(widget, show) {
  patchWidget(widget);
  widget._umlHidden = !show;
  widget.hidden = !show;
  widget.disabled = false;
  styleWidget(widget);
}

function patchModelSelectorSerialization(node, widget) {
  if (!MODEL_SELECTOR_WIDGETS.has(widget.name) || widget._umlSerializePatched) return;
  widget._umlSerializePatched = true;
  const originalSerializeValue = widget.serializeValue;
  widget.serializeValue = function(workflowNode, widgetIndex) {
    const modelType = selected(node, "model_type", "checkpoint");
    const activeSelector = MODEL_SELECTOR_BY_TYPE[modelType] || "checkpoint_name";
    const serialized = originalSerializeValue
      ? originalSerializeValue.call(this, workflowNode, widgetIndex)
      : this.value;

    if (this.name === activeSelector) return serialized;

    if (workflowNode?.widgets_values) workflowNode.widgets_values[widgetIndex] = null;
    return serialized;
  };
}

function applyTheme(node) {
  node.color = THEME.header;
  node.bgcolor = THEME.body;
  node.boxcolor = THEME.headerDot;
  node.title_text_color = "#ffffff";
}

function updateNode(node, forceInitialWidth = false) {
  normalizeOptionalSelector(node, "clip2_name");
  normalizeOptionalSelector(node, "vae2_name");
  const visible = visibleWidgets(node);
  const widthBefore = Array.isArray(node.size) ? node.size[0] : INITIAL_WIDTH;
  applyTheme(node);

  for (const w of node.widgets || []) {
    patchModelSelectorSerialization(node, w);
    setWidgetVisible(w, visible.has(w.name));
  }

  const computed = node.computeSize?.([forceInitialWidth ? INITIAL_WIDTH : widthBefore, 0]);
  const width = forceInitialWidth ? INITIAL_WIDTH : (widthBefore || INITIAL_WIDTH);
  const height = Math.max(computed?.[1] || MIN_HEIGHT, MIN_HEIGHT);
  if (Array.isArray(node.size)) {
    node.size[0] = width;
    node.size[1] = height;
  } else {
    node.size = [width, height];
  }
  app.graph?.setDirtyCanvas(true, true);
}

function drawInfoMark(node, ctx) {
  const x = (node.size?.[0] || INITIAL_WIDTH) - 28;
  // onDrawForeground is called with the body/slot coordinate origin in this
  // ComfyUI renderer, so use a negative y to place the mark in the title bar.
  const y = -14;
  ctx.save();
  ctx.fillStyle = THEME.info;
  ctx.font = "bold 16px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("?", x, y);
  ctx.restore();
  node._umlInfoRect = [x - 10, y - 11, 20, 22];
}

function patchCallback(node, widgetName) {
  const w = node.widgets?.find((item) => item.name === widgetName);
  if (!w || w._umlCallbackPatched) return;

  w._umlCallbackPatched = true;
  const originalCallback = w.callback;
  w.callback = function(value, canvas, nodeArg, pos, event) {
    const result = originalCallback?.call(this, value, canvas, nodeArg, pos, event);
    queueMicrotask(() => updateNode(node));
    return result;
  };
}

function patchLiveWidgets(node) {
  patchCallback(node, "model_type");
  patchCallback(node, "diffusion_model_name");
  patchCallback(node, "gguf_unet_name");
  patchCallback(node, "clip2_name");
}

app.registerExtension({
  name: "UniversalModelLoader.DynamicWidgets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      originalCreated?.apply(this, arguments);
      applyTheme(this);
      patchLiveWidgets(this);
      queueMicrotask(() => updateNode(this, true));
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function(info) {
      originalConfigure?.apply(this, arguments);
      patchLiveWidgets(this);
      queueMicrotask(() => updateNode(this));
    };

    const originalDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function(ctx) {
      originalDrawForeground?.apply(this, arguments);
      drawInfoMark(this, ctx);
    };

    const originalMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function(event, pos, canvas) {
      const r = this._umlInfoRect;
      if (r && pos?.[0] >= r[0] && pos[0] <= r[0] + r[2] && pos?.[1] >= r[1] && pos[1] <= r[1] + r[3]) {
        showUsageInfo(this);
        return true;
      }
      return originalMouseDown?.apply(this, arguments);
    };
  },
});
