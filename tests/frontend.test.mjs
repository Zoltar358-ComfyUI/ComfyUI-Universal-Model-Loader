import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const sourcePath = process.env.UML_FRONTEND || new URL('../web/universal_model_loader.js', import.meta.url);
const source = fs.readFileSync(sourcePath, 'utf8').replace(/^import .*;\n/, '');
let extension;
const context = vm.createContext({ app: { registerExtension(e) { extension = e; }, graph: { setDirtyCanvas() {} } }, queueMicrotask });
vm.runInContext(source, context);
class Node {
  constructor() {
    const values = { model_type: 'diffusion_model', checkpoint_name: 'hidden.ckpt', diffusion_model_name: 'KR2/model', diffusers_model_path: 'hidden_folder', gguf_unet_name: 'hidden.gguf', clip_type: 'auto', clip2_type: 'auto', clip2_name: 'none', vae2_name: 'none', vae_name: 'explicit_vae' };
    this.widgets = Object.entries(values).map(([name, value]) => ({ name, value, options: { values: ['auto', 'stable_diffusion', 'krea2', 'qwen_image', 'none'] } }));
    this.size = [780, 400];
    this.pos = [0, 0];
  }
  computeSize() { return [200, 300]; }
}
await extension.beforeRegisterNodeDef(Node, { name: 'UniversalModelLoader' });
const tick = () => new Promise(resolve => queueMicrotask(resolve));
const w = (node, name) => node.widgets.find(w => w.name === name);

test('auto remains auto after creation, model switches and configure', async () => {
  const node = new Node(); node.onNodeCreated(); await tick();
  assert.equal(w(node, 'clip_type').value, 'auto');
  for (const hint of ['QWN2/model', 'wan_model', 'unknown']) {
    w(node, 'diffusion_model_name').value = hint;
    w(node, 'diffusion_model_name').callback(hint);
    await tick();
    assert.equal(w(node, 'clip_type').value, 'auto');
    assert.equal(w(node, 'clip2_type').value, 'auto');
    assert.equal(w(node, 'vae_name').value, 'explicit_vae');
  }
  node.size[0] = 840; node.onConfigure({}); await tick();
  assert.equal(node.size[0], 840);
});

test('explicit types survive configuration and model changes', async () => {
  const node = new Node();
  w(node, 'clip_type').value = 'stable_diffusion';
  w(node, 'clip2_type').value = 'qwen_image';
  node.onConfigure({}); await tick();
  assert.equal(w(node, 'clip_type').value, 'stable_diffusion');
  assert.equal(w(node, 'clip2_type').value, 'qwen_image');
});

test('inactive metadata is null but execution serializer remains valid', async () => {
  const node = new Node(); node.onConfigure({}); await tick();
  const metadata = { widgets_values: node.widgets.map(w => w.value) };
  for (const name of ['checkpoint_name', 'diffusion_model_name', 'diffusers_model_path', 'gguf_unet_name']) {
    const item = w(node, name), index = node.widgets.indexOf(item);
    assert.equal(await item.serializeValue(metadata, index), item.value);
    assert.equal(metadata.widgets_values[index], name === 'diffusion_model_name' ? item.value : null);
  }
});

test('secondary visibility tracks optional encoder', async () => {
  const node = new Node(); node.onNodeCreated(); await tick();
  assert.equal(w(node, 'clip2_type').hidden, true);
  w(node, 'clip2_name').value = 'encoder'; w(node, 'clip2_name').callback('encoder'); await tick();
  assert.equal(w(node, 'clip2_type').hidden, false);
  assert.equal(w(node, 'vae2_name').hidden, false);
});
