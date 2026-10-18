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
test('MOSS menu creates typed sibling without rewiring or substituting files', () => {
  const source = new Node();
  source.widgets.push({ name: 'clip_name', value: 'MI/moss-image-qwen35-bf16.safetensors' });
  source.widgets.find(w => w.name === 'diffusion_model_name').value = 'MI/mossImage_sft_pruned_int8.safetensors';
  const links = [[1, 4, 0, 9, 0, 'MODEL']];
  let added, message;
  source.graph = { links, add(node) { added = node; } };
  context.LiteGraph = { createNode(type) {
    assert.equal(type, 'UniversalMOSSLoader');
    return { widgets: ['diffusion_model_name', 'text_encoder_name', 'vae_name'].map(name => ({ name, value: 'DO NOT SUBSTITUTE' })) };
  } };
  context.alert = text => { message = text; };
  const options = []; source.getExtraMenuOptions(null, options);
  options.find(x => x.content === 'Add MOSS-Image loader (typed)').callback();
  assert.equal(added.widgets[0].value, 'MI/mossImage_sft_pruned_int8.safetensors');
  assert.equal(added.widgets[1].value, 'MI/moss-image-qwen35-bf16.safetensors');
  assert.equal(added.widgets[2].value, 'explicit_vae');
  assert.equal(source.graph.links, links);
  assert.match(message, /raw\/pruned INT8 is unsupported/);
  assert.match(message, /NOT converted or validated/);
  assert.equal(source.widgets.find(w => w.name === 'clip_type').value, 'auto');
});

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

test('legacy graph migration preserves every edge and is idempotent', () => {
  for (const count of [4, 6]) for (const objects of [false, true]) {
    const names = ['model', 'clip', 'vae', 'loaded_info', 'clip2', 'vae2'].slice(0, count);
    const graph = { nodes: [{ id: 7, type: 'UniversalModelLoader', outputs: names.map((name, i) => ({ name, slot_index: i, links: [100+i] })) }], links: names.map((_, i) => objects ? ({ id: 100+i, origin_id: 7, origin_slot: i, target_id: 20+i, target_slot: 0 }) : [100+i, 7, i, 20+i, 0, 'TYPE']) };
    const original = structuredClone(graph.links);
    extension.beforeConfigureGraph(graph);
    assert.equal(graph.nodes[0].type, 'UniversalModelLoaderV2');
    assert.equal(graph.nodes[0].outputs[1].name, 'model_name');
    assert.equal(graph.nodes[0].outputs.length, 7);
    graph.links.forEach((link, i) => {
      assert.equal(objects ? link.origin_slot : link[2], i === 0 ? 0 : i+1);
      assert.equal(objects ? link.target_id : link[3], objects ? original[i].target_id : original[i][3]);
      assert.deepEqual(graph.nodes[0].outputs[i === 0 ? 0 : i+1].links, [100+i]);
    });
    const once = JSON.stringify(graph); extension.beforeConfigureGraph(graph);
    assert.equal(JSON.stringify(graph), once);
  }
});

test('unknown schemas and API prompts remain untouched; nested graphs migrate', () => {
  const api = { '7': { class_type: 'UniversalModelLoader', inputs: {} }, '8': { inputs: { clip: ['7', 1] } } };
  const original = JSON.stringify(api); extension.beforeConfigureGraph(api); assert.equal(JSON.stringify(api), original);
  const graph = { nodes: [{ id: 1, type: 'UniversalModelLoader', outputs: [{name: 'unknown'}] }], links: [] };
  const originalGraph = JSON.stringify(graph); extension.beforeConfigureGraph(graph); assert.equal(JSON.stringify(graph), originalGraph);
  graph.nodes[0].outputs = ['model', 'clip', 'vae', 'loaded_info'].map(name => ({name}));
  graph.links = [[1, 1, 99, 2, 0, 'CLIP']];
  const malformed = JSON.stringify(graph); extension.beforeConfigureGraph(graph);
  assert.equal(JSON.stringify(graph), malformed);
  graph.links = [];
  extension.beforeConfigureGraph({ definitions: { subgraphs: [graph] } });
  assert.equal(graph.nodes[0].type, 'UniversalModelLoaderV2');
});

test('secondary visibility tracks optional encoder', async () => {
  const node = new Node(); node.onNodeCreated(); await tick();
  assert.equal(w(node, 'clip2_type').hidden, true);
  w(node, 'clip2_name').value = 'encoder'; w(node, 'clip2_name').callback('encoder'); await tick();
  assert.equal(w(node, 'clip2_type').hidden, false);
  assert.equal(w(node, 'vae2_name').hidden, false);
});
