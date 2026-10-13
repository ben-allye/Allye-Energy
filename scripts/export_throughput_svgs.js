/* Run the existing HTML deck's chart JavaScript and save its SVG output. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'notebooks/throughput_summary_deck.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
if (!code) throw new Error('Deck chart script was not found');

class Element {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.style = {}; this._text = ''; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener() {}
  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text; }
}

const containers = new Map();
const document = {
  documentElement: new Element('html'),
  createElement: tag => new Element(tag),
  createElementNS: (_namespace, tag) => new Element(tag),
  getElementById: id => {
    if (!containers.has(id)) containers.set(id, new Element('div'));
    return containers.get(id);
  },
};
const css = { '--seq-lo': '#F3F1FA', '--seq-hi': '#514898' };
vm.runInNewContext(code, {
  document,
  window: {innerWidth: 1920, innerHeight: 1080},
  getComputedStyle: () => ({getPropertyValue: name => css[name] || ''}),
}, {filename: 'throughput_summary_deck.html'});

const escape = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
function serialize(node) {
  const attrs = Object.entries(node.attrs).map(([key, value]) => ` ${key}="${escape(value)}"`).join('');
  const content = escape(node._text) + node.children.map(serialize).join('');
  return `<${node.tag}${attrs}>${content}</${node.tag}>`;
}
const output = path.join(root, 'reports', 'throughput_svgs');
fs.mkdirSync(output, {recursive: true});
const charts = {
  monthly: 'chart-monthly',
  per_live: 'chart-per-live',
  weekday: 'chart-weekday',
  tou_hourly: 'chart-tou-hourly',
  tou_heatmap: 'chart-tou-heatmap',
  efc_line: 'chart-efc-line',
  efficiency: 'chart-efficiency',
  charge_time: 'chart-charge-time',
  boxplot: 'chart-boxplot',
};
for (const [name, id] of Object.entries(charts)) {
  const svg = containers.get(id)?.children.find(x => x.tag === 'svg');
  if (!svg) throw new Error(`${id} SVG was not generated`);
  fs.writeFileSync(path.join(output, `${name}.raw.svg`), serialize(svg));
}
const hist = containers.get('chart-hist')?.children[0]?.children.map(card => card.children[0]);
if (!hist || hist.length !== 4 || hist.some(x => x.tag !== 'svg')) throw new Error('Histogram SVGs were not generated');
hist.forEach((svg, i) => fs.writeFileSync(path.join(output, `histogram_${i+1}.raw.svg`), serialize(svg)));
console.log(`Exported ${Object.keys(charts).length + hist.length} SVG charts to ${output}`);
