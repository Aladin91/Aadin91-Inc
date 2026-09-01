import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../dist/aladin-ai.js', import.meta.url), 'utf8');
const context = { console, setTimeout, clearTimeout, fetch: async()=>{throw new Error('network disabled in bundle test');} };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(source, context, {filename:'dist/aladin-ai.js'});

const AI = context.AladinAI;
assert.ok(AI, 'Browser bundle must expose global AladinAI');
assert.equal(AI.version, '0.4.0');
assert.equal(typeof AI.BM25Index, 'function');
assert.equal(typeof AI.RandomForestClassifier, 'function');
assert.equal(typeof AI.NeuralNetworkClassifier, 'function');
assert.equal(typeof AI.RandomForestRegressor, 'function');
assert.equal(typeof AI.sobelEdges, 'function');
assert.equal(typeof AI.perceptualSimilarity, 'function');
assert.equal(typeof AI.GitHubCloud, 'function');

const search = new AI.BM25Index().fit([
  'airflow fan duct',
  'voltage power disconnect',
  'water drain piping'
]);
assert.equal(search.search('power disconnect',{topK:1})[0].index, 1);

const forest = new AI.RandomForestClassifier({trees:5,maxDepth:3,seed:42}).fit([[1],[2],[8],[9]],['low','low','high','high']);
assert.equal(forest.predict([[8.5]])[0], 'high');

const img={width:2,height:2,data:Uint8Array.from([0,0,255,255])};
assert.equal(AI.averageHash(img).length,64);
assert.equal(AI.perceptualSimilarity(img,img),1);

const cloud=new AI.GitHubCloud({owner:'Aladin91',repo:'Aadin91-Inc',branch:'aladin-ai-engine'});
assert.ok(cloud.rawUrl('aladin-ai/cloud/manifest.json').includes('raw.githubusercontent.com'));

console.log('AladinAI generated dist bundle tests passed.');
