import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../dist/aladin-ai.js', import.meta.url), 'utf8');
const context = { console, setTimeout, clearTimeout };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(source, context, {filename:'dist/aladin-ai.js'});

const AI = context.AladinAI;
assert.ok(AI, 'Browser bundle must expose global AladinAI');
assert.equal(AI.version, '0.3.0');
assert.equal(typeof AI.BM25Index, 'function');
assert.equal(typeof AI.RandomForestClassifier, 'function');
assert.equal(typeof AI.NeuralNetworkClassifier, 'function');

const search = new AI.BM25Index().fit([
  'airflow fan duct',
  'voltage power disconnect',
  'water drain piping'
]);
assert.equal(search.search('power disconnect',{topK:1})[0].index, 1);

const forest = new AI.RandomForestClassifier({trees:5,maxDepth:3,seed:42}).fit([[1],[2],[8],[9]],['low','low','high','high']);
assert.equal(forest.predict([[8.5]])[0], 'high');

console.log('AladinAI generated dist bundle tests passed.');
