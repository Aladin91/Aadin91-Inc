import assert from 'node:assert/strict';
import { CountVectorizer, FeatureHasher, BM25Index, TextSimilarityIndex, NearestNeighborsIndex, reciprocalRankFusion } from '../src/retrieval.js';

const docs = [
  'air handling unit supply fan airflow',
  'electrical disconnect voltage power wiring',
  'condensate drain water piping plumbing'
];

const cv = new CountVectorizer({ngramRange:[1,2],maxFeatures:20});
const X = cv.fitTransform(docs);
assert.equal(X.length, 3);
assert.ok(cv.getFeatureNames().includes('air'));
assert.ok(cv.getFeatureNames().some(x => x.includes('air handling')));

const hasher = new FeatureHasher({features:32});
const H = hasher.transform(docs);
assert.equal(H[0].length, 32);

const bm25 = new BM25Index().fit(docs, [{id:'hvac'},{id:'electrical'},{id:'plumbing'}]);
assert.equal(bm25.search('power voltage', {topK:1})[0].metadata.id, 'electrical');

const sim = new TextSimilarityIndex().fit(docs, [{id:'hvac'},{id:'electrical'},{id:'plumbing'}]);
assert.equal(sim.search('supply airflow fan', {topK:1})[0].metadata.id, 'hvac');

const nn = new NearestNeighborsIndex({metric:'euclidean'}).fit([[0,0],[5,5],[10,10]], [{id:'a'},{id:'b'},{id:'c'}]);
assert.equal(nn.search([5.2,4.9], {topK:1})[0].metadata.id, 'b');

const fused = reciprocalRankFusion([
  [{index:0,score:2},{index:1,score:1}],
  [{index:1,score:.9},{index:0,score:.8}]
], {topK:2});
assert.equal(fused.length, 2);
assert.ok(fused[0].fusionScore > 0);

console.log('AladinAI retrieval tests passed.');
