import assert from 'node:assert/strict';
import { DecisionTreeClassifier } from '../src/aladin-ai.js';
import { ImageFeatureExtractor, ImageSimilarityIndex, ImageClassifier } from '../src/vision-ml.js';

const dark={width:4,height:4,data:Uint8Array.from(Array(16).fill(10))};
const light={width:4,height:4,data:Uint8Array.from(Array(16).fill(240))};
const mixed={width:4,height:4,data:Uint8Array.from([10,10,10,10,10,10,10,10,240,240,240,240,240,240,240,240])};
const extractor=new ImageFeatureExtractor({histogramBins:8,edgeBins:4});
const X=extractor.transform([dark,light,mixed]);assert.equal(X.length,3);assert.equal(X[0].length,15);
const index=new ImageSimilarityIndex({histogramBins:8,edgeBins:4}).fit([dark,light,mixed],['dark','light','mixed']);
assert.equal(index.search(light,{topK:1})[0].metadata,'light');
const clf=new ImageClassifier(new DecisionTreeClassifier({maxDepth:3}),{descriptorOptions:{histogramBins:8,edgeBins:4}}).fit([dark,light,mixed],['dark','light','mixed']);
assert.equal(clf.predict([dark])[0],'dark');
console.log('AladinAI image ML adapter tests passed.');
