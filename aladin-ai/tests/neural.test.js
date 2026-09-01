import assert from 'node:assert/strict';
import { NeuralNetworkClassifier, NeuralNetworkRegressor } from '../src/neural.js';

const X=[[-2],[-1.5],[-1],[1],[1.5],[2]];
const y=['low','low','low','high','high','high'];
const clf=new NeuralNetworkClassifier({hiddenLayers:[6],activation:'tanh',learningRate:.08,epochs:1200,seed:4}).fit(X,y);
assert.deepEqual(clf.predict([[-1.8],[1.8]]),['low','high']);
const cp=clf.predictProba([[1.8]])[0];
assert.ok(cp.high>.7);
assert.ok(clf.lossHistory.at(-1).loss<clf.lossHistory[0].loss);

const reg=new NeuralNetworkRegressor({hiddenLayers:[6],activation:'tanh',learningRate:.02,epochs:1800,seed:3}).fit([[-1],[0],[1],[2]],[-1,1,3,5]);
const pred=reg.predict([[1.5]])[0];
assert.ok(Math.abs(pred-4)<.75);
assert.ok(reg.lossHistory.at(-1).loss<reg.lossHistory[0].loss);

console.log('AladinAI neural tests passed.');
