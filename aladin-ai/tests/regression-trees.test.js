import assert from 'node:assert/strict';
import { DecisionTreeRegressor, RandomForestRegressor } from '../src/regression-trees.js';

const X=[[1],[2],[3],[8],[9],[10]];
const y=[2,4,6,16,18,20];

const tree=new DecisionTreeRegressor({maxDepth:5}).fit(X,y);
const tp=tree.predict([[9]])[0];
assert.ok(Math.abs(tp-18)<3);
assert.ok(tree.score(X,y)>.9);
assert.ok(tree.explain([9],['x']).path.length>0);

const forest=new RandomForestRegressor({trees:25,maxDepth:5,maxFeatures:1,seed:42}).fit(X,y);
const fp=forest.predict([[9]])[0];
assert.ok(fp>12&&fp<22);
assert.ok(forest.score(X,y)>.7);
assert.equal(forest.explain(['x'])[0].feature,'x');

console.log('AladinAI regression tree tests passed.');
