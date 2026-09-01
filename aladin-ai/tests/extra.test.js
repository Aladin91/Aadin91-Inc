import assert from 'node:assert/strict';
import { PolynomialFeatures, PCA, SoftmaxRegression, HierarchicalClustering, RobustScaler, ExponentialMovingAverage } from '../src/extra.js';

const poly=new PolynomialFeatures({degree:2}).fit([[2,3],[4,5]]);
assert.deepEqual(poly.transform([[2,3]])[0],[2,3,4,6,9]);
assert.deepEqual(poly.featureNames(['x','y']),['x','y','x*x','x*y','y*y']);

const pcaData=[[2,0],[1,0],[-1,0],[-2,0]];
const pca=new PCA({components:1,seed:2}).fit(pcaData);
const projected=pca.transform(pcaData);
assert.equal(projected[0].length,1);
assert.ok(pca.explainedVarianceRatio[0]>.99);

const X=[[-3,0],[-2,0],[-1,0],[0,3],[0,2],[0,1],[3,0],[2,0],[1,0]];
const y=['left','left','left','up','up','up','right','right','right'];
const softmax=new SoftmaxRegression({learningRate:.1,epochs:3000}).fit(X,y);
assert.deepEqual(softmax.predict([[-2,0],[0,2],[2,0]]),['left','up','right']);
const prob=softmax.predictProba([[0,2]])[0];
assert.ok(prob.up>prob.left&&prob.up>prob.right);

const hc=new HierarchicalClustering({clusters:2,linkage:'average'}).fit([[0,0],[.1,0],[10,10],[10.1,10]]);
assert.equal(new Set(hc.labels).size,2);
assert.equal(hc.dendrogram.length,2);

const rs=new RobustScaler().fit([[1],[2],[3],[100]]);
const scaled=rs.transform([[2]])[0][0];
assert.ok(Number.isFinite(scaled));

const ema=new ExponentialMovingAverage({alpha:.5});
assert.deepEqual(ema.fit([10,20,30]),[10,15,22.5]);

console.log('AladinAI extra tests passed.');
