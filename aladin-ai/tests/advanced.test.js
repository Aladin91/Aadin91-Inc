import assert from 'node:assert/strict';
import {
  SeededRandom, LabelEncoder, OneHotEncoder,
  LinearRegression, LogisticRegression, RandomForestClassifier,
  KMeansPlusPlus, ZScoreAnomalyDetector, IQRAnomalyDetector,
  Pipeline, crossValidate, SerializableModel, createDefaultRegistry
} from '../src/advanced.js';
import { StandardScaler, DecisionTreeClassifier } from '../src/aladin-ai.js';

const r1=new SeededRandom(7), r2=new SeededRandom(7);
assert.deepEqual([r1.next(),r1.next(),r1.next()],[r2.next(),r2.next(),r2.next()]);

const le=new LabelEncoder();
assert.deepEqual(le.fitTransform(['hvac','plumbing','hvac']),[0,1,0]);
assert.deepEqual(le.inverseTransform([1,0]),['plumbing','hvac']);

const oh=new OneHotEncoder();
assert.deepEqual(oh.fitTransform([['A'],['B'],['A']]),[[1,0],[0,1],[1,0]]);

const lr=new LinearRegression({learningRate:.05,epochs:2500}).fit([[0],[1],[2],[3]],[1,3,5,7]);
assert.ok(Math.abs(lr.predict([[4]])[0]-9)<0.2);
assert.ok(lr.score([[0],[1],[2],[3]],[1,3,5,7])>.99);

const logreg=new LogisticRegression({learningRate:.1,epochs:2500}).fit([[-2],[-1],[1],[2]],['no','no','yes','yes']);
assert.deepEqual(logreg.predict([[-1.5],[1.5]]),['no','yes']);
assert.ok(logreg.predictProba([[2]])[0].yes>.5);

const X=[[1,1],[1.2,1],[1.1,.8],[8.8,9],[9,9.2],[9.1,8.9]];
const y=['low','low','low','high','high','high'];
const rf=new RandomForestClassifier({trees:25,maxDepth:4,seed:12}).fit(X,y);
assert.deepEqual(rf.predict([[1,1],[9,9]]),['low','high']);
assert.equal(rf.featureImportances.length,2);
assert.ok(Math.abs(rf.featureImportances.reduce((a,b)=>a+b,0)-1)<1e-9);

const km=new KMeansPlusPlus({clusters:2,seed:9}).fit(X);
assert.equal(new Set(km.labels).size,2);
assert.ok(km.inertia>=0);

const z=new ZScoreAnomalyDetector({threshold:1.5}).fit([[0],[0.1],[-.1],[0.05]]);
assert.equal(z.predict([[5]])[0],-1);

const iqr=new IQRAnomalyDetector().fit([[1],[2],[2],[3],[2]]);
assert.equal(iqr.predict([[20]])[0],-1);

const pipe=new Pipeline([new StandardScaler(),new DecisionTreeClassifier({maxDepth:3})]).fit([[1],[2],[8],[9]],['a','a','b','b']);
assert.deepEqual(pipe.predict([[1.5],[8.5]]),['a','b']);

const cv=crossValidate(()=>new DecisionTreeClassifier({maxDepth:3}),[[1],[2],[3],[8],[9],[10]],['a','a','a','b','b','b'],{k:3,seed:2});
assert.equal(cv.length,3);
assert.ok(cv.every(v=>v>=0&&v<=1));

const registry=createDefaultRegistry();
const lrJson=SerializableModel.stringify(lr);
const restoredLr=SerializableModel.parse(lrJson,registry);
assert.ok(Math.abs(restoredLr.predict([[4]])[0]-lr.predict([[4]])[0])<1e-9);

const labelJson=SerializableModel.stringify(le);
const restoredLabel=SerializableModel.parse(labelJson,registry);
assert.deepEqual(restoredLabel.transform(['plumbing','hvac']),[1,0]);

const forestJson=SerializableModel.stringify(rf);
const restoredForest=SerializableModel.parse(forestJson,registry);
assert.deepEqual(restoredForest.predict([[1,1],[9,9]]),rf.predict([[1,1],[9,9]]));

const pipelineJson=SerializableModel.stringify(pipe);
const restoredPipeline=SerializableModel.parse(pipelineJson,registry);
assert.deepEqual(restoredPipeline.predict([[1.5],[8.5]]),['a','b']);

console.log('AladinAI advanced tests passed.');
