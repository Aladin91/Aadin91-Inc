import { MathX, DecisionTreeClassifier, StandardScaler } from './aladin-ai.js';

export const ALADIN_AI_ADVANCED_VERSION = '0.2.0';

export class SeededRandom {
  constructor(seed = 42) { this.state = seed >>> 0; }
  next() { this.state = (1664525 * this.state + 1013904223) >>> 0; return this.state / 4294967296; }
  int(max) { return Math.floor(this.next() * max); }
  shuffle(values) { const a = [...values]; for (let i=a.length-1;i>0;i--){ const j=this.int(i+1); [a[i],a[j]]=[a[j],a[i]]; } return a; }
}

export const Validation = {
  matrix(X, name='X') {
    if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0])) throw new Error(`${name} must be a non-empty 2D array`);
    const cols=X[0].length;
    if (!cols) throw new Error(`${name} must contain at least one feature`);
    X.forEach((r,i)=>{ if(!Array.isArray(r)||r.length!==cols) throw new Error(`${name}[${i}] has inconsistent feature count`); r.forEach((v,j)=>{ if(typeof v!=='number'||!Number.isFinite(v)) throw new Error(`${name}[${i}][${j}] must be a finite number`); }); });
    return X;
  },
  supervised(X,y){ this.matrix(X); if(!Array.isArray(y)||y.length!==X.length) throw new Error('y length must match X rows'); if(!y.length) throw new Error('y cannot be empty'); return true; }
};

export class LabelEncoder {
  fit(values){ this.classes=[...new Set(values)]; this.index=new Map(this.classes.map((v,i)=>[v,i])); return this; }
  transform(values){ if(!this.index) throw new Error('LabelEncoder is not fitted'); return values.map(v=>{ if(!this.index.has(v)) throw new Error(`Unknown label: ${v}`); return this.index.get(v); }); }
  inverseTransform(values){ if(!this.classes) throw new Error('LabelEncoder is not fitted'); return values.map(i=>this.classes[i]); }
  fitTransform(values){ return this.fit(values).transform(values); }
}

export class OneHotEncoder {
  fit(rows){ if(!Array.isArray(rows)||!rows.length) throw new Error('rows required'); const data=Array.isArray(rows[0])?rows:rows.map(v=>[v]); const cols=data[0].length; this.categories=[]; for(let j=0;j<cols;j++) this.categories.push([...new Set(data.map(r=>r[j]))]); return this; }
  transform(rows){ if(!this.categories) throw new Error('OneHotEncoder is not fitted'); const data=Array.isArray(rows[0])?rows:rows.map(v=>[v]); return data.map(r=>this.categories.flatMap((cats,j)=>cats.map(c=>r[j]===c?1:0))); }
  fitTransform(rows){ return this.fit(rows).transform(rows); }
}

export class LinearRegression {
  constructor({learningRate=0.01,epochs=2000,l2=0}={}){ this.learningRate=learningRate; this.epochs=epochs; this.l2=l2; }
  fit(X,y){ Validation.supervised(X,y); const n=X.length,p=X[0].length; this.weights=Array(p).fill(0); this.bias=0; this.lossHistory=[]; for(let e=0;e<this.epochs;e++){ const pred=X.map(r=>this.bias+r.reduce((s,v,j)=>s+v*this.weights[j],0)); const err=pred.map((v,i)=>v-y[i]); const db=2*err.reduce((a,b)=>a+b,0)/n; const dw=this.weights.map((w,j)=>2*err.reduce((s,er,i)=>s+er*X[i][j],0)/n+2*this.l2*w); this.bias-=this.learningRate*db; this.weights=this.weights.map((w,j)=>w-this.learningRate*dw[j]); if(e%100===0) this.lossHistory.push(err.reduce((s,v)=>s+v*v,0)/n); } return this; }
  predict(X){ Validation.matrix(X); return X.map(r=>this.bias+r.reduce((s,v,j)=>s+v*this.weights[j],0)); }
  score(X,y){ const p=this.predict(X),m=MathX.mean(y),ssRes=y.reduce((s,v,i)=>s+(v-p[i])**2,0),ssTot=y.reduce((s,v)=>s+(v-m)**2,0); return 1-ssRes/(ssTot||1); }
}

export class LogisticRegression {
  constructor({learningRate=0.05,epochs=1500,l2=0}={}){ this.learningRate=learningRate; this.epochs=epochs; this.l2=l2; }
  _sigmoid(z){ return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z)); }
  fit(X,y){ Validation.supervised(X,y); this.classes=[...new Set(y)]; if(this.classes.length!==2) throw new Error('LogisticRegression currently supports binary classification'); const yy=y.map(v=>v===this.classes[1]?1:0),n=X.length,p=X[0].length; this.weights=Array(p).fill(0); this.bias=0; for(let e=0;e<this.epochs;e++){ const probs=X.map(r=>this._sigmoid(this.bias+r.reduce((s,v,j)=>s+v*this.weights[j],0))); const err=probs.map((v,i)=>v-yy[i]); const db=err.reduce((a,b)=>a+b,0)/n; const dw=this.weights.map((w,j)=>err.reduce((s,er,i)=>s+er*X[i][j],0)/n+this.l2*w); this.bias-=this.learningRate*db; this.weights=this.weights.map((w,j)=>w-this.learningRate*dw[j]); } return this; }
  predictProba(X){ Validation.matrix(X); return X.map(r=>{ const p=this._sigmoid(this.bias+r.reduce((s,v,j)=>s+v*this.weights[j],0)); return {[this.classes[0]]:1-p,[this.classes[1]]:p}; }); }
  predict(X,threshold=.5){ return this.predictProba(X).map(o=>o[this.classes[1]]>=threshold?this.classes[1]:this.classes[0]); }
}

export class RandomForestClassifier {
  constructor({trees=50,maxDepth=8,minSamplesSplit=2,maxFeatures='sqrt',sampleRate=1,seed=42}={}){ Object.assign(this,{trees,maxDepth,minSamplesSplit,maxFeatures,sampleRate,seed}); }
  _featureCount(p){ if(this.maxFeatures==='sqrt') return Math.max(1,Math.floor(Math.sqrt(p))); if(this.maxFeatures==='log2') return Math.max(1,Math.floor(Math.log2(p))); if(typeof this.maxFeatures==='number'&&this.maxFeatures>0&&this.maxFeatures<=1) return Math.max(1,Math.floor(p*this.maxFeatures)); if(Number.isInteger(this.maxFeatures)) return Math.min(p,this.maxFeatures); return p; }
  fit(X,y){ Validation.supervised(X,y); const rng=new SeededRandom(this.seed),p=X[0].length,m=this._featureCount(p),n=Math.max(2,Math.round(X.length*this.sampleRate)); this.models=[]; this.featureUsage=Array(p).fill(0); for(let t=0;t<this.trees;t++){ const features=rng.shuffle([...Array(p).keys()]).slice(0,m); const sx=[],sy=[]; for(let i=0;i<n;i++){ const idx=rng.int(X.length); sx.push(features.map(f=>X[idx][f])); sy.push(y[idx]); } const tree=new DecisionTreeClassifier({maxDepth:this.maxDepth,minSamplesSplit:this.minSamplesSplit}).fit(sx,sy); this.models.push({tree,features}); const walk=node=>{ if(!node||node.leaf)return; this.featureUsage[features[node.feature]]++; walk(node.left);walk(node.right); }; walk(tree.root); } const total=this.featureUsage.reduce((a,b)=>a+b,0)||1; this.featureImportances=this.featureUsage.map(v=>v/total); return this; }
  predict(X){ Validation.matrix(X); const all=this.models.map(({tree,features})=>tree.predict(X.map(r=>features.map(f=>r[f])))); return X.map((_,i)=>{ const counts={}; all.forEach(p=>counts[p[i]]=(counts[p[i]]||0)+1); return Object.entries(counts).sort((a,b)=>b[1]-a[1])[0][0]; }); }
  predictProba(X){ Validation.matrix(X); const all=this.models.map(({tree,features})=>tree.predict(X.map(r=>features.map(f=>r[f])))); return X.map((_,i)=>{ const c={}; all.forEach(p=>c[p[i]]=(c[p[i]]||0)+1); Object.keys(c).forEach(k=>c[k]/=this.models.length); return c; }); }
  explain(featureNames=[]){ return this.featureImportances.map((importance,i)=>({feature:featureNames[i]??i,importance})).sort((a,b)=>b.importance-a.importance); }
}

export class KMeansPlusPlus {
  constructor({clusters=3,maxIterations=100,seed=42,tolerance=1e-8}={}){ this.k=clusters; this.maxIterations=maxIterations; this.seed=seed; this.tolerance=tolerance; }
  fit(X){ Validation.matrix(X); if(X.length<this.k) throw new Error('clusters cannot exceed number of rows'); const rng=new SeededRandom(this.seed); this.centroids=[[...X[rng.int(X.length)]]]; while(this.centroids.length<this.k){ const d2=X.map(r=>Math.min(...this.centroids.map(c=>MathX.euclidean(r,c)**2))); const total=d2.reduce((a,b)=>a+b,0); if(total===0){ const unused=X.find(r=>!this.centroids.some(c=>MathX.euclidean(c,r)===0)); this.centroids.push([...(unused||X[rng.int(X.length)])]); continue; } let target=rng.next()*total,idx=0; for(;idx<d2.length-1;idx++){ target-=d2[idx]; if(target<=0) break; } this.centroids.push([...X[idx]]); }
    let labels=[]; for(let it=0;it<this.maxIterations;it++){ labels=X.map(r=>this.predictOne(r)); const next=this.centroids.map((c,k)=>{ const rows=X.filter((_,i)=>labels[i]===k); return rows.length?c.map((_,j)=>MathX.mean(rows.map(r=>r[j]))):c; }); const move=next.reduce((s,c,i)=>s+MathX.euclidean(c,this.centroids[i]),0); this.centroids=next; this.iterations=it+1; if(move<this.tolerance) break; } this.labels=labels; this.inertia=X.reduce((s,r,i)=>s+MathX.euclidean(r,this.centroids[this.labels[i]])**2,0); return this; }
  predictOne(r){ let best=0,d=Infinity; this.centroids.forEach((c,i)=>{ const x=MathX.euclidean(r,c); if(x<d){d=x;best=i;} }); return best; }
  predict(X){ Validation.matrix(X); return X.map(r=>this.predictOne(r)); }
}

export class ZScoreAnomalyDetector {
  constructor({threshold=3}={}){ this.threshold=threshold; }
  fit(X){ Validation.matrix(X); const p=X[0].length; this.means=[];this.stds=[]; for(let j=0;j<p;j++){const col=X.map(r=>r[j]);this.means.push(MathX.mean(col));this.stds.push(MathX.std(col)||1);} return this; }
  scoreSamples(X){ return X.map(r=>Math.max(...r.map((v,j)=>Math.abs((v-this.means[j])/this.stds[j])))); }
  predict(X){ return this.scoreSamples(X).map(s=>s>this.threshold?-1:1); }
}

export class IQRAnomalyDetector {
  constructor({factor=1.5}={}){ this.factor=factor; }
  _q(a,q){ const s=[...a].sort((x,y)=>x-y),pos=(s.length-1)*q,lo=Math.floor(pos),hi=Math.ceil(pos); return lo===hi?s[lo]:s[lo]+(s[hi]-s[lo])*(pos-lo); }
  fit(X){ Validation.matrix(X); const p=X[0].length; this.bounds=[]; for(let j=0;j<p;j++){const c=X.map(r=>r[j]),q1=this._q(c,.25),q3=this._q(c,.75),iqr=q3-q1;this.bounds.push([q1-this.factor*iqr,q3+this.factor*iqr]);}return this; }
  predict(X){ return X.map(r=>r.some((v,j)=>v<this.bounds[j][0]||v>this.bounds[j][1])?-1:1); }
}

export class Pipeline {
  constructor(steps=[]){ if(!steps.length) throw new Error('Pipeline requires at least one step'); this.steps=steps; }
  fit(X,y){ let data=X; for(let i=0;i<this.steps.length-1;i++){ const s=this.steps[i]; data=typeof s.fitTransform==='function'?s.fitTransform(data):s.fit(data,y).transform(data); } this.steps[this.steps.length-1].fit(data,y); return this; }
  predict(X){ let data=X; for(let i=0;i<this.steps.length-1;i++) data=this.steps[i].transform(data); return this.steps[this.steps.length-1].predict(data); }
  transform(X){ let data=X; for(const s of this.steps){ if(typeof s.transform!=='function') throw new Error('Every pipeline step must support transform for Pipeline.transform'); data=s.transform(data); } return data; }
}

export function kFoldIndices(n,k=5,seed=42){ if(k<2||k>n) throw new Error('k must be between 2 and n'); const idx=new SeededRandom(seed).shuffle([...Array(n).keys()]),folds=Array.from({length:k},()=>[]); idx.forEach((v,i)=>folds[i%k].push(v)); return folds.map(test=>({test,train:idx.filter(i=>!test.includes(i))})); }
export function crossValidate(modelFactory,X,y,{k=5,seed=42,metric=(a,b)=>a.filter((v,i)=>v===b[i]).length/a.length}={}){ Validation.supervised(X,y); return kFoldIndices(X.length,k,seed).map(({train,test})=>{ const m=modelFactory(); m.fit(train.map(i=>X[i]),train.map(i=>y[i])); return metric(test.map(i=>y[i]),m.predict(test.map(i=>X[i]))); }); }

export class DecisionEngine {
  constructor({models=[],ruleEngine=null,weights=null}={}){ this.models=models;this.ruleEngine=ruleEngine;this.weights=weights||models.map(()=>1); }
  analyze(row,context={}){ const votes=this.models.map((m,i)=>({prediction:m.predict([row])[0],weight:this.weights[i],model:m.constructor.name})); const totals={}; votes.forEach(v=>totals[v.prediction]=(totals[v.prediction]||0)+v.weight); const ranked=Object.entries(totals).sort((a,b)=>b[1]-a[1]),sum=this.weights.reduce((a,b)=>a+b,0)||1; const rules=this.ruleEngine?this.ruleEngine.evaluate(context):[]; return {decision:ranked[0]?.[0]??null,confidence:(ranked[0]?.[1]??0)/sum,votes,rules,reasons:[...votes.map(v=>`${v.model}: ${v.prediction}`),...rules.map(r=>r.reason)]}; }
}

export class SerializableModel {
  static dump(model){ const state={}; for(const [k,v] of Object.entries(model)) if(typeof v!=='function') state[k]=v; return {format:'aladin-ai-model',version:1,type:model.constructor.name,state}; }
  static restore(payload,registry={}){ if(!payload||payload.format!=='aladin-ai-model') throw new Error('Invalid AladinAI model payload'); const C=registry[payload.type]; if(!C) throw new Error(`Model type not registered: ${payload.type}`); const m=Object.create(C.prototype); Object.assign(m,payload.state); return m; }
}

export function createDefaultRegistry(extra={}){ return {LinearRegression,LogisticRegression,RandomForestClassifier,KMeansPlusPlus,ZScoreAnomalyDetector,IQRAnomalyDetector,LabelEncoder,OneHotEncoder,StandardScaler,...extra}; }
