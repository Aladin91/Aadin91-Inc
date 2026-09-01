import { MathX } from './aladin-ai.js';
import { Validation, SeededRandom } from './advanced.js';

export class PolynomialFeatures {
  constructor({degree=2,includeBias=false,interactionOnly=false}={}){ if(!Number.isInteger(degree)||degree<1)throw new Error('degree must be a positive integer');Object.assign(this,{degree,includeBias,interactionOnly}); }
  _combos(p){ const out=[]; const walk=(start,left,path)=>{ if(left===0){out.push([...path]);return;} for(let i=start;i<p;i++)walk(this.interactionOnly?i+1:i,left-1,[...path,i]); }; for(let d=1;d<=this.degree;d++)walk(0,d,[]); return out; }
  fit(X){ Validation.matrix(X); this.nFeatures=X[0].length; this.combinations=this._combos(this.nFeatures); return this; }
  transform(X){ Validation.matrix(X); if(!this.combinations)throw new Error('PolynomialFeatures is not fitted'); return X.map(row=>{const values=this.combinations.map(c=>c.reduce((v,i)=>v*row[i],1));return this.includeBias?[1,...values]:values;}); }
  fitTransform(X){ return this.fit(X).transform(X); }
  featureNames(names=[]){ const n=names.length?names:[...Array(this.nFeatures).keys()].map(i=>`x${i}`); const generated=this.combinations.map(c=>c.map(i=>n[i]).join('*'));return this.includeBias?['1',...generated]:generated; }
}

function identity(n){return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));}
function matVec(A,v){return A.map(r=>MathX.dot(r,v));}
function norm(v){return Math.sqrt(MathX.dot(v,v))||1;}
function outer(a,b){return a.map(x=>b.map(y=>x*y));}

export class PCA {
  constructor({components=2,maxIterations=500,tolerance=1e-9,seed=42}={}){Object.assign(this,{components,maxIterations,tolerance,seed});}
  fit(X){ Validation.matrix(X); const n=X.length,p=X[0].length,k=Math.min(this.components,p); this.mean=Array.from({length:p},(_,j)=>MathX.mean(X.map(r=>r[j]))); const C=X.map(r=>r.map((v,j)=>v-this.mean[j])); let cov=Array.from({length:p},(_,i)=>Array.from({length:p},(_,j)=>C.reduce((s,r)=>s+r[i]*r[j],0)/Math.max(n-1,1))); this.components_=[];this.explainedVariance=[]; const rng=new SeededRandom(this.seed);
    for(let comp=0;comp<k;comp++){ let v=Array.from({length:p},()=>rng.next()-.5),vn=norm(v);v=v.map(x=>x/vn); for(let it=0;it<this.maxIterations;it++){let next=matVec(cov,v),nn=norm(next);next=next.map(x=>x/nn);const diff=Math.sqrt(next.reduce((s,x,i)=>s+(x-v[i])**2,0));v=next;if(diff<this.tolerance)break;} const Av=matVec(cov,v),lambda=MathX.dot(v,Av);this.components_.push(v);this.explainedVariance.push(Math.max(lambda,0)); const o=outer(v,v);cov=cov.map((r,i)=>r.map((x,j)=>x-lambda*o[i][j])); }
    const totalVar=Array.from({length:p},(_,j)=>MathX.variance(C.map(r=>r[j]),true)).reduce((a,b)=>a+b,0)||1;this.explainedVarianceRatio=this.explainedVariance.map(v=>v/totalVar);return this; }
  transform(X){Validation.matrix(X);if(!this.components_)throw new Error('PCA is not fitted');return X.map(r=>{const c=r.map((v,j)=>v-this.mean[j]);return this.components_.map(pc=>MathX.dot(c,pc));});}
  fitTransform(X){return this.fit(X).transform(X);}
}

export class SoftmaxRegression {
  constructor({learningRate=0.05,epochs=2000,l2=0}={}){Object.assign(this,{learningRate,epochs,l2});}
  fit(X,y){Validation.supervised(X,y);this.classes=[...new Set(y)];const n=X.length,p=X[0].length,c=this.classes.length;this.weights=Array.from({length:c},()=>Array(p).fill(0));this.bias=Array(c).fill(0);const yi=y.map(v=>this.classes.indexOf(v));for(let e=0;e<this.epochs;e++){const gradW=Array.from({length:c},()=>Array(p).fill(0)),gradB=Array(c).fill(0);for(let i=0;i<n;i++){const logits=this.weights.map((w,k)=>this.bias[k]+MathX.dot(w,X[i])),probs=MathX.softmax(logits);for(let k=0;k<c;k++){const err=probs[k]-(yi[i]===k?1:0);gradB[k]+=err;for(let j=0;j<p;j++)gradW[k][j]+=err*X[i][j];}}for(let k=0;k<c;k++){this.bias[k]-=this.learningRate*gradB[k]/n;for(let j=0;j<p;j++)this.weights[k][j]-=this.learningRate*(gradW[k][j]/n+this.l2*this.weights[k][j]);}}return this;}
  predictProba(X){Validation.matrix(X);return X.map(r=>{const probs=MathX.softmax(this.weights.map((w,k)=>this.bias[k]+MathX.dot(w,r)));return Object.fromEntries(this.classes.map((c,i)=>[c,probs[i]]));});}
  predict(X){return this.predictProba(X).map(o=>Object.entries(o).sort((a,b)=>b[1]-a[1])[0][0]);}
}

export class HierarchicalClustering {
  constructor({clusters=2,linkage='average',distance='euclidean'}={}){if(!['single','complete','average'].includes(linkage))throw new Error('linkage must be single, complete, or average');Object.assign(this,{clusters,linkage,distance});}
  _d(a,b){return this.distance==='manhattan'?MathX.manhattan(a,b):MathX.euclidean(a,b);}
  _clusterDistance(A,B,X){const ds=[];for(const i of A)for(const j of B)ds.push(this._d(X[i],X[j]));if(this.linkage==='single')return Math.min(...ds);if(this.linkage==='complete')return Math.max(...ds);return MathX.mean(ds);}
  fit(X){Validation.matrix(X);if(this.clusters<1||this.clusters>X.length)throw new Error('clusters must be between 1 and number of rows');let groups=X.map((_,i)=>[i]);this.dendrogram=[];let nextId=X.length;let ids=X.map((_,i)=>i);while(groups.length>this.clusters){let best={a:0,b:1,d:Infinity};for(let a=0;a<groups.length;a++)for(let b=a+1;b<groups.length;b++){const d=this._clusterDistance(groups[a],groups[b],X);if(d<best.d)best={a,b,d};}const merged=[...groups[best.a],...groups[best.b]];this.dendrogram.push({left:ids[best.a],right:ids[best.b],distance:best.d,size:merged.length,id:nextId});groups=groups.filter((_,i)=>i!==best.a&&i!==best.b);ids=ids.filter((_,i)=>i!==best.a&&i!==best.b);groups.push(merged);ids.push(nextId++);}this.clusters_=groups;this.labels=Array(X.length).fill(-1);groups.forEach((g,k)=>g.forEach(i=>this.labels[i]=k));return this;}
}

export class RobustScaler {
  constructor({quantileRange=[.25,.75]}={}){this.quantileRange=quantileRange;}
  _q(a,q){const s=[...a].sort((x,y)=>x-y),p=(s.length-1)*q,l=Math.floor(p),h=Math.ceil(p);return l===h?s[l]:s[l]+(s[h]-s[l])*(p-l);}
  fit(X){Validation.matrix(X);const p=X[0].length;this.center=[];this.scale=[];for(let j=0;j<p;j++){const c=X.map(r=>r[j]),q1=this._q(c,this.quantileRange[0]),q3=this._q(c,this.quantileRange[1]);this.center.push(this._q(c,.5));this.scale.push(q3-q1||1);}return this;}
  transform(X){Validation.matrix(X);return X.map(r=>r.map((v,j)=>(v-this.center[j])/this.scale[j]));}
  fitTransform(X){return this.fit(X).transform(X);}
}

export class ExponentialMovingAverage {
  constructor({alpha=.2}={}){if(alpha<=0||alpha>1)throw new Error('alpha must be in (0, 1]');this.alpha=alpha;this.value=null;}
  update(x){if(typeof x!=='number'||!Number.isFinite(x))throw new Error('EMA value must be finite');this.value=this.value===null?x:this.alpha*x+(1-this.alpha)*this.value;return this.value;}
  fit(values){this.value=null;return values.map(v=>this.update(v));}
}
