const DEFAULT_STOP_WORDS=['the','a','an','and','or','of','to','in','for','on','with','is','are','by'];

function assertMatrix(X,name='X'){
  if(!Array.isArray(X)||!X.length||!Array.isArray(X[0]))throw new Error(`${name} must be a non-empty 2D array`);
  const cols=X[0].length;if(!cols)throw new Error(`${name} must have at least one feature`);
  X.forEach((r,i)=>{if(!Array.isArray(r)||r.length!==cols)throw new Error(`${name}[${i}] has inconsistent feature count`);r.forEach((v,j)=>{if(typeof v!=='number'||!Number.isFinite(v))throw new Error(`${name}[${i}][${j}] must be a finite number`);});});
  return X;
}
function assertSupervised(X,y){assertMatrix(X);if(!Array.isArray(y)||y.length!==X.length)throw new Error('y length must match X rows');if(!y.length)throw new Error('y cannot be empty');}
function countsOf(values){const c={};values.forEach(v=>c[v]=(c[v]||0)+1);return c;}
function majority(y){const c=countsOf(y);return Object.entries(c).sort((a,b)=>b[1]-a[1])[0]?.[0];}
function gini(y){const c=countsOf(y),n=y.length||1;return 1-Object.values(c).reduce((s,v)=>s+(v/n)**2,0);}
function seededRandom(seed=42){let r=seed>>>0;return()=>((r=(1664525*r+1013904223)>>>0)/4294967296);}
function shuffleIndices(n,seed=42){const a=[...Array(n).keys()],rand=seededRandom(seed);for(let i=a.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}

export class Tokenizer{
  constructor(options={}){this.lowercase=options.lowercase??true;this.removeStopWords=options.removeStopWords??false;this.stopWords=new Set(options.stopWords||DEFAULT_STOP_WORDS);this.keepNumbers=options.keepNumbers??true;this.minLength=options.minLength??1;}
  normalize(text=''){let s=String(text).normalize('NFKD').replace(/[\u0300-\u036f]/g,'');return this.lowercase?s.toLowerCase():s;}
  tokenize(text=''){let tokens=this.normalize(text).match(/[a-z0-9]+(?:[-_.\/][a-z0-9]+)*/gi)||[];if(!this.keepNumbers)tokens=tokens.filter(t=>!/^\d+(?:\.\d+)?$/.test(t));if(this.removeStopWords)tokens=tokens.filter(t=>!this.stopWords.has(t));return tokens.filter(t=>t.length>=this.minLength);}
  sentences(text=''){return String(text).split(/(?<=[.!?])\s+/).map(s=>s.trim()).filter(Boolean);}
  ngrams(tokens,n=2){if(!Number.isInteger(n)||n<1)throw new Error('n must be a positive integer');const out=[];for(let i=0;i<=tokens.length-n;i++)out.push(tokens.slice(i,i+n).join(' '));return out;}
  termFrequency(text=''){const c={};for(const t of this.tokenize(text))c[t]=(c[t]||0)+1;return c;}
}

export class TFIDF{
  constructor(options={}){this.tokenizer=options.tokenizer||new Tokenizer({removeStopWords:true});this.sublinearTf=options.sublinearTf??false;this.normalize=options.normalize??false;this.vocab=[];this.idf=[];}
  fit(docs){if(!Array.isArray(docs)||!docs.length)throw new Error('docs must be a non-empty array');const sets=docs.map(d=>new Set(this.tokenizer.tokenize(d)));this.vocab=[...new Set(sets.flatMap(s=>[...s]))];const N=docs.length;this.idf=this.vocab.map(term=>Math.log((1+N)/(1+sets.filter(s=>s.has(term)).length))+1);return this;}
  transform(doc){if(!this.vocab.length)return[];const tokens=this.tokenizer.tokenize(doc),c={};tokens.forEach(t=>c[t]=(c[t]||0)+1);let v=this.vocab.map((term,i)=>{const raw=c[term]||0,tf=this.sublinearTf?(raw?1+Math.log(raw):0):raw/Math.max(tokens.length,1);return tf*this.idf[i];});if(this.normalize){const norm=Math.sqrt(v.reduce((s,x)=>s+x*x,0))||1;v=v.map(x=>x/norm);}return v;}
  transformMany(docs){return docs.map(d=>this.transform(d));}
  fitTransform(docs){return this.fit(docs).transformMany(docs);}
  topTerms(vector,n=10){return vector.map((score,i)=>({term:this.vocab[i],score})).sort((a,b)=>b.score-a.score).slice(0,n);}
}

export const MathX={
  sum:a=>a.reduce((s,v)=>s+v,0),
  mean:a=>a.reduce((s,v)=>s+v,0)/Math.max(a.length,1),
  variance(a,sample=false){if(!a.length)return 0;const m=this.mean(a),d=sample?Math.max(a.length-1,1):a.length;return a.reduce((s,v)=>s+(v-m)**2,0)/d;},
  std(a,sample=false){return Math.sqrt(this.variance(a,sample));},
  dot:(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),
  euclidean:(a,b)=>Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0)),
  manhattan:(a,b)=>a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0),
  cosine(a,b){const dot=this.dot(a,b),na=Math.sqrt(this.dot(a,a)),nb=Math.sqrt(this.dot(b,b));return dot/(na*nb||1);},
  jaccard(a,b){const A=new Set(a),B=new Set(b),u=new Set([...A,...B]);return [...A].filter(x=>B.has(x)).length/(u.size||1);},
  argmax:a=>a.reduce((best,v,i)=>v>a[best]?i:best,0),
  softmax(a){const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),s=e.reduce((x,y)=>x+y,0)||1;return e.map(v=>v/s);}
};

export class StandardScaler{
  fit(X){assertMatrix(X);const p=X[0].length;this.means=[];this.stds=[];for(let j=0;j<p;j++){const c=X.map(r=>r[j]);this.means.push(MathX.mean(c));this.stds.push(MathX.std(c)||1);}return this;}
  transform(X){assertMatrix(X);if(!this.means)throw new Error('StandardScaler is not fitted');return X.map(r=>r.map((v,j)=>(v-this.means[j])/this.stds[j]));}
  inverseTransform(X){assertMatrix(X);return X.map(r=>r.map((v,j)=>v*this.stds[j]+this.means[j]));}
  fitTransform(X){return this.fit(X).transform(X);}
}

export class MinMaxScaler{
  constructor({featureRange=[0,1]}={}){this.featureRange=featureRange;}
  fit(X){assertMatrix(X);const p=X[0].length;this.min=[];this.max=[];for(let j=0;j<p;j++){const c=X.map(r=>r[j]);this.min.push(Math.min(...c));this.max.push(Math.max(...c));}return this;}
  transform(X){assertMatrix(X);if(!this.min)throw new Error('MinMaxScaler is not fitted');const [lo,hi]=this.featureRange;return X.map(r=>r.map((v,j)=>lo+(v-this.min[j])/((this.max[j]-this.min[j])||1)*(hi-lo)));}
  inverseTransform(X){const [lo,hi]=this.featureRange;return X.map(r=>r.map((v,j)=>this.min[j]+((v-lo)/((hi-lo)||1))*(this.max[j]-this.min[j])));}
  fitTransform(X){return this.fit(X).transform(X);}
}

export class DecisionTreeClassifier{
  constructor({maxDepth=8,minSamplesSplit=2,minSamplesLeaf=1,minGain=1e-12}={}){Object.assign(this,{maxDepth,minSamplesSplit,minSamplesLeaf,minGain});}
  fit(X,y){assertSupervised(X,y);this.nFeatures=X[0].length;this.classes=[...new Set(y)];this.featureGain=Array(this.nFeatures).fill(0);this.root=this._build(X,y,0);const total=this.featureGain.reduce((a,b)=>a+b,0)||1;this.featureImportances=this.featureGain.map(v=>v/total);return this;}
  _leaf(y){const c=countsOf(y),n=y.length||1,proba={};Object.entries(c).forEach(([k,v])=>proba[k]=v/n);return{leaf:true,value:majority(y),proba,count:y.length};}
  _build(X,y,depth){if(depth>=this.maxDepth||X.length<this.minSamplesSplit||new Set(y).size===1)return this._leaf(y);let best=null,base=gini(y);for(let f=0;f<X[0].length;f++){const vals=[...new Set(X.map(r=>r[f]))].sort((a,b)=>a-b);for(let i=0;i<vals.length-1;i++){const t=(vals[i]+vals[i+1])/2,li=[],ri=[];X.forEach((r,k)=>(r[f]<=t?li:ri).push(k));if(li.length<this.minSamplesLeaf||ri.length<this.minSamplesLeaf)continue;const score=(li.length/X.length)*gini(li.map(k=>y[k]))+(ri.length/X.length)*gini(ri.map(k=>y[k])),gain=base-score;if(!best||gain>best.gain)best={f,t,li,ri,gain};}}
    if(!best||best.gain<=this.minGain)return this._leaf(y);this.featureGain[best.f]+=best.gain*X.length;return{leaf:false,feature:best.f,threshold:best.t,gain:best.gain,count:X.length,left:this._build(best.li.map(i=>X[i]),best.li.map(i=>y[i]),depth+1),right:this._build(best.ri.map(i=>X[i]),best.ri.map(i=>y[i]),depth+1)};}
  _node(row,n=this.root){if(n.leaf)return n;return this._node(row,row[n.feature]<=n.threshold?n.left:n.right);}
  predict(X){assertMatrix(X);return X.map(r=>this._node(r).value);}
  predictProba(X){assertMatrix(X);return X.map(r=>this._node(r).proba);}
  explain(row,featureNames=[]){const path=[];let n=this.root;while(n&&!n.leaf){const value=row[n.feature],left=value<=n.threshold;path.push({feature:featureNames[n.feature]??n.feature,value,threshold:n.threshold,direction:left?'left':'right',gain:n.gain});n=left?n.left:n.right;}return{prediction:n?.value,probabilities:n?.proba||{},path};}
}

export class KNNClassifier{
  constructor({k=5,distance='euclidean',weighted=false}={}){this.k=k;this.distance=distance;this.weighted=weighted;}
  fit(X,y){assertSupervised(X,y);this.X=X.map(r=>[...r]);this.y=[...y];this.classes=[...new Set(y)];return this;}
  _distance(a,b){return this.distance==='manhattan'?MathX.manhattan(a,b):MathX.euclidean(a,b);}
  kneighbors(row){return this.X.map((r,i)=>({distance:this._distance(r,row),label:this.y[i],index:i})).sort((a,b)=>a.distance-b.distance).slice(0,Math.min(this.k,this.X.length));}
  predictProba(X){assertMatrix(X);return X.map(row=>{const c={};this.classes.forEach(k=>c[k]=0);for(const n of this.kneighbors(row)){const w=this.weighted?1/(n.distance+1e-12):1;c[n.label]+=w;}const s=Object.values(c).reduce((a,b)=>a+b,0)||1;Object.keys(c).forEach(k=>c[k]/=s);return c;});}
  predict(X){return this.predictProba(X).map(o=>Object.entries(o).sort((a,b)=>b[1]-a[1])[0][0]);}
}

export class GaussianNaiveBayes{
  constructor({varSmoothing=1e-9}={}){this.varSmoothing=varSmoothing;}
  fit(X,y){assertSupervised(X,y);this.classes=[...new Set(y)];this.stats={};this.priors={};for(const c of this.classes){const rows=X.filter((_,i)=>y[i]===c);this.priors[c]=rows.length/X.length;this.stats[c]=rows[0].map((_,j)=>{const col=rows.map(r=>r[j]);return{mean:MathX.mean(col),variance:MathX.variance(col)+this.varSmoothing};});}return this;}
  _logProb(row,c){let s=Math.log(this.priors[c]||1e-12);this.stats[c].forEach((st,j)=>{s+=-.5*Math.log(2*Math.PI*st.variance)-((row[j]-st.mean)**2)/(2*st.variance);});return s;}
  predictLogProba(X){assertMatrix(X);return X.map(r=>Object.fromEntries(this.classes.map(c=>[c,this._logProb(r,c)])));}
  predictProba(X){return this.predictLogProba(X).map(o=>{const labels=Object.keys(o),p=MathX.softmax(labels.map(k=>o[k]));return Object.fromEntries(labels.map((k,i)=>[k,p[i]]));});}
  predict(X){return this.predictLogProba(X).map(o=>Object.entries(o).sort((a,b)=>b[1]-a[1])[0][0]);}
}

export class MultinomialNaiveBayesText{
  constructor({alpha=1,tokenizer=null}={}){this.alpha=alpha;this.tokenizer=tokenizer||new Tokenizer({removeStopWords:true});this.docs=[];}
  train(text,label){this.docs.push({text:String(text),label});return this;}
  partialFit(text,label){this.docs.push({text:String(text),label});return this.fit();}
  fit(){if(!this.docs.length)throw new Error('No training documents');this.labels=[...new Set(this.docs.map(d=>d.label))];this.vocab=new Set();this.counts={};this.totals={};this.docCounts={};for(const l of this.labels){this.counts[l]={};this.totals[l]=0;this.docCounts[l]=0;}for(const d of this.docs){this.docCounts[d.label]++;for(const t of this.tokenizer.tokenize(d.text)){this.vocab.add(t);this.counts[d.label][t]=(this.counts[d.label][t]||0)+1;this.totals[d.label]++;}}return this;}
  _scores(text){if(!this.labels)this.fit();const toks=this.tokenizer.tokenize(text),N=this.docs.length,V=Math.max(this.vocab.size,1);return Object.fromEntries(this.labels.map(l=>{let s=Math.log(this.docCounts[l]/N);for(const t of toks)s+=Math.log(((this.counts[l][t]||0)+this.alpha)/(this.totals[l]+this.alpha*V));return[l,s];}));}
  predictProba(text){const scores=this._scores(text),labels=Object.keys(scores),p=MathX.softmax(labels.map(l=>scores[l]));return Object.fromEntries(labels.map((l,i)=>[l,p[i]]));}
  predict(text){const p=this.predictProba(text);return Object.entries(p).sort((a,b)=>b[1]-a[1])[0][0];}
}

export class KMeans{
  constructor({clusters=3,maxIterations=100,seed=42,tolerance=1e-8}={}){this.k=clusters;this.maxIterations=maxIterations;this.seed=seed;this.tolerance=tolerance;}
  fit(X){assertMatrix(X);if(X.length<this.k)throw new Error('clusters cannot exceed number of rows');const idx=shuffleIndices(X.length,this.seed).slice(0,this.k);this.centroids=idx.map(i=>[...X[i]]);let labels=[];for(let it=0;it<this.maxIterations;it++){labels=this.predict(X);const next=this.centroids.map((c,k)=>{const rows=X.filter((_,i)=>labels[i]===k);return rows.length?c.map((_,j)=>MathX.mean(rows.map(r=>r[j]))):c;});const movement=next.reduce((s,c,i)=>s+MathX.euclidean(c,this.centroids[i]),0);this.centroids=next;this.iterations=it+1;if(movement<this.tolerance)break;}this.labels=labels;this.inertia=X.reduce((s,r,i)=>s+MathX.euclidean(r,this.centroids[this.labels[i]])**2,0);return this;}
  predict(X){assertMatrix(X);return X.map(r=>this.centroids.map((c,i)=>[i,MathX.euclidean(r,c)]).sort((a,b)=>a[1]-b[1])[0][0]);}
}

export class DBSCAN{
  constructor({epsilon=.5,minPoints=5}={}){this.eps=epsilon;this.minPoints=minPoints;}
  fit(X){assertMatrix(X);const labels=Array(X.length).fill(undefined),visited=Array(X.length).fill(false);let cluster=0;const region=i=>X.map((r,j)=>MathX.euclidean(X[i],r)<=this.eps?j:-1).filter(j=>j>=0);for(let i=0;i<X.length;i++){if(visited[i])continue;visited[i]=true;let neighbors=region(i);if(neighbors.length<this.minPoints){labels[i]=-1;continue;}labels[i]=cluster;const seeds=[...neighbors];for(let s=0;s<seeds.length;s++){const p=seeds[s];if(!visited[p]){visited[p]=true;const pn=region(p);if(pn.length>=this.minPoints)for(const q of pn)if(!seeds.includes(q))seeds.push(q);}if(labels[p]===undefined||labels[p]===-1)labels[p]=cluster;}cluster++;}this.labels=labels;this.nClusters=cluster;this.noiseCount=labels.filter(v=>v===-1).length;return this;}
}

export const Metrics={
  accuracy(y,p){if(y.length!==p.length)throw new Error('Prediction length mismatch');return y.filter((v,i)=>v===p[i]).length/Math.max(y.length,1);},
  mse(y,p){return y.reduce((s,v,i)=>s+(v-p[i])**2,0)/Math.max(y.length,1);},
  rmse(y,p){return Math.sqrt(this.mse(y,p));},
  mae(y,p){return y.reduce((s,v,i)=>s+Math.abs(v-p[i]),0)/Math.max(y.length,1);},
  r2(y,p){const m=MathX.mean(y),res=y.reduce((s,v,i)=>s+(v-p[i])**2,0),tot=y.reduce((s,v)=>s+(v-m)**2,0);return 1-res/(tot||1);},
  confusionMatrix(y,p){const labels=[...new Set([...y,...p])],m=Object.fromEntries(labels.map(a=>[a,Object.fromEntries(labels.map(b=>[b,0]))]));y.forEach((v,i)=>m[v][p[i]]++);return m;},
  precisionRecallF1(y,p,label){let tp=0,fp=0,fn=0;y.forEach((v,i)=>{if(p[i]===label&&v===label)tp++;else if(p[i]===label)fp++;else if(v===label)fn++;});const precision=tp/(tp+fp||1),recall=tp/(tp+fn||1);return{precision,recall,f1:2*precision*recall/(precision+recall||1),support:y.filter(v=>v===label).length};}
};

export function trainTestSplit(X,y,testSize=.2,seed=42){assertSupervised(X,y);if(testSize<=0||testSize>=1)throw new Error('testSize must be between 0 and 1');const idx=shuffleIndices(X.length,seed),n=Math.max(1,Math.min(X.length-1,Math.round(X.length*testSize))),te=idx.slice(0,n),tr=idx.slice(n);return{XTrain:tr.map(i=>X[i]),XTest:te.map(i=>X[i]),yTrain:tr.map(i=>y[i]),yTest:te.map(i=>y[i]),trainIndices:tr,testIndices:te};}

export class RuleEngine{
  constructor(){this.rules=[];}
  add(rule){if(typeof rule?.when!=='function')throw new Error('Rule requires a when(input) function');this.rules.push(rule);return this;}
  remove(id){this.rules=this.rules.filter(r=>r.id!==id);return this;}
  evaluate(input){const out=[];for(const r of this.rules){let match=false;try{match=Boolean(r.when(input));}catch(error){out.push({id:r.id||null,label:r.label||null,severity:'error',reason:`Rule evaluation failed: ${error.message}`});continue;}if(match)out.push({id:r.id||null,label:r.label||null,severity:r.severity||'info',reason:typeof r.reason==='function'?r.reason(input):(r.reason||'Rule matched')});}return out;}
}

export class EnsembleClassifier{
  constructor(models=[],weights=null){this.models=models;this.weights=weights||models.map(()=>1);}
  fit(X,y){this.models.forEach(m=>m.fit(X,y));return this;}
  predictProba(X){const predictions=this.models.map(m=>m.predict(X));const total=this.weights.reduce((a,b)=>a+b,0)||1;return X.map((_,i)=>{const c={};predictions.forEach((p,j)=>c[p[i]]=(c[p[i]]||0)+this.weights[j]);Object.keys(c).forEach(k=>c[k]/=total);return c;});}
  predict(X){return this.predictProba(X).map(o=>Object.entries(o).sort((a,b)=>b[1]-a[1])[0][0]);}
}

export class ModelStorage{
  static _storage(){if(typeof localStorage==='undefined')throw new Error('localStorage is not available in this environment');return localStorage;}
  static save(name,payload){this._storage().setItem(`aladinai:${name}`,JSON.stringify(payload));return name;}
  static load(name){const v=this._storage().getItem(`aladinai:${name}`);return v?JSON.parse(v):null;}
  static remove(name){this._storage().removeItem(`aladinai:${name}`);}
  static list(){const s=this._storage(),out=[];for(let i=0;i<s.length;i++){const k=s.key(i);if(k?.startsWith('aladinai:'))out.push(k.slice(9));}return out;}
}

export const AladinAI={version:'0.2.0',Tokenizer,TFIDF,MathX,StandardScaler,MinMaxScaler,DecisionTreeClassifier,KNNClassifier,GaussianNaiveBayes,MultinomialNaiveBayesText,KMeans,DBSCAN,Metrics,trainTestSplit,RuleEngine,EnsembleClassifier,ModelStorage};
if(typeof window!=='undefined')window.AladinAI=AladinAI;
export default AladinAI;
