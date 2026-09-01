export class Tokenizer {
  constructor(options = {}) {
    this.lowercase = options.lowercase ?? true;
    this.removeStopWords = options.removeStopWords ?? false;
    this.stopWords = new Set(options.stopWords || ['the','a','an','and','or','of','to','in','for','on','with','is','are','by']);
  }
  tokenize(text='') {
    let s = String(text).normalize('NFKD').replace(/[\u0300-\u036f]/g,'');
    if (this.lowercase) s = s.toLowerCase();
    let tokens = s.match(/[a-z0-9]+(?:[-_.\/][a-z0-9]+)*/gi) || [];
    if (this.removeStopWords) tokens = tokens.filter(t => !this.stopWords.has(t));
    return tokens;
  }
  sentences(text='') { return String(text).split(/(?<=[.!?])\s+/).map(s=>s.trim()).filter(Boolean); }
  ngrams(tokens, n=2) { const out=[]; for(let i=0;i<=tokens.length-n;i++) out.push(tokens.slice(i,i+n).join(' ')); return out; }
}

export class TFIDF {
  constructor(){ this.vocab=[]; this.idf=[]; }
  fit(docs){
    const tok = new Tokenizer({removeStopWords:true});
    const tokenDocs = docs.map(d=>new Set(tok.tokenize(d)));
    this.vocab = [...new Set(tokenDocs.flatMap(s=>[...s]))];
    const N = docs.length;
    this.idf = this.vocab.map(term => Math.log((1+N)/(1+tokenDocs.filter(s=>s.has(term)).length))+1);
    return this;
  }
  transform(doc){
    const tok = new Tokenizer({removeStopWords:true});
    const tokens = tok.tokenize(doc); const counts={}; tokens.forEach(t=>counts[t]=(counts[t]||0)+1);
    return this.vocab.map((term,i)=>((counts[term]||0)/Math.max(tokens.length,1))*this.idf[i]);
  }
  fitTransform(docs){ this.fit(docs); return docs.map(d=>this.transform(d)); }
}

export const MathX = {
  mean: a => a.reduce((s,v)=>s+v,0)/Math.max(a.length,1),
  variance(a){ const m=this.mean(a); return a.reduce((s,v)=>s+(v-m)**2,0)/Math.max(a.length,1); },
  std(a){ return Math.sqrt(this.variance(a)); },
  euclidean:(a,b)=>Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0)),
  manhattan:(a,b)=>a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0),
  cosine(a,b){ const dot=a.reduce((s,v,i)=>s+v*b[i],0); const na=Math.sqrt(a.reduce((s,v)=>s+v*v,0)); const nb=Math.sqrt(b.reduce((s,v)=>s+v*v,0)); return dot/(na*nb||1); },
  jaccard(a,b){ const A=new Set(a),B=new Set(b); const inter=[...A].filter(x=>B.has(x)).length; return inter/(new Set([...A,...B]).size||1); }
};

export class StandardScaler {
  fit(X){ const cols=X[0].length; this.means=[]; this.stds=[]; for(let j=0;j<cols;j++){ const c=X.map(r=>r[j]); this.means.push(MathX.mean(c)); this.stds.push(MathX.std(c)||1); } return this; }
  transform(X){ return X.map(r=>r.map((v,j)=>(v-this.means[j])/this.stds[j])); }
  fitTransform(X){ return this.fit(X).transform(X); }
}

export class MinMaxScaler {
  fit(X){ const cols=X[0].length; this.min=[]; this.max=[]; for(let j=0;j<cols;j++){const c=X.map(r=>r[j]);this.min.push(Math.min(...c));this.max.push(Math.max(...c));} return this; }
  transform(X){ return X.map(r=>r.map((v,j)=>(v-this.min[j])/((this.max[j]-this.min[j])||1))); }
}

function gini(y){ const counts={}; y.forEach(v=>counts[v]=(counts[v]||0)+1); return 1-Object.values(counts).reduce((s,c)=>s+(c/y.length)**2,0); }
function majority(y){ const counts={}; y.forEach(v=>counts[v]=(counts[v]||0)+1); return Object.entries(counts).sort((a,b)=>b[1]-a[1])[0][0]; }

export class DecisionTreeClassifier {
  constructor({maxDepth=8,minSamplesSplit=2}={}){ this.maxDepth=maxDepth; this.minSamplesSplit=minSamplesSplit; }
  fit(X,y){ this.root=this._build(X,y,0); return this; }
  _build(X,y,depth){
    if(depth>=this.maxDepth||X.length<this.minSamplesSplit||new Set(y).size===1) return {leaf:true,value:majority(y)};
    let best=null, base=gini(y);
    for(let f=0;f<X[0].length;f++){
      const vals=[...new Set(X.map(r=>r[f]))].sort((a,b)=>a-b);
      for(let i=0;i<vals.length-1;i++){
        const t=(vals[i]+vals[i+1])/2, li=[],ri=[]; X.forEach((r,idx)=>(r[f]<=t?li:ri).push(idx));
        if(!li.length||!ri.length) continue;
        const score=(li.length/X.length)*gini(li.map(i=>y[i]))+(ri.length/X.length)*gini(ri.map(i=>y[i]));
        const gain=base-score; if(!best||gain>best.gain) best={f,t,li,ri,gain};
      }
    }
    if(!best||best.gain<=0) return {leaf:true,value:majority(y)};
    return {leaf:false,feature:best.f,threshold:best.t,
      left:this._build(best.li.map(i=>X[i]),best.li.map(i=>y[i]),depth+1),
      right:this._build(best.ri.map(i=>X[i]),best.ri.map(i=>y[i]),depth+1)};
  }
  _predictOne(row,n=this.root){ if(n.leaf)return n.value; return this._predictOne(row,row[n.feature]<=n.threshold?n.left:n.right); }
  predict(X){ return X.map(r=>this._predictOne(r)); }
}

export class KNNClassifier {
  constructor({k=5}={}){this.k=k;}
  fit(X,y){this.X=X;this.y=y;return this;}
  predict(X){return X.map(row=>{const n=this.X.map((r,i)=>({d:MathX.euclidean(r,row),y:this.y[i]})).sort((a,b)=>a.d-b.d).slice(0,this.k);return majority(n.map(v=>v.y));});}
}

export class GaussianNaiveBayes {
  fit(X,y){this.classes=[...new Set(y)];this.stats={}; for(const c of this.classes){const rows=X.filter((_,i)=>y[i]===c);this.stats[c]=rows[0].map((_,j)=>{const col=rows.map(r=>r[j]);return {m:MathX.mean(col),v:MathX.variance(col)+1e-9};});}return this;}
  _logProb(row,c){return this.stats[c].reduce((s,st,j)=>s+(-.5*Math.log(2*Math.PI*st.v)-((row[j]-st.m)**2)/(2*st.v)),Math.log(1/this.classes.length));}
  predict(X){return X.map(r=>this.classes.map(c=>[c,this._logProb(r,c)]).sort((a,b)=>b[1]-a[1])[0][0]);}
}

export class MultinomialNaiveBayesText {
  constructor(){this.tokenizer=new Tokenizer({removeStopWords:true});this.docs=[];}
  train(text,label){this.docs.push({text,label});return this;}
  fit(){this.labels=[...new Set(this.docs.map(d=>d.label))];this.vocab=new Set();this.counts={};this.totals={};this.docCounts={};for(const l of this.labels){this.counts[l]={};this.totals[l]=0;this.docCounts[l]=0;}for(const d of this.docs){this.docCounts[d.label]++;for(const t of this.tokenizer.tokenize(d.text)){this.vocab.add(t);this.counts[d.label][t]=(this.counts[d.label][t]||0)+1;this.totals[d.label]++;}}return this;}
  predict(text){if(!this.labels)this.fit();const toks=this.tokenizer.tokenize(text);const N=this.docs.length,V=this.vocab.size;return this.labels.map(l=>{let s=Math.log(this.docCounts[l]/N);for(const t of toks)s+=Math.log(((this.counts[l][t]||0)+1)/(this.totals[l]+V));return [l,s];}).sort((a,b)=>b[1]-a[1])[0][0];}
}

export class KMeans {
  constructor({clusters=3,maxIterations=100,seed=42}={}){this.k=clusters;this.maxIterations=maxIterations;this.seed=seed;}
  fit(X){if(X.length<this.k)throw new Error('clusters cannot exceed number of rows');this.centroids=X.slice(0,this.k).map(r=>[...r]);let labels=[];for(let it=0;it<this.maxIterations;it++){labels=X.map(r=>this.centroids.map((c,i)=>[i,MathX.euclidean(r,c)]).sort((a,b)=>a[1]-b[1])[0][0]);const next=this.centroids.map((c,k)=>{const rows=X.filter((_,i)=>labels[i]===k);if(!rows.length)return c;return c.map((_,j)=>MathX.mean(rows.map(r=>r[j])));});const movement=next.reduce((s,c,i)=>s+MathX.euclidean(c,this.centroids[i]),0);this.centroids=next;if(movement<1e-8)break;}this.labels=labels;return this;}
  predict(X){return X.map(r=>this.centroids.map((c,i)=>[i,MathX.euclidean(r,c)]).sort((a,b)=>a[1]-b[1])[0][0]);}
}

export class DBSCAN {
  constructor({epsilon=.5,minPoints=5}={}){this.eps=epsilon;this.minPoints=minPoints;}
  fit(X){const labels=Array(X.length).fill(undefined);let cluster=0;const region=i=>X.map((r,j)=>MathX.euclidean(X[i],r)<=this.eps?j:-1).filter(j=>j>=0);for(let i=0;i<X.length;i++){if(labels[i]!==undefined)continue;let n=region(i);if(n.length<this.minPoints){labels[i]=-1;continue;}labels[i]=cluster;const seeds=[...n];for(let s=0;s<seeds.length;s++){const p=seeds[s];if(labels[p]===-1)labels[p]=cluster;if(labels[p]!==undefined)continue;labels[p]=cluster;const pn=region(p);if(pn.length>=this.minPoints)for(const q of pn)if(!seeds.includes(q))seeds.push(q);}cluster++;}this.labels=labels;return this;}
}

export const Metrics = {
  accuracy(y,p){return y.filter((v,i)=>v===p[i]).length/Math.max(y.length,1);},
  confusionMatrix(y,p){const labels=[...new Set([...y,...p])];const m=Object.fromEntries(labels.map(a=>[a,Object.fromEntries(labels.map(b=>[b,0]))]));y.forEach((v,i)=>m[v][p[i]]++);return m;},
  precisionRecallF1(y,p,label){let tp=0,fp=0,fn=0;y.forEach((v,i)=>{if(p[i]===label&&v===label)tp++;else if(p[i]===label)fp++;else if(v===label)fn++;});const precision=tp/(tp+fp||1),recall=tp/(tp+fn||1);return {precision,recall,f1:2*precision*recall/(precision+recall||1)};}
};

export function trainTestSplit(X,y,testSize=.2,seed=42){let idx=X.map((_,i)=>i);let r=seed>>>0;const rand=()=>((r=(1664525*r+1013904223)>>>0)/4294967296);for(let i=idx.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}const n=Math.max(1,Math.round(idx.length*testSize)),te=idx.slice(0,n),tr=idx.slice(n);return {XTrain:tr.map(i=>X[i]),XTest:te.map(i=>X[i]),yTrain:tr.map(i=>y[i]),yTest:te.map(i=>y[i])};}

export class RuleEngine {
  constructor(){this.rules=[];}
  add(rule){this.rules.push(rule);return this;}
  evaluate(input){return this.rules.filter(r=>r.when(input)).map(r=>({id:r.id||null,label:r.label||null,severity:r.severity||'info',reason:r.reason?.(input)||r.reason||'Rule matched'}));}
}

export class EnsembleClassifier {
  constructor(models=[]){this.models=models;}
  fit(X,y){this.models.forEach(m=>m.fit(X,y));return this;}
  predict(X){const ps=this.models.map(m=>m.predict(X));return X.map((_,i)=>majority(ps.map(p=>p[i])));}
}

export class ModelStorage {
  static save(name,model){localStorage.setItem(`aladinai:${name}`,JSON.stringify(model));}
  static load(name){const v=localStorage.getItem(`aladinai:${name}`);return v?JSON.parse(v):null;}
  static remove(name){localStorage.removeItem(`aladinai:${name}`);}
}

export const AladinAI = {
  version:'0.1.0',
  Tokenizer, TFIDF, MathX, StandardScaler, MinMaxScaler,
  DecisionTreeClassifier, KNNClassifier, GaussianNaiveBayes, MultinomialNaiveBayesText,
  KMeans, DBSCAN, Metrics, trainTestSplit, RuleEngine, EnsembleClassifier, ModelStorage
};

if(typeof window!=='undefined') window.AladinAI = AladinAI;
export default AladinAI;
