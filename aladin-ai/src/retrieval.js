import { Tokenizer, TFIDF, MathX } from './aladin-ai.js';
import { Validation } from './advanced.js';

export class CountVectorizer {
  constructor({lowercase=true,removeStopWords=true,minDf=1,maxFeatures=null,ngramRange=[1,1],binary=false}={}){
    Object.assign(this,{lowercase,removeStopWords,minDf,maxFeatures,ngramRange,binary});
    this.tokenizer=new Tokenizer({lowercase,removeStopWords});
  }
  _terms(text){
    const tokens=this.tokenizer.tokenize(text),out=[];
    for(let n=this.ngramRange[0];n<=this.ngramRange[1];n++){
      if(n===1) out.push(...tokens); else out.push(...this.tokenizer.ngrams(tokens,n));
    }
    return out;
  }
  fit(docs){
    if(!Array.isArray(docs)||!docs.length) throw new Error('CountVectorizer requires a non-empty document array');
    const df=new Map(),tf=new Map();
    for(const d of docs){
      const terms=this._terms(d),seen=new Set();
      for(const t of terms){tf.set(t,(tf.get(t)||0)+1);seen.add(t);}
      for(const t of seen) df.set(t,(df.get(t)||0)+1);
    }
    let terms=[...df.keys()].filter(t=>df.get(t)>=this.minDf);
    terms.sort((a,b)=>(tf.get(b)-tf.get(a))||a.localeCompare(b));
    if(this.maxFeatures) terms=terms.slice(0,this.maxFeatures);
    this.vocabulary=terms;
    this.index=new Map(terms.map((t,i)=>[t,i]));
    return this;
  }
  transform(docs){
    if(!this.index) throw new Error('CountVectorizer is not fitted');
    const input=Array.isArray(docs)?docs:[docs];
    return input.map(d=>{
      const row=Array(this.vocabulary.length).fill(0);
      for(const t of this._terms(d)) if(this.index.has(t)){const i=this.index.get(t);row[i]=this.binary?1:row[i]+1;}
      return row;
    });
  }
  fitTransform(docs){return this.fit(docs).transform(docs);}
  getFeatureNames(){return [...(this.vocabulary||[])];}
}

export class FeatureHasher {
  constructor({features=1024,signed=true,tokenizerOptions={removeStopWords:true}}={}){
    if(!Number.isInteger(features)||features<2) throw new Error('features must be an integer >= 2');
    this.features=features;this.signed=signed;this.tokenizer=new Tokenizer(tokenizerOptions);
  }
  _hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return h>>>0;}
  transform(docs){
    const input=Array.isArray(docs)?docs:[docs];
    return input.map(d=>{const row=Array(this.features).fill(0);for(const t of this.tokenizer.tokenize(d)){const h=this._hash(t),i=h%this.features,sign=this.signed&&((h>>>31)&1)?-1:1;row[i]+=sign;}return row;});
  }
}

export class BM25Index {
  constructor({k1=1.5,b=0.75,tokenizerOptions={removeStopWords:true}}={}){
    this.k1=k1;this.b=b;this.tokenizer=new Tokenizer(tokenizerOptions);
  }
  fit(docs,metadata=null){
    if(!Array.isArray(docs)||!docs.length) throw new Error('BM25Index requires documents');
    this.docs=[...docs];this.metadata=metadata?metadata.map(x=>x):docs.map((_,i)=>({id:i}));
    this.tokens=docs.map(d=>this.tokenizer.tokenize(d));
    this.lengths=this.tokens.map(t=>t.length);this.avgdl=this.lengths.reduce((a,b)=>a+b,0)/docs.length;
    this.df=new Map();this.tfs=[];
    for(const toks of this.tokens){const tf=new Map();for(const t of toks)tf.set(t,(tf.get(t)||0)+1);this.tfs.push(tf);for(const t of new Set(toks))this.df.set(t,(this.df.get(t)||0)+1);}
    return this;
  }
  _idf(term){const n=this.docs.length,df=this.df.get(term)||0;return Math.log(1+(n-df+0.5)/(df+0.5));}
  score(query,index){
    if(!this.docs) throw new Error('BM25Index is not fitted');
    const q=this.tokenizer.tokenize(query),tf=this.tfs[index],dl=this.lengths[index];let score=0;
    for(const term of q){const f=tf.get(term)||0;if(!f)continue;const denom=f+this.k1*(1-this.b+this.b*dl/(this.avgdl||1));score+=this._idf(term)*(f*(this.k1+1))/denom;}
    return score;
  }
  search(query,{topK=5,minScore=0}={}){
    if(!this.docs) throw new Error('BM25Index is not fitted');
    return this.docs.map((doc,i)=>({index:i,score:this.score(query,i),document:doc,metadata:this.metadata[i]}))
      .filter(r=>r.score>=minScore).sort((a,b)=>b.score-a.score).slice(0,topK);
  }
}

export class TextSimilarityIndex {
  constructor({tokenizerOptions={removeStopWords:true}}={}){this.tokenizerOptions=tokenizerOptions;}
  fit(docs,metadata=null){
    if(!Array.isArray(docs)||!docs.length) throw new Error('TextSimilarityIndex requires documents');
    this.docs=[...docs];this.metadata=metadata?metadata.map(x=>x):docs.map((_,i)=>({id:i}));
    this.vectorizer=new TFIDF();this.matrix=this.vectorizer.fitTransform(docs);return this;
  }
  search(query,{topK=5,minScore=-Infinity}={}){
    if(!this.matrix) throw new Error('TextSimilarityIndex is not fitted');
    const q=this.vectorizer.transform(query);
    return this.matrix.map((v,i)=>({index:i,score:MathX.cosine(v,q),document:this.docs[i],metadata:this.metadata[i]}))
      .filter(r=>r.score>=minScore).sort((a,b)=>b.score-a.score).slice(0,topK);
  }
}

export class NearestNeighborsIndex {
  constructor({metric='euclidean'}={}){if(!['euclidean','manhattan','cosine'].includes(metric))throw new Error('Unsupported metric');this.metric=metric;}
  fit(X,metadata=null){Validation.matrix(X);this.X=X.map(r=>[...r]);this.metadata=metadata?metadata.map(x=>x):X.map((_,i)=>({id:i}));return this;}
  _distance(a,b){if(this.metric==='manhattan')return MathX.manhattan(a,b);if(this.metric==='cosine')return 1-MathX.cosine(a,b);return MathX.euclidean(a,b);}
  search(row,{topK=5,maxDistance=Infinity}={}){if(!this.X)throw new Error('NearestNeighborsIndex is not fitted');if(!Array.isArray(row)||row.length!==this.X[0].length)throw new Error('Query feature count mismatch');return this.X.map((r,i)=>({index:i,distance:this._distance(r,row),metadata:this.metadata[i],vector:r})).filter(x=>x.distance<=maxDistance).sort((a,b)=>a.distance-b.distance).slice(0,topK);}
}

export function reciprocalRankFusion(resultSets,{k=60,topK=10,key=r=>r.index}={}){
  const scores=new Map(),items=new Map();
  for(const set of resultSets){set.forEach((r,rank)=>{const id=key(r);scores.set(id,(scores.get(id)||0)+1/(k+rank+1));if(!items.has(id))items.set(id,r);});}
  return [...scores.entries()].map(([id,score])=>({...items.get(id),fusionScore:score})).sort((a,b)=>b.fusionScore-a.fusionScore).slice(0,topK);
}
