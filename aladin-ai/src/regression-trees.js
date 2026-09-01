import { MathX } from './aladin-ai.js';
import { Validation, SeededRandom } from './advanced.js';

function mse(y){if(!y.length)return 0;const m=MathX.mean(y);return y.reduce((s,v)=>s+(v-m)**2,0)/y.length;}

export class DecisionTreeRegressor {
  constructor({maxDepth=8,minSamplesSplit=2,minSamplesLeaf=1,minGain=1e-12}={}){Object.assign(this,{maxDepth,minSamplesSplit,minSamplesLeaf,minGain});}
  fit(X,y){Validation.supervised(X,y);if(!y.every(Number.isFinite))throw new Error('y must contain finite numbers');this.nFeatures=X[0].length;this.featureUsage=Array(this.nFeatures).fill(0);this.root=this._build(X,y,0);const total=this.featureUsage.reduce((a,b)=>a+b,0)||1;this.featureImportances=this.featureUsage.map(v=>v/total);return this;}
  _build(X,y,depth){
    const value=MathX.mean(y);
    if(depth>=this.maxDepth||X.length<this.minSamplesSplit||X.length<2*this.minSamplesLeaf||mse(y)<=this.minGain)return{leaf:true,value,samples:X.length};
    const base=mse(y);let best=null;
    for(let f=0;f<X[0].length;f++){
      const vals=[...new Set(X.map(r=>r[f]))].sort((a,b)=>a-b);
      for(let i=0;i<vals.length-1;i++){
        const threshold=(vals[i]+vals[i+1])/2,left=[],right=[];
        X.forEach((r,idx)=>(r[f]<=threshold?left:right).push(idx));
        if(left.length<this.minSamplesLeaf||right.length<this.minSamplesLeaf)continue;
        const score=(left.length/X.length)*mse(left.map(i=>y[i]))+(right.length/X.length)*mse(right.map(i=>y[i]));
        const gain=base-score;if(!best||gain>best.gain)best={f,threshold,left,right,gain};
      }
    }
    if(!best||best.gain<=this.minGain)return{leaf:true,value,samples:X.length};
    this.featureUsage[best.f]+=best.gain*X.length;
    return{leaf:false,feature:best.f,threshold:best.threshold,value,samples:X.length,gain:best.gain,
      left:this._build(best.left.map(i=>X[i]),best.left.map(i=>y[i]),depth+1),
      right:this._build(best.right.map(i=>X[i]),best.right.map(i=>y[i]),depth+1)};
  }
  _predictOne(row,node=this.root){if(node.leaf)return node.value;return this._predictOne(row,row[node.feature]<=node.threshold?node.left:node.right);}
  predict(X){Validation.matrix(X);if(X[0].length!==this.nFeatures)throw new Error('Feature count mismatch');return X.map(r=>this._predictOne(r));}
  score(X,y){const p=this.predict(X),mean=MathX.mean(y),res=y.reduce((s,v,i)=>s+(v-p[i])**2,0),tot=y.reduce((s,v)=>s+(v-mean)**2,0);return 1-res/(tot||1);}
  explain(row,featureNames=[]){if(!this.root)throw new Error('Model is not fitted');const path=[];let node=this.root;while(!node.leaf){const name=featureNames[node.feature]??node.feature,goLeft=row[node.feature]<=node.threshold;path.push({feature:name,value:row[node.feature],threshold:node.threshold,direction:goLeft?'left':'right',gain:node.gain});node=goLeft?node.left:node.right;}return{prediction:node.value,path};}
}

export class RandomForestRegressor {
  constructor({trees=50,maxDepth=8,minSamplesSplit=2,minSamplesLeaf=1,maxFeatures='sqrt',sampleRate=1,seed=42}={}){Object.assign(this,{trees,maxDepth,minSamplesSplit,minSamplesLeaf,maxFeatures,sampleRate,seed});}
  _featureCount(p){if(this.maxFeatures==='sqrt')return Math.max(1,Math.floor(Math.sqrt(p)));if(this.maxFeatures==='log2')return Math.max(1,Math.floor(Math.log2(p)));if(typeof this.maxFeatures==='number'&&this.maxFeatures>0&&this.maxFeatures<=1)return Math.max(1,Math.floor(p*this.maxFeatures));if(Number.isInteger(this.maxFeatures))return Math.min(p,this.maxFeatures);return p;}
  fit(X,y){Validation.supervised(X,y);if(!y.every(Number.isFinite))throw new Error('y must contain finite numbers');const rng=new SeededRandom(this.seed),p=X[0].length,m=this._featureCount(p),n=Math.max(2,Math.round(X.length*this.sampleRate));this.nFeatures=p;this.models=[];this.featureImportances=Array(p).fill(0);
    for(let t=0;t<this.trees;t++){
      const features=rng.shuffle([...Array(p).keys()]).slice(0,m),sx=[],sy=[];
      for(let i=0;i<n;i++){const idx=rng.int(X.length);sx.push(features.map(f=>X[idx][f]));sy.push(y[idx]);}
      const tree=new DecisionTreeRegressor({maxDepth:this.maxDepth,minSamplesSplit:this.minSamplesSplit,minSamplesLeaf:this.minSamplesLeaf}).fit(sx,sy);
      this.models.push({tree,features});tree.featureImportances.forEach((imp,j)=>this.featureImportances[features[j]]+=imp);
    }
    const total=this.featureImportances.reduce((a,b)=>a+b,0)||1;this.featureImportances=this.featureImportances.map(v=>v/total);return this;
  }
  predict(X){Validation.matrix(X);if(X[0].length!==this.nFeatures)throw new Error('Feature count mismatch');const all=this.models.map(({tree,features})=>tree.predict(X.map(r=>features.map(f=>r[f]))));return X.map((_,i)=>MathX.mean(all.map(p=>p[i])));}
  score(X,y){const p=this.predict(X),mean=MathX.mean(y),res=y.reduce((s,v,i)=>s+(v-p[i])**2,0),tot=y.reduce((s,v)=>s+(v-mean)**2,0);return 1-res/(tot||1);}
  explain(featureNames=[]){return this.featureImportances.map((importance,i)=>({feature:featureNames[i]??i,importance})).sort((a,b)=>b.importance-a.importance);}
}
