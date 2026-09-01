import { MathX } from './aladin-ai.js';
import { Validation, SeededRandom } from './advanced.js';

function zeros(rows,cols){return Array.from({length:rows},()=>Array(cols).fill(0));}
function argmax(a){let b=0;for(let i=1;i<a.length;i++)if(a[i]>a[b])b=i;return b;}

export class NeuralNetworkClassifier {
  constructor({hiddenLayers=[16],activation='relu',learningRate=.01,epochs=500,l2=0,seed=42}={}){
    if(!['relu','tanh','sigmoid'].includes(activation))throw new Error('activation must be relu, tanh, or sigmoid');
    Object.assign(this,{hiddenLayers,activation,learningRate,epochs,l2,seed});
  }
  _act(x){if(this.activation==='relu')return Math.max(0,x);if(this.activation==='tanh')return Math.tanh(x);return 1/(1+Math.exp(-x));}
  _dact(x){if(this.activation==='relu')return x>0?1:0;if(this.activation==='tanh'){const t=Math.tanh(x);return 1-t*t;}const s=1/(1+Math.exp(-x));return s*(1-s);}
  _init(sizes){const rng=new SeededRandom(this.seed);this.weights=[];this.biases=[];for(let l=0;l<sizes.length-1;l++){const fanIn=sizes[l],fanOut=sizes[l+1],limit=Math.sqrt(6/(fanIn+fanOut));this.weights.push(Array.from({length:fanIn},()=>Array.from({length:fanOut},()=>((rng.next()*2)-1)*limit)));this.biases.push(Array(fanOut).fill(0));}}
  _forward(row){const activations=[row],zs=[];let a=row;for(let l=0;l<this.weights.length;l++){const z=this.biases[l].map((b,j)=>b+a.reduce((s,v,i)=>s+v*this.weights[l][i][j],0));zs.push(z);a=l===this.weights.length-1?MathX.softmax(z):z.map(v=>this._act(v));activations.push(a);}return{activations,zs};}
  fit(X,y){Validation.supervised(X,y);this.classes=[...new Set(y)];if(this.classes.length<2)throw new Error('NeuralNetworkClassifier requires at least two classes');const sizes=[X[0].length,...this.hiddenLayers,this.classes.length];this._init(sizes);this.lossHistory=[];const n=X.length;
    for(let epoch=0;epoch<this.epochs;epoch++){
      const gradW=this.weights.map(w=>zeros(w.length,w[0].length)),gradB=this.biases.map(b=>Array(b.length).fill(0));let loss=0;
      for(let r=0;r<n;r++){
        const target=this.classes.indexOf(y[r]),{activations,zs}=this._forward(X[r]);const probs=activations.at(-1);loss-=Math.log(probs[target]+1e-12);let delta=probs.map((p,j)=>p-(j===target?1:0));
        for(let l=this.weights.length-1;l>=0;l--){const prev=activations[l];for(let i=0;i<prev.length;i++)for(let j=0;j<delta.length;j++)gradW[l][i][j]+=prev[i]*delta[j];for(let j=0;j<delta.length;j++)gradB[l][j]+=delta[j];if(l>0){const next=Array(this.weights[l].length).fill(0);for(let i=0;i<next.length;i++){let s=0;for(let j=0;j<delta.length;j++)s+=this.weights[l][i][j]*delta[j];next[i]=s*this._dact(zs[l-1][i]);}delta=next;}}
      }
      for(let l=0;l<this.weights.length;l++){for(let i=0;i<this.weights[l].length;i++)for(let j=0;j<this.weights[l][i].length;j++){const g=gradW[l][i][j]/n+this.l2*this.weights[l][i][j];this.weights[l][i][j]-=this.learningRate*g;}for(let j=0;j<this.biases[l].length;j++)this.biases[l][j]-=this.learningRate*gradB[l][j]/n;}
      if(epoch%Math.max(1,Math.floor(this.epochs/50))===0||epoch===this.epochs-1)this.lossHistory.push({epoch,loss:loss/n});
    }return this;
  }
  predictProba(X){Validation.matrix(X);if(!this.weights?.length)throw new Error('NeuralNetworkClassifier is not fitted');return X.map(r=>{const p=this._forward(r).activations.at(-1);return Object.fromEntries(this.classes.map((c,i)=>[c,p[i]]));});}
  predict(X){return this.predictProba(X).map(p=>this.classes[argmax(this.classes.map(c=>p[c]))]);}
}

export class NeuralNetworkRegressor {
  constructor({hiddenLayers=[16],activation='relu',learningRate=.01,epochs=500,l2=0,seed=42}={}){if(!['relu','tanh','sigmoid'].includes(activation))throw new Error('activation must be relu, tanh, or sigmoid');Object.assign(this,{hiddenLayers,activation,learningRate,epochs,l2,seed});}
  _act(x){if(this.activation==='relu')return Math.max(0,x);if(this.activation==='tanh')return Math.tanh(x);return 1/(1+Math.exp(-x));}
  _dact(x){if(this.activation==='relu')return x>0?1:0;if(this.activation==='tanh'){const t=Math.tanh(x);return 1-t*t;}const s=1/(1+Math.exp(-x));return s*(1-s);}
  _init(sizes){const rng=new SeededRandom(this.seed);this.weights=[];this.biases=[];for(let l=0;l<sizes.length-1;l++){const fanIn=sizes[l],fanOut=sizes[l+1],limit=Math.sqrt(6/(fanIn+fanOut));this.weights.push(Array.from({length:fanIn},()=>Array.from({length:fanOut},()=>((rng.next()*2)-1)*limit)));this.biases.push(Array(fanOut).fill(0));}}
  _forward(row){const activations=[row],zs=[];let a=row;for(let l=0;l<this.weights.length;l++){const z=this.biases[l].map((b,j)=>b+a.reduce((s,v,i)=>s+v*this.weights[l][i][j],0));zs.push(z);a=l===this.weights.length-1?z:z.map(v=>this._act(v));activations.push(a);}return{activations,zs};}
  fit(X,y){Validation.supervised(X,y);if(!y.every(v=>typeof v==='number'&&Number.isFinite(v)))throw new Error('y must contain finite numbers');this._init([X[0].length,...this.hiddenLayers,1]);this.lossHistory=[];const n=X.length;for(let epoch=0;epoch<this.epochs;epoch++){const gradW=this.weights.map(w=>zeros(w.length,w[0].length)),gradB=this.biases.map(b=>Array(b.length).fill(0));let loss=0;for(let r=0;r<n;r++){const f=this._forward(X[r]),pred=f.activations.at(-1)[0],err=pred-y[r];loss+=err*err;let delta=[2*err];for(let l=this.weights.length-1;l>=0;l--){const prev=f.activations[l];for(let i=0;i<prev.length;i++)for(let j=0;j<delta.length;j++)gradW[l][i][j]+=prev[i]*delta[j];for(let j=0;j<delta.length;j++)gradB[l][j]+=delta[j];if(l>0){const next=Array(this.weights[l].length).fill(0);for(let i=0;i<next.length;i++){let s=0;for(let j=0;j<delta.length;j++)s+=this.weights[l][i][j]*delta[j];next[i]=s*this._dact(f.zs[l-1][i]);}delta=next;}}}for(let l=0;l<this.weights.length;l++){for(let i=0;i<this.weights[l].length;i++)for(let j=0;j<this.weights[l][i].length;j++)this.weights[l][i][j]-=this.learningRate*(gradW[l][i][j]/n+this.l2*this.weights[l][i][j]);for(let j=0;j<this.biases[l].length;j++)this.biases[l][j]-=this.learningRate*gradB[l][j]/n;}if(epoch%Math.max(1,Math.floor(this.epochs/50))===0||epoch===this.epochs-1)this.lossHistory.push({epoch,loss:loss/n});}return this;}
  predict(X){Validation.matrix(X);return X.map(r=>this._forward(r).activations.at(-1)[0]);}
}
