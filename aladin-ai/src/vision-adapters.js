import { MathX } from './aladin-ai.js';

export class LocalVisionModelAdapter {
  constructor({predictor,preprocess=null,name='local-vision-model'}={}){
    if(typeof predictor!=='function')throw new Error('predictor must be a function');Object.assign(this,{predictor,preprocess,name});
  }
  async predict(input,options={}){const prepared=this.preprocess?await this.preprocess(input,options):input;return this.predictor(prepared,options);}
  async predictBatch(inputs,{concurrency=1,...options}={}){
    if(!Array.isArray(inputs))throw new Error('inputs must be an array');if(!Number.isInteger(concurrency)||concurrency<1)throw new Error('concurrency must be >= 1');
    const out=Array(inputs.length);let next=0;const worker=async()=>{while(true){const i=next++;if(i>=inputs.length)return;out[i]=await this.predict(inputs[i],options);}};await Promise.all(Array.from({length:Math.min(concurrency,inputs.length)},worker));return out;
  }
}

export class TesseractOCRAdapter {
  constructor(engine,{language='eng',logger=null}={}){if(!engine||typeof engine.recognize!=='function')throw new Error('A Tesseract-compatible engine with recognize() is required');Object.assign(this,{engine,language,logger});}
  async recognize(source,{language=this.language,options={}}={}){
    const result=await this.engine.recognize(source,language,{...options,...(this.logger?{logger:this.logger}:{})});const data=result?.data??result??{};return {text:String(data.text??'').trim(),confidence:Number.isFinite(data.confidence)?data.confidence:null,data};
  }
}

export class ImageEmbeddingIndex {
  constructor(embedder){if(typeof embedder!=='function')throw new Error('embedder must be an async or sync function');this.embedder=embedder;}
  async fit(images,metadata=null){if(!Array.isArray(images)||!images.length)throw new Error('images must be a non-empty array');if(metadata&&metadata.length!==images.length)throw new Error('metadata length must match images');this.vectors=[];for(const image of images){const v=await this.embedder(image);if(!Array.isArray(v)&&!ArrayBuffer.isView(v))throw new Error('embedder must return a numeric vector');this.vectors.push(Array.from(v));}const n=this.vectors[0].length;if(!n||this.vectors.some(v=>v.length!==n||v.some(x=>typeof x!=='number'||!Number.isFinite(x))))throw new Error('embedding vectors must have equal finite numeric dimensions');this.metadata=metadata||images.map((_,i)=>({index:i}));return this;}
  async search(image,{topK=5}={}){if(!this.vectors)throw new Error('ImageEmbeddingIndex is not fitted');const q=Array.from(await this.embedder(image));if(q.length!==this.vectors[0].length)throw new Error('query embedding dimension mismatch');return this.vectors.map((v,i)=>({index:i,score:MathX.cosine(q,v),metadata:this.metadata[i]})).sort((a,b)=>b.score-a.score).slice(0,topK);}
}

export function detectLocalVisionRuntimes(){
  const g=globalThis;return {tesseract:!!g.Tesseract,onnxRuntime:!!g.ort,webGPU:typeof navigator!=='undefined'&&!!navigator.gpu,webNN:typeof navigator!=='undefined'&&('ml'in navigator)};
}
