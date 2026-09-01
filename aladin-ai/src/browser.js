import { SerializableModel } from './advanced.js';

export const BROWSER_ENGINE_VERSION = '0.4.0';

export function schemaFingerprint(schema){
  const text=JSON.stringify(schema);
  let hash=2166136261;
  for(let i=0;i<text.length;i++){hash^=text.charCodeAt(i);hash=Math.imul(hash,16777619);}
  return `fnv1a-${(hash>>>0).toString(16).padStart(8,'0')}`;
}

export class ModelPackage {
  static create(model,{name='',featureNames=[],targetName='',trainingRows=null,notes='',tags=[]}={}){
    return {
      format:'aladin-ai-package',
      version:1,
      metadata:{name,modelType:model.constructor.name,engineVersion:BROWSER_ENGINE_VERSION,featureNames,targetName,trainingRows,notes,tags,createdAt:new Date().toISOString(),schemaFingerprint:schemaFingerprint({featureNames,targetName})},
      model:SerializableModel.dump(model)
    };
  }
  static restore(pkg,registry){if(!pkg||pkg.format!=='aladin-ai-package')throw new Error('Invalid AladinAI package');return SerializableModel.restore(pkg.model,registry);}
  static validateSchema(pkg,{featureNames=[],targetName=''}={}){
    if(!pkg?.metadata?.schemaFingerprint)return {valid:false,reason:'Package has no schema fingerprint'};
    const actual=schemaFingerprint({featureNames,targetName});
    return {valid:actual===pkg.metadata.schemaFingerprint,expected:pkg.metadata.schemaFingerprint,actual};
  }
}

export class IndexedDBModelStore {
  constructor({dbName='AladinAI',storeName='models',version=1}={}){Object.assign(this,{dbName,storeName,version});}
  _open(){if(typeof indexedDB==='undefined')return Promise.reject(new Error('IndexedDB is not available in this environment'));return new Promise((resolve,reject)=>{const req=indexedDB.open(this.dbName,this.version);req.onupgradeneeded=()=>{const db=req.result;if(!db.objectStoreNames.contains(this.storeName))db.createObjectStore(this.storeName,{keyPath:'name'});};req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});}
  async put(name,payload){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readwrite');tx.objectStore(this.storeName).put({name,payload,updatedAt:new Date().toISOString()});tx.oncomplete=()=>{db.close();resolve(name);};tx.onerror=()=>{db.close();reject(tx.error);};});}
  async get(name){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readonly'),req=tx.objectStore(this.storeName).get(name);req.onsuccess=()=>{db.close();resolve(req.result?.payload??null);};req.onerror=()=>{db.close();reject(req.error);};});}
  async remove(name){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readwrite');tx.objectStore(this.storeName).delete(name);tx.oncomplete=()=>{db.close();resolve();};tx.onerror=()=>{db.close();reject(tx.error);};});}
  async list(){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readonly'),req=tx.objectStore(this.storeName).getAll();req.onsuccess=()=>{db.close();resolve(req.result.map(({name,updatedAt})=>({name,updatedAt})));};req.onerror=()=>{db.close();reject(req.error);};});}
  async saveModel(name,model,meta={}){return this.put(name,ModelPackage.create(model,{name,...meta}));}
  async loadModel(name,registry,{featureNames=null,targetName=''}={}){const pkg=await this.get(name);if(!pkg)return null;if(featureNames){const check=ModelPackage.validateSchema(pkg,{featureNames,targetName});if(!check.valid)throw new Error(`Model schema mismatch: expected ${check.expected}, got ${check.actual}`);}return ModelPackage.restore(pkg,registry);}
}

export async function processInBatches(items,processor,{batchSize=100,onProgress=null,signal=null,yieldMs=0}={}){
  if(!Number.isInteger(batchSize)||batchSize<1)throw new Error('batchSize must be an integer >= 1');
  const out=[];for(let start=0;start<items.length;start+=batchSize){if(signal?.aborted)throw new DOMException('Operation aborted','AbortError');const batch=items.slice(start,start+batchSize);for(let i=0;i<batch.length;i++)out.push(await processor(batch[i],start+i));onProgress?.({processed:Math.min(start+batch.length,items.length),total:items.length,progress:Math.min(start+batch.length,items.length)/Math.max(items.length,1)});await new Promise(resolve=>setTimeout(resolve,yieldMs));}return out;
}

export function getRuntimeCapabilities(){return {webWorker:typeof Worker!=='undefined',webGPU:typeof navigator!=='undefined'&&'gpu'in navigator,indexedDB:typeof indexedDB!=='undefined',hardwareConcurrency:typeof navigator!=='undefined'?navigator.hardwareConcurrency??null:null,deviceMemory:typeof navigator!=='undefined'?navigator.deviceMemory??null:null};}
