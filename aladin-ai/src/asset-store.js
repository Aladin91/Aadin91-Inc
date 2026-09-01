import { fetchVerifiedAsset, verifySha256 } from './cloud-integrity.js';

export class MemoryAssetStore {
  constructor(){this.items=new Map();}
  async put(name,data,meta={}){this.items.set(name,{data,meta:{...meta,updatedAt:new Date().toISOString()}});return name;}
  async get(name){return this.items.get(name)??null;}
  async remove(name){this.items.delete(name);}
  async list(){return [...this.items.entries()].map(([name,v])=>({name,meta:v.meta}));}
  async clear(){this.items.clear();}
}

export class IndexedDBAssetStore {
  constructor({dbName='AladinAIAssets',storeName='assets',version=1}={}){Object.assign(this,{dbName,storeName,version});}
  _open(){if(typeof indexedDB==='undefined')return Promise.reject(new Error('IndexedDB is not available'));return new Promise((resolve,reject)=>{const req=indexedDB.open(this.dbName,this.version);req.onupgradeneeded=()=>{const db=req.result;if(!db.objectStoreNames.contains(this.storeName))db.createObjectStore(this.storeName,{keyPath:'name'});};req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});}
  async put(name,data,meta={}){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readwrite');tx.objectStore(this.storeName).put({name,data,meta:{...meta,updatedAt:new Date().toISOString()}});tx.oncomplete=()=>{db.close();resolve(name);};tx.onerror=()=>{db.close();reject(tx.error);};});}
  async get(name){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readonly'),req=tx.objectStore(this.storeName).get(name);req.onsuccess=()=>{db.close();resolve(req.result??null);};req.onerror=()=>{db.close();reject(req.error);};});}
  async remove(name){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readwrite');tx.objectStore(this.storeName).delete(name);tx.oncomplete=()=>{db.close();resolve();};tx.onerror=()=>{db.close();reject(tx.error);};});}
  async list(){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readonly'),req=tx.objectStore(this.storeName).getAll();req.onsuccess=()=>{db.close();resolve(req.result.map(({name,meta})=>({name,meta})));};req.onerror=()=>{db.close();reject(req.error);};});}
  async clear(){const db=await this._open();return new Promise((resolve,reject)=>{const tx=db.transaction(this.storeName,'readwrite');tx.objectStore(this.storeName).clear();tx.oncomplete=()=>{db.close();resolve();};tx.onerror=()=>{db.close();reject(tx.error);};});}
}

export class VerifiedAssetCache {
  constructor(store=new MemoryAssetStore()){this.store=store;}
  async put(name,data,{sha256=null,url=null,...meta}={}){if(sha256){const check=await verifySha256(data,sha256);if(!check.valid)throw new Error(`Asset SHA-256 mismatch: expected ${check.expected}, got ${check.actual}`);meta.sha256=check.actual;}return this.store.put(name,data,{...meta,url});}
  async get(name,{sha256=null}={}){const item=await this.store.get(name);if(!item)return null;if(sha256){const check=await verifySha256(item.data,sha256);if(!check.valid){await this.store.remove(name);return null;}}return item;}
  async fetch(name,url,{sha256=null,preferCache=true,signal=null}={}){
    if(preferCache){const cached=await this.get(name,{sha256});if(cached)return {...cached,source:'cache'};}
    const data=await fetchVerifiedAsset(url,{sha256,signal,as:'arrayBuffer'});await this.put(name,data,{sha256,url});return {data,meta:{sha256,url},source:'network'};
  }
  remove(name){return this.store.remove(name);}
  list(){return this.store.list();}
  clear(){return this.store.clear();}
}
