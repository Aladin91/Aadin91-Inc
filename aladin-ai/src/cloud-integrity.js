export async function sha256Hex(data){
  if(!globalThis.crypto?.subtle)throw new Error('Web Crypto SHA-256 is not available');
  let bytes;if(typeof data==='string')bytes=new TextEncoder().encode(data);else if(data instanceof ArrayBuffer)bytes=new Uint8Array(data);else if(ArrayBuffer.isView(data))bytes=new Uint8Array(data.buffer,data.byteOffset,data.byteLength);else if(data instanceof Blob)bytes=new Uint8Array(await data.arrayBuffer());else throw new Error('Unsupported SHA-256 input');
  const digest=await crypto.subtle.digest('SHA-256',bytes);return [...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,'0')).join('');
}

export async function verifySha256(data,expected){const actual=await sha256Hex(data);const normalized=String(expected||'').toLowerCase().replace(/^sha256:/,'');return {valid:actual===normalized,actual,expected:normalized};}

export async function fetchVerifiedAsset(url,{sha256=null,signal=null,cache='default',as='arrayBuffer'}={}){
  const r=await fetch(url,{signal,cache});if(!r.ok)throw new Error(`Asset fetch failed (${r.status})`);const buffer=await r.arrayBuffer();if(sha256){const check=await verifySha256(buffer,sha256);if(!check.valid)throw new Error(`Asset SHA-256 mismatch: expected ${check.expected}, got ${check.actual}`);}
  if(as==='arrayBuffer')return buffer;if(as==='blob')return new Blob([buffer],{type:r.headers.get('content-type')||'application/octet-stream'});if(as==='text')return new TextDecoder().decode(buffer);if(as==='json')return JSON.parse(new TextDecoder().decode(buffer));throw new Error(`Unsupported asset output type: ${as}`);
}

export class VerifiedManifestLoader {
  constructor({baseUrl='',manifest=null}={}){this.baseUrl=baseUrl.replace(/\/$/,'');this.manifest=manifest;}
  setManifest(manifest){this.manifest=manifest;return this;}
  _entry(name){const e=this.manifest?.assets?.[name];if(!e)throw new Error(`Manifest asset not found: ${name}`);return typeof e==='string'?{path:e}:e;}
  async load(name,{as='arrayBuffer',signal=null}={}){const e=this._entry(name),url=/^https?:\/\//.test(e.path)?e.path:`${this.baseUrl}/${String(e.path).replace(/^\//,'')}`;return fetchVerifiedAsset(url,{sha256:e.sha256||null,as,signal});}
}
