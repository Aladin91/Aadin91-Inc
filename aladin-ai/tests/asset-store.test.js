import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
import { sha256Hex } from '../src/cloud-integrity.js';
import { MemoryAssetStore, VerifiedAssetCache } from '../src/asset-store.js';

const store=new MemoryAssetStore(),cache=new VerifiedAssetCache(store),data=new TextEncoder().encode('model-bytes'),hash=await sha256Hex(data);
await cache.put('model.bin',data,{sha256:hash,url:'https://example.com/model.bin'});
const item=await cache.get('model.bin',{sha256:hash});assert.ok(item);assert.equal(new TextDecoder().decode(item.data),'model-bytes');
assert.equal((await cache.list()).length,1);
await cache.remove('model.bin');assert.equal(await cache.get('model.bin'),null);
await cache.put('x',data);await cache.clear();assert.equal((await cache.list()).length,0);
console.log('AladinAI asset store tests passed.');
