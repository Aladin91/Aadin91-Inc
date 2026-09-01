import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
import { sha256Hex, verifySha256, VerifiedManifestLoader } from '../src/cloud-integrity.js';

const h=await sha256Hex('abc');assert.equal(h,'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
assert.equal((await verifySha256('abc',h)).valid,true);
assert.equal((await verifySha256('abd',h)).valid,false);
const loader=new VerifiedManifestLoader({baseUrl:'https://example.com/assets',manifest:{assets:{model:{path:'model.bin',sha256:h}}}});assert.equal(loader._entry('model').path,'model.bin');
assert.throws(()=>loader._entry('missing'),/Manifest asset not found/);
console.log('AladinAI cloud integrity tests passed.');
