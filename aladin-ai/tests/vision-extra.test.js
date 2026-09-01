import assert from 'node:assert/strict';
import { threshold } from '../src/vision.js';
import { hogDescriptor, erode, dilate, openMorphology, closeMorphology, structuralSimilarity, edgeDensity, imageMoments } from '../src/vision-extra.js';

const img={width:8,height:8,data:Uint8Array.from(Array.from({length:64},(_,i)=>(i%8)<4?0:255))};
const hog=hogDescriptor(img,{width:16,height:16,cellSize:4,bins:9,blockSize:2});assert.ok(hog.vector.length>0);assert.ok(hog.vector.every(Number.isFinite));
const bin=threshold(img,128);const d=dilate(bin,{radius:1}),e=erode(bin,{radius:1});assert.equal(d.data.length,64);assert.equal(e.data.length,64);assert.equal(openMorphology(bin).data.length,64);assert.equal(closeMorphology(bin).data.length,64);
assert.ok(structuralSimilarity(img,img)>.9999);assert.ok(edgeDensity(img)>=0&&edgeDensity(img)<=1);
const moments=imageMoments(img,{binaryThreshold:128});assert.ok(moments.m00>0);assert.ok(moments.centroid.x>3);
console.log('AladinAI advanced vision tests passed.');
