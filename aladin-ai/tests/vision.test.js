import assert from 'node:assert/strict';
import {
  toGrayscale,histogram,sobelEdges,otsuThreshold,threshold,connectedComponents,
  averageHash,differenceHash,hammingDistance,perceptualSimilarity,templateMatch,imageDescriptor,compareDescriptors
} from '../src/vision.js';

const img={width:4,height:4,data:Uint8ClampedArray.from([
  0,0,0,255, 0,0,0,255, 255,255,255,255, 255,255,255,255,
  0,0,0,255, 0,0,0,255, 255,255,255,255, 255,255,255,255,
  0,0,0,255, 0,0,0,255, 255,255,255,255, 255,255,255,255,
  0,0,0,255, 0,0,0,255, 255,255,255,255, 255,255,255,255
])};
const gray=toGrayscale(img);assert.equal(gray.data.length,16);assert.equal(gray.data[0],0);assert.ok(Math.abs(gray.data[3]-255)<1e-9);
const hist=histogram(img,{bins:2,normalize:false});assert.deepEqual(hist,[8,8]);
const edges=sobelEdges(img);assert.equal(edges.data.length,16);assert.ok(Math.max(...edges.data)>0);
const t=otsuThreshold(img);assert.ok(t>=0&&t<=255);
const bin=threshold(img,128);assert.equal([...bin.data].filter(Boolean).length,8);
const components=connectedComponents(bin,{minArea:1});assert.equal(components.length,1);assert.equal(components[0].area,8);
const ah=averageHash(img),dh=differenceHash(img);assert.equal(ah.length,64);assert.equal(dh.length,64);assert.equal(hammingDistance(ah,ah),0);assert.equal(perceptualSimilarity(img,img),1);
const tpl={width:2,height:2,data:Uint8Array.from([255,255,255,255])};
const match=templateMatch(gray,tpl);assert.ok(match.x>=0&&match.y>=0&&Number.isFinite(match.score));
const d1=imageDescriptor(img),d2=imageDescriptor(img);const cmp=compareDescriptors(d1,d2);assert.ok(cmp.cosine>.999);assert.equal(cmp.hashSimilarity,1);
console.log('AladinAI vision tests passed.');
