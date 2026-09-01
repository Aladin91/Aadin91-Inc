import assert from 'node:assert/strict';
import { LocalVisionModelAdapter, TesseractOCRAdapter, ImageEmbeddingIndex } from '../src/vision-adapters.js';

const model=new LocalVisionModelAdapter({predictor:async x=>({value:x*2})});
assert.deepEqual(await model.predict(3),{value:6});
assert.deepEqual(await model.predictBatch([1,2,3],{concurrency:2}),[{value:2},{value:4},{value:6}]);

const fakeOCR={recognize:async()=>({data:{text:'  local OCR text  ',confidence:97}})};
const ocr=new TesseractOCRAdapter(fakeOCR,{language:'eng'});
const result=await ocr.recognize({});assert.equal(result.text,'local OCR text');assert.equal(result.confidence,97);

const embed=async image=>image.vector;
const index=new ImageEmbeddingIndex(embed);await index.fit([{vector:[1,0]},{vector:[0,1]}],['horizontal','vertical']);
const found=await index.search({vector:[.9,.1]},{topK:1});assert.equal(found[0].metadata,'horizontal');assert.ok(found[0].score>.9);
console.log('AladinAI local vision adapter tests passed.');
