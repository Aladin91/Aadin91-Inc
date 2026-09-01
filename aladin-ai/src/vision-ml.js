import { MathX } from './aladin-ai.js';
import { imageDescriptor } from './vision.js';

export class ImageFeatureExtractor {
  constructor(options={}){this.options=options;}
  transform(images){if(!Array.isArray(images))throw new Error('images must be an array');return images.map(img=>imageDescriptor(img,this.options).vector);}
  describe(images){if(!Array.isArray(images))throw new Error('images must be an array');return images.map(img=>imageDescriptor(img,this.options));}
}

export class ImageSimilarityIndex {
  constructor(options={}){this.extractor=new ImageFeatureExtractor(options);}
  fit(images,metadata=null){if(!Array.isArray(images)||!images.length)throw new Error('images must be a non-empty array');if(metadata&&metadata.length!==images.length)throw new Error('metadata length must match images');this.descriptors=this.extractor.describe(images);this.metadata=metadata||images.map((_,i)=>({index:i}));return this;}
  search(image,{topK=5,weightHash=.25}={}){if(!this.descriptors)throw new Error('ImageSimilarityIndex is not fitted');const q=imageDescriptor(image,this.extractor.options);return this.descriptors.map((d,i)=>{const cosine=MathX.cosine(q.vector,d.vector);let hd=0;for(let j=0;j<q.hash.length;j++)if(q.hash[j]!==d.hash[j])hd++;const hashSimilarity=1-hd/q.hash.length;return {index:i,score:(1-weightHash)*cosine+weightHash*hashSimilarity,cosine,hashSimilarity,metadata:this.metadata[i]};}).sort((a,b)=>b.score-a.score).slice(0,topK);}
}

export class ImageClassifier {
  constructor(model,{descriptorOptions={}}={}){if(!model||typeof model.fit!=='function'||typeof model.predict!=='function')throw new Error('model must implement fit and predict');this.model=model;this.extractor=new ImageFeatureExtractor(descriptorOptions);}
  fit(images,labels){if(!Array.isArray(labels)||labels.length!==images.length)throw new Error('labels length must match images');this.model.fit(this.extractor.transform(images),labels);return this;}
  predict(images){return this.model.predict(this.extractor.transform(images));}
  predictProba(images){if(typeof this.model.predictProba!=='function')throw new Error('underlying model does not support predictProba');return this.model.predictProba(this.extractor.transform(images));}
}

export class ImageRegressor {
  constructor(model,{descriptorOptions={}}={}){if(!model||typeof model.fit!=='function'||typeof model.predict!=='function')throw new Error('model must implement fit and predict');this.model=model;this.extractor=new ImageFeatureExtractor(descriptorOptions);}
  fit(images,targets){if(!Array.isArray(targets)||targets.length!==images.length)throw new Error('targets length must match images');this.model.fit(this.extractor.transform(images),targets);return this;}
  predict(images){return this.model.predict(this.extractor.transform(images));}
}
