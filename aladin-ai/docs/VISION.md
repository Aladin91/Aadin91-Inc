# AladinAI Vision

AladinAI v0.4 adds a local computer-vision layer written in plain JavaScript. It is designed for browser-first personal applications that need deterministic image analysis without uploading images to a cloud AI service.

## Current capabilities

### Image preprocessing
- RGBA to grayscale conversion
- nearest-neighbor resizing
- normalized intensity histograms
- arbitrary 2D convolution
- blur, Gaussian blur, sharpening and Laplacian kernels
- Sobel edge maps
- Otsu automatic threshold selection
- binary thresholding

### Shape / region analysis
- connected-component detection
- component bounding boxes, areas and centroids
- binary dilation / erosion
- morphological opening / closing
- image moments and centroids

### Visual descriptors
- average hash (aHash)
- difference hash (dHash)
- Hamming distance
- perceptual hash similarity
- combined intensity/edge descriptors
- Histogram of Oriented Gradients (HOG)
- edge density
- global structural similarity (SSIM)

### Matching / segmentation
- normalized template matching
- RGB K-Means segmentation
- descriptor comparison
- image similarity index with top-K retrieval

### Machine-learning adapters
Image descriptors can feed the existing AladinAI classifiers/regressors:

```js
const model = new AladinAI.RandomForestClassifier({trees:100, seed:42});
const vision = new AladinAI.ImageClassifier(model, {
  descriptorOptions: { histogramBins:32, edgeBins:16 }
});

vision.fit(trainingImages, labels);
const predictions = vision.predict(newImages);
```

Image similarity:

```js
const index = new AladinAI.ImageSimilarityIndex()
  .fit(images, metadata);

const nearest = index.search(queryImage, {topK:5});
```

HOG features can be used directly with any numeric model:

```js
const features = AladinAI.hogDescriptor(image, {
  width:64,
  height:64,
  cellSize:8,
  bins:9
}).vector;
```

## Browser file analysis

```js
const file = input.files[0];
const image = await AladinAI.imageFromSource(file, {
  maxWidth:1200,
  maxHeight:1200
});

const descriptor = AladinAI.imageDescriptor(image);
const edges = AladinAI.sobelEdges(image);
const binary = AladinAI.threshold(image);
const regions = AladinAI.connectedComponents(binary, {minArea:25});
```

The selected file is decoded and analyzed locally in the browser.

## What this vision layer is — and is not

The current implementation is classical/local computer vision. It can inspect pixels, edges, patterns, regions, similarity and learned classifications when you provide labeled training examples.

It does not claim to provide general semantic visual understanding equivalent to a large multimodal model. For example, it will not automatically know that an arbitrary object is a pump, dog or vehicle unless you build/train a classifier or attach an optional local vision model.

## Future local-only adapters

AladinAI is intentionally structured so additional local inference backends can be added without changing application APIs. Candidate optional runtimes include:

- locally bundled Tesseract/OCR
- ONNX Runtime Web
- WebGPU-hosted vision models
- WebNN models
- locally cached embedding models

Those model/runtime files can be distributed through GitHub Releases and cached locally after download.

## Privacy recommendation

For personal apps:

- keep private images in browser memory / IndexedDB
- use GitHub only for public/versioned reference images or model files
- do not upload a user's image unless the application explicitly requires that workflow
- do not embed GitHub write tokens in frontend code
