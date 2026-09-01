# AladinAI.js

Offline-first machine-learning and NLP engine written in plain JavaScript for reuse across browser applications.

## Goals

- No cloud dependency
- No external AI API required
- Reusable from any HTML/JS application
- Explainable, deterministic building blocks where possible
- Browser-first, offline-friendly architecture
- Small enough to embed in standalone applications

## Browser usage

```html
<script src="aladin-ai/dist/aladin-ai.js"></script>
<script>
  const tokenizer = new AladinAI.Tokenizer({ removeStopWords: true });
  console.log(tokenizer.tokenize('The cooling tower requires 480V power.'));
</script>
```

## Included in v0.1.0

### NLP
- Tokenization
- Sentence splitting
- N-grams
- TF-IDF

### Similarity / math
- Cosine similarity
- Euclidean distance
- Manhattan distance
- Jaccard similarity
- Mean, variance and standard deviation

### Preprocessing
- StandardScaler
- MinMaxScaler
- Reproducible train/test split

### Classification
- Decision Tree classifier
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Multinomial Naive Bayes for text
- Ensemble majority-vote classifier

### Clustering
- K-Means
- DBSCAN

### Validation
- Accuracy
- Confusion matrix
- Precision / Recall / F1 for a selected class

### Decision support
- Rule engine
- Local model storage helper

## Example: Decision Tree

```js
const X = [
  [5000, 208, 5],
  [7500, 208, 7.5],
  [18500, 480, 15],
  [21000, 480, 20]
];
const y = ['small', 'small', 'large', 'large'];

const model = new AladinAI.DecisionTreeClassifier({ maxDepth: 5 });
model.fit(X, y);
console.log(model.predict([[19000, 480, 15]]));
```

## Example: Text classifier

```js
const model = new AladinAI.MultinomialNaiveBayesText();

model.train('power wiring voltage disconnect', 'electrical')
     .train('supply air duct cfm fan', 'hvac')
     .train('drain piping water condensate', 'plumbing')
     .fit();

console.log(model.predict('provide power wiring for disconnect'));
```

## Example: Clustering

```js
const model = new AladinAI.KMeans({ clusters: 2 });
model.fit([[1,1],[1.2,.9],[9,9],[9.2,8.8]]);
console.log(model.labels);
```

## Design principle

AladinAI is not intended to pretend that classical ML is an LLM. Its purpose is to provide reusable local intelligence: tokenization, classification, clustering, similarity, anomaly detection, rules, explainable decisions and later lightweight neural models.

## Planned roadmap

### v0.2
- Random Forest
- Linear and logistic regression
- K-Means++ initialization
- Hierarchical clustering
- IQR and Z-score anomaly detection
- LabelEncoder / OneHotEncoder
- Pipeline abstraction
- Model serialization with explicit `toJSON/fromJSON`
- Better probability/confidence APIs

### v0.3
- Incremental learning APIs (`partialFit`) where mathematically appropriate
- Cross-validation
- Feature importance
- Decision-tree explanations
- Weighted ensembles
- IndexedDB model registry

### v0.4
- Small feed-forward neural network
- Backpropagation
- Adam optimizer
- Web Workers
- Optional WebGPU acceleration

### v1.0 target
- Stable browser API
- Standalone minified build
- Full tests
- Documentation site
- Benchmark suite
- Versioned model format

## Development

```bash
cd aladin-ai
npm test
```

Open `examples/browser-demo.html` in a local web server to test the browser bundle.
