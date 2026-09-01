# AladinAI.js

AladinAI is an offline-first machine-learning, NLP and decision engine written in plain JavaScript for reuse across browser applications.

It is designed for local intelligence without depending on OpenAI, Google, Anthropic, Hugging Face, cloud inference, or an external AI API. Classical ML is not an LLM; AladinAI focuses on deterministic and explainable capabilities that are useful inside standalone applications.

## v0.2.0 capabilities

### NLP
- Tokenization
- Sentence splitting
- N-grams
- TF-IDF vectorization
- Multinomial Naive Bayes text classification

### Similarity and math
- Cosine similarity
- Euclidean distance
- Manhattan distance
- Jaccard similarity
- Mean, variance and standard deviation
- Seeded reproducible random generator

### Preprocessing
- StandardScaler
- MinMaxScaler
- LabelEncoder
- OneHotEncoder
- Train/test split
- Pipeline abstraction
- Input shape and finite-number validation in advanced models

### Classification
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Logistic Regression (binary)
- Random Forest with bootstrap sampling and feature subsampling
- Majority-vote Ensemble

### Regression
- Linear Regression with gradient descent and optional L2 regularization
- R² scoring

### Clustering
- K-Means
- K-Means++ with deterministic seeded initialization and inertia
- DBSCAN

### Anomaly detection
- Z-score detector
- IQR detector

### Model evaluation
- Accuracy
- Confusion Matrix
- Precision / Recall / F1
- K-fold index generation
- Cross-validation

### Decision support / explainability
- Rule Engine
- Decision Engine combining model votes, rule evidence and confidence
- Random Forest feature-importance summary
- Explicit model serialization envelope
- Local model storage helper in the core package

## Use as an ES module

For new applications this is the preferred interface:

```js
import AladinAI from './aladin-ai/src/index.js';

const forest = new AladinAI.RandomForestClassifier({
  trees: 100,
  maxDepth: 8,
  seed: 42
});

forest.fit(X, y);
console.log(forest.predict(newRows));
console.table(forest.explain(['cfm', 'voltage', 'hp']));
```

## Existing classic-browser build

The original core remains available as a classic script:

```html
<script src="aladin-ai/dist/aladin-ai.js"></script>
<script>
  const tokenizer = new AladinAI.Tokenizer({ removeStopWords: true });
  console.log(tokenizer.tokenize('The cooling tower requires 480V power.'));
</script>
```

The ES-module entry point contains the complete v0.2 API. A single-file v0.2 classic bundle is planned before the API is marked stable.

## Random Forest

```js
import { RandomForestClassifier } from './src/advanced.js';

const X = [
  [5000, 208, 5],
  [7500, 208, 7.5],
  [18500, 480, 15],
  [21000, 480, 20]
];
const y = ['small', 'small', 'large', 'large'];

const model = new RandomForestClassifier({
  trees: 75,
  maxDepth: 6,
  maxFeatures: 'sqrt',
  seed: 10
}).fit(X, y);

console.log(model.predict([[19000, 480, 15]]));
console.log(model.predictProba([[19000, 480, 15]]));
console.log(model.explain(['cfm','voltage','hp']));
```

## Linear Regression

```js
import { LinearRegression } from './src/advanced.js';

const model = new LinearRegression({
  learningRate: 0.05,
  epochs: 2500
}).fit([[0],[1],[2],[3]], [1,3,5,7]);

console.log(model.predict([[4]]));
console.log(model.score([[0],[1],[2],[3]], [1,3,5,7]));
```

## Logistic Regression

```js
const model = new AladinAI.LogisticRegression({
  learningRate: 0.1,
  epochs: 2000
}).fit([[-2],[-1],[1],[2]], ['low','low','high','high']);

console.log(model.predict([[1.5]]));
console.log(model.predictProba([[1.5]]));
```

## Pipeline

```js
const pipeline = new AladinAI.Pipeline([
  new AladinAI.StandardScaler(),
  new AladinAI.DecisionTreeClassifier({ maxDepth: 5 })
]);

pipeline.fit(X, y);
console.log(pipeline.predict(newRows));
```

## K-Means++

```js
const model = new AladinAI.KMeansPlusPlus({
  clusters: 3,
  seed: 42
}).fit(data);

console.log(model.labels);
console.log(model.centroids);
console.log(model.inertia);
```

## Anomaly detection

```js
const detector = new AladinAI.ZScoreAnomalyDetector({ threshold: 3 });
detector.fit(historicalData);

// 1 = normal, -1 = anomaly
console.log(detector.predict(newData));
```

## Cross-validation

```js
const scores = AladinAI.crossValidate(
  () => new AladinAI.DecisionTreeClassifier({ maxDepth: 5 }),
  X,
  y,
  { k: 5, seed: 42 }
);

console.log(scores);
```

## Decision Engine

The Decision Engine is intended to make applications less dependent on a single model opinion.

```js
const engine = new AladinAI.DecisionEngine({
  models: [tree, forest, knn],
  weights: [1, 2, 1],
  ruleEngine: rules
});

const result = engine.analyze(
  [18500, 480, 15],
  { projectType: 'healthcare', value: 18500 }
);

console.log(result);
```

Result shape:

```js
{
  decision: 'HIGH',
  confidence: 0.75,
  votes: [
    { model: 'DecisionTreeClassifier', prediction: 'HIGH', weight: 1 },
    { model: 'RandomForestClassifier', prediction: 'HIGH', weight: 2 },
    { model: 'KNNClassifier', prediction: 'MEDIUM', weight: 1 }
  ],
  rules: [],
  reasons: []
}
```

## Model serialization

```js
const payload = AladinAI.SerializableModel.dump(model);
const json = JSON.stringify(payload);

const restored = AladinAI.SerializableModel.restore(
  JSON.parse(json),
  AladinAI.createDefaultRegistry()
);
```

Serialization is versioned so future migrations can be handled explicitly instead of silently breaking saved models.

## Engineering principles

1. Offline-first: trained models and inference should be usable without an internet connection.
2. No silent assumptions: invalid numerical matrices throw explicit errors in advanced models.
3. Reproducibility: randomized algorithms expose a seed.
4. Explainability: expose votes, confidence, feature importance or rule evidence when practical.
5. Modular design: applications can import only the capabilities they need.
6. Human control: prediction confidence is not treated as truth.
7. Stable persistence: serialized models use an explicit format and version.

## Testing

```bash
cd aladin-ai
npm test
```

The repository includes both core smoke tests and an advanced test suite. GitHub Actions also runs the tests on relevant pushes and pull requests.

## Current limitations

- Logistic Regression is binary only.
- Random Forest feature importance currently measures split usage rather than permutation importance or total impurity reduction.
- Linear/Logistic Regression use gradient descent and generally benefit from scaled features.
- `SerializableModel` restores state for compatible classes, but complex nested models such as Random Forest require a dedicated nested serializer before long-term persistence should be considered stable.
- The original `dist/aladin-ai.js` is the v0.1 classic bundle; the complete v0.2 API is currently provided through `src/index.js` as an ES module.

These limitations are documented intentionally rather than hidden.

## Roadmap

### v0.2.x hardening
- Single-file v0.2 classic browser bundle
- Better Gaussian NB class priors
- Core-wide validation helpers
- Robust nested serialization for Random Forest and Pipeline
- Model metadata: feature names, training timestamp and schema fingerprint
- Probability calibration utilities

### v0.3
- Multiclass logistic regression / softmax
- Hierarchical clustering
- PCA
- Polynomial regression
- Online / incremental learning where mathematically appropriate
- Permutation feature importance
- Per-row Decision Tree explanations
- Weighted ensemble probabilities
- IndexedDB model registry
- Dataset schema validation

### v0.4
- Small feed-forward neural network
- Backpropagation
- Adam optimizer
- Web Workers
- Optional WebGPU numerical backend

### v1.0 target
- Stable browser API
- Standalone minified bundle
- Full unit/integration tests
- Documentation site
- Benchmark suite
- Stable versioned model format
- Performance and memory budgets

## Development philosophy

AladinAI should not pretend that decision trees, tokenization or clustering are equivalent to a large language model. The goal is different: provide a reusable, private, inspectable and trainable local intelligence layer that can be embedded in standalone applications and combined with domain-specific rules.
