# Changelog

## 0.3.0

- Added CountVectorizer with configurable n-gram ranges.
- Added bounded-memory FeatureHasher.
- Added BM25 local document retrieval.
- Added TF-IDF cosine similarity index.
- Added numeric nearest-neighbor index.
- Added reciprocal-rank fusion for combining local search results.
- Unified public engine version at 0.3.0.
- Added model-package schema validation before restore.
- Added retrieval, versioning and model-package tests.
- Added a standalone retrieval demo.

## 0.2.0

- Added Random Forest classifier with probabilities and feature importance.
- Added Linear Regression and binary Logistic Regression.
- Added deterministic K-Means++.
- Added Z-score and IQR anomaly detection.
- Added LabelEncoder and OneHotEncoder.
- Added Pipeline, K-fold helpers and cross-validation.
- Added recursive serialization for model state, Maps and Sets.
- Added PCA, polynomial features, softmax regression and hierarchical clustering.
- Added feed-forward neural-network classifier and regressor.
- Added IndexedDB model persistence, model packages and runtime capability detection.
- Added build scripts, browser demos, expanded tests and GitHub Actions CI.

## 0.1.0

- Initial offline-first engine.
- Tokenization, sentence splitting, n-grams and TF-IDF.
- Decision Tree, KNN, Gaussian Naive Bayes and text Naive Bayes.
- K-Means and DBSCAN.
- StandardScaler and MinMaxScaler.
- Basic metrics, rule engine, ensemble voting and LocalStorage helper.
