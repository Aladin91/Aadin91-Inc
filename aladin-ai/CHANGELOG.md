# Changelog

## 0.4.0

- Added local computer-vision primitives for grayscale conversion, histograms and image resizing.
- Added generic convolution filters and Sobel edge detection.
- Added Otsu thresholding and binary connected-component analysis.
- Added average hash and difference hash perceptual image fingerprints.
- Added perceptual similarity and descriptor-based image comparison.
- Added normalized template matching.
- Added basic RGB K-Means image segmentation.
- Added reusable image descriptors combining intensity, edges and perceptual hashes.
- Added ImageFeatureExtractor, ImageSimilarityIndex, ImageClassifier and ImageRegressor adapters so existing AladinAI models can consume image-derived features.
- Added a standalone browser vision demo; selected images remain local to the browser.
- Added GitHubCloud, release-asset resolver and local-first repository helpers.
- Added protection against accidentally embedding GitHub personal access tokens in browser code.
- Added a versioned cloud manifest and documentation for using GitHub as deployment/versioned asset storage while keeping mutable personal data local-first.
- Added a manual GitHub Pages deployment workflow for the verified framework bundle and demos.
- Expanded the generated browser bundle and CI suite to cover vision and GitHub-cloud helpers.

## 0.3.0

- Added CountVectorizer with configurable n-gram ranges.
- Added bounded-memory FeatureHasher.
- Added BM25 local document retrieval.
- Added TF-IDF cosine similarity index.
- Added numeric nearest-neighbor index.
- Added reciprocal-rank fusion for combining local search results.
- Added Decision Tree and Random Forest regressors.
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
