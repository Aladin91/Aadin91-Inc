export * from './aladin-ai.js';
export * from './advanced.js';
export * from './extra.js';
export * from './neural.js';
export * from './browser.js';
export * from './retrieval.js';

import Base from './aladin-ai.js';
import {
  ALADIN_AI_ADVANCED_VERSION,
  SeededRandom, Validation, LabelEncoder, OneHotEncoder,
  LinearRegression, LogisticRegression, RandomForestClassifier,
  KMeansPlusPlus, ZScoreAnomalyDetector, IQRAnomalyDetector,
  Pipeline, kFoldIndices, crossValidate, DecisionEngine,
  SerializableModel, createDefaultRegistry
} from './advanced.js';
import {
  PolynomialFeatures, PCA, SoftmaxRegression,
  HierarchicalClustering, RobustScaler, ExponentialMovingAverage
} from './extra.js';
import { NeuralNetworkClassifier, NeuralNetworkRegressor } from './neural.js';
import { schemaFingerprint, ModelPackage, IndexedDBModelStore, processInBatches, getRuntimeCapabilities } from './browser.js';
import { CountVectorizer, FeatureHasher, BM25Index, TextSimilarityIndex, NearestNeighborsIndex, reciprocalRankFusion } from './retrieval.js';

export function createFullRegistry(extra={}){
  return createDefaultRegistry({
    PolynomialFeatures,PCA,SoftmaxRegression,HierarchicalClustering,RobustScaler,ExponentialMovingAverage,
    NeuralNetworkClassifier,NeuralNetworkRegressor,
    CountVectorizer,FeatureHasher,BM25Index,TextSimilarityIndex,NearestNeighborsIndex,
    ...extra
  });
}

export const AladinAI = {
  ...Base,
  version: ALADIN_AI_ADVANCED_VERSION,
  SeededRandom, Validation, LabelEncoder, OneHotEncoder,
  LinearRegression, LogisticRegression, RandomForestClassifier,
  KMeansPlusPlus, ZScoreAnomalyDetector, IQRAnomalyDetector,
  Pipeline, kFoldIndices, crossValidate, DecisionEngine,
  SerializableModel, createDefaultRegistry, createFullRegistry,
  PolynomialFeatures, PCA, SoftmaxRegression,
  HierarchicalClustering, RobustScaler, ExponentialMovingAverage,
  NeuralNetworkClassifier, NeuralNetworkRegressor,
  schemaFingerprint, ModelPackage, IndexedDBModelStore, processInBatches, getRuntimeCapabilities,
  CountVectorizer, FeatureHasher, BM25Index, TextSimilarityIndex, NearestNeighborsIndex, reciprocalRankFusion
};

export default AladinAI;
