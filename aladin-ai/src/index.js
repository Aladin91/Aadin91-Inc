export * from './aladin-ai.js';
export * from './advanced.js';
export * from './extra.js';
export * from './neural.js';
export * from './browser.js';
export * from './retrieval.js';
export * from './regression-trees.js';
export * from './vision.js';
export * from './vision-extra.js';
export * from './vision-ml.js';
export * from './github-cloud.js';
export * from './cloud-integrity.js';

import Base from './aladin-ai.js';
import {
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
import { DecisionTreeRegressor, RandomForestRegressor } from './regression-trees.js';
import {
  validateImage,toGrayscale,imageFromCanvas,imageFromSource,resizeNearest,histogram,convolve,Kernels,sobelEdges,
  otsuThreshold,threshold,connectedComponents,averageHash,differenceHash,hammingDistance,perceptualSimilarity,
  templateMatch,colorKMeans,imageDescriptor,compareDescriptors
} from './vision.js';
import { hogDescriptor,erode,dilate,openMorphology,closeMorphology,structuralSimilarity,edgeDensity,imageMoments } from './vision-extra.js';
import { ImageFeatureExtractor, ImageSimilarityIndex, ImageClassifier, ImageRegressor } from './vision-ml.js';
import { GitHubCloud, GitHubReleaseAssetResolver, LocalFirstRepository, assertNoEmbeddedGitHubToken, GitHubCloudPatterns } from './github-cloud.js';
import { sha256Hex,verifySha256,fetchVerifiedAsset,VerifiedManifestLoader } from './cloud-integrity.js';

export const VERSION = '0.4.0';

export function createFullRegistry(extra={}){
  return createDefaultRegistry({
    PolynomialFeatures,PCA,SoftmaxRegression,HierarchicalClustering,RobustScaler,ExponentialMovingAverage,
    NeuralNetworkClassifier,NeuralNetworkRegressor,
    CountVectorizer,FeatureHasher,BM25Index,TextSimilarityIndex,NearestNeighborsIndex,
    DecisionTreeRegressor,RandomForestRegressor,
    ImageFeatureExtractor,ImageSimilarityIndex,ImageClassifier,ImageRegressor,
    ...extra
  });
}

export const AladinAI = {
  ...Base,
  version: VERSION,
  SeededRandom, Validation, LabelEncoder, OneHotEncoder,
  LinearRegression, LogisticRegression, RandomForestClassifier,
  KMeansPlusPlus, ZScoreAnomalyDetector, IQRAnomalyDetector,
  Pipeline, kFoldIndices, crossValidate, DecisionEngine,
  SerializableModel, createDefaultRegistry, createFullRegistry,
  PolynomialFeatures, PCA, SoftmaxRegression,
  HierarchicalClustering, RobustScaler, ExponentialMovingAverage,
  NeuralNetworkClassifier, NeuralNetworkRegressor,
  schemaFingerprint, ModelPackage, IndexedDBModelStore, processInBatches, getRuntimeCapabilities,
  CountVectorizer, FeatureHasher, BM25Index, TextSimilarityIndex, NearestNeighborsIndex, reciprocalRankFusion,
  DecisionTreeRegressor, RandomForestRegressor,
  validateImage,toGrayscale,imageFromCanvas,imageFromSource,resizeNearest,histogram,convolve,Kernels,sobelEdges,
  otsuThreshold,threshold,connectedComponents,averageHash,differenceHash,hammingDistance,perceptualSimilarity,
  templateMatch,colorKMeans,imageDescriptor,compareDescriptors,
  hogDescriptor,erode,dilate,openMorphology,closeMorphology,structuralSimilarity,edgeDensity,imageMoments,
  ImageFeatureExtractor,ImageSimilarityIndex,ImageClassifier,ImageRegressor,
  GitHubCloud,GitHubReleaseAssetResolver,LocalFirstRepository,assertNoEmbeddedGitHubToken,GitHubCloudPatterns,
  sha256Hex,verifySha256,fetchVerifiedAsset,VerifiedManifestLoader
};

export default AladinAI;
