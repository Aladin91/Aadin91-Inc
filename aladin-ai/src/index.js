export * from './aladin-ai.js';
export * from './advanced.js';

import Base from './aladin-ai.js';
import {
  ALADIN_AI_ADVANCED_VERSION,
  SeededRandom, Validation, LabelEncoder, OneHotEncoder,
  LinearRegression, LogisticRegression, RandomForestClassifier,
  KMeansPlusPlus, ZScoreAnomalyDetector, IQRAnomalyDetector,
  Pipeline, kFoldIndices, crossValidate, DecisionEngine,
  SerializableModel, createDefaultRegistry
} from './advanced.js';

export const AladinAI = {
  ...Base,
  version: ALADIN_AI_ADVANCED_VERSION,
  SeededRandom, Validation, LabelEncoder, OneHotEncoder,
  LinearRegression, LogisticRegression, RandomForestClassifier,
  KMeansPlusPlus, ZScoreAnomalyDetector, IQRAnomalyDetector,
  Pipeline, kFoldIndices, crossValidate, DecisionEngine,
  SerializableModel, createDefaultRegistry
};

export default AladinAI;
