import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here=dirname(fileURLToPath(import.meta.url));
const root=resolve(here,'..');
const VERSION='0.4.0';

function stripModuleSyntax(source,{core=false}={}){
  let s=source;
  s=s.replace(/^import[\s\S]*?from\s+['"][^'"]+['"];?\s*/m,'');
  s=s.replace(/^import\s+[^;]+;?\s*$/gm,'');
  s=s.replace(/export\s+default\s+AladinAI\s*;?/g,'');
  s=s.replace(/export\s+async\s+function\s+/g,'async function ');
  s=s.replace(/export\s+(?=(class|const|function)\b)/g,'');
  if(core)s=s.replace(/if\(typeof window!=='undefined'\)window\.AladinAI=AladinAI;?/g,'');
  return s.trim();
}

const sourceNames=['aladin-ai.js','advanced.js','extra.js','neural.js','browser.js','retrieval.js','regression-trees.js','vision.js','vision-extra.js','vision-ml.js','github-cloud.js','cloud-integrity.js'];
const parts=[];
for(const [i,name] of sourceNames.entries())parts.push(stripModuleSyntax(await readFile(resolve(root,`src/${name}`),'utf8'),{core:i===0}));

const names=[
  'SeededRandom','Validation','LabelEncoder','OneHotEncoder','LinearRegression','LogisticRegression',
  'RandomForestClassifier','KMeansPlusPlus','ZScoreAnomalyDetector','IQRAnomalyDetector','Pipeline',
  'kFoldIndices','crossValidate','DecisionEngine','SerializableModel','createDefaultRegistry',
  'PolynomialFeatures','PCA','SoftmaxRegression','HierarchicalClustering','RobustScaler','ExponentialMovingAverage',
  'NeuralNetworkClassifier','NeuralNetworkRegressor','schemaFingerprint','ModelPackage','IndexedDBModelStore',
  'processInBatches','getRuntimeCapabilities','CountVectorizer','FeatureHasher','BM25Index','TextSimilarityIndex',
  'NearestNeighborsIndex','reciprocalRankFusion','DecisionTreeRegressor','RandomForestRegressor',
  'validateImage','toGrayscale','imageFromCanvas','imageFromSource','resizeNearest','histogram','convolve','Kernels','sobelEdges',
  'otsuThreshold','threshold','connectedComponents','averageHash','differenceHash','hammingDistance','perceptualSimilarity',
  'templateMatch','colorKMeans','imageDescriptor','compareDescriptors','hogDescriptor','erode','dilate','openMorphology','closeMorphology',
  'structuralSimilarity','edgeDensity','imageMoments','ImageFeatureExtractor','ImageSimilarityIndex','ImageClassifier','ImageRegressor',
  'GitHubCloud','GitHubReleaseAssetResolver','LocalFirstRepository','assertNoEmbeddedGitHubToken','GitHubCloudPatterns',
  'sha256Hex','verifySha256','fetchVerifiedAsset','VerifiedManifestLoader'
];

const banner=`/* AladinAI.js v${VERSION} | offline-first ML/NLP/retrieval/vision engine | generated file: do not edit directly */`;
const fullRegistry='function createFullRegistry(extra={}){return createDefaultRegistry({PolynomialFeatures,PCA,SoftmaxRegression,HierarchicalClustering,RobustScaler,ExponentialMovingAverage,NeuralNetworkClassifier,NeuralNetworkRegressor,CountVectorizer,FeatureHasher,BM25Index,TextSimilarityIndex,NearestNeighborsIndex,DecisionTreeRegressor,RandomForestRegressor,ImageFeatureExtractor,ImageSimilarityIndex,ImageClassifier,ImageRegressor,...extra});}';
const bundle=`${banner}\n(function(global){\n'use strict';\n${parts.join('\n\n')}\n\n${fullRegistry}\nconst AladinAIFull={...AladinAI,version:'${VERSION}',${names.join(',')},createFullRegistry};\nglobal.AladinAI=AladinAIFull;\nif(typeof global.dispatchEvent==='function'&&typeof global.CustomEvent==='function')global.dispatchEvent(new global.CustomEvent('aladinai:ready',{detail:{version:AladinAIFull.version}}));\n})(typeof window!=='undefined'?window:globalThis);\n`;

await writeFile(resolve(root,'dist/aladin-ai.js'),bundle,'utf8');
console.log(`Built dist/aladin-ai.js v${VERSION} (${Buffer.byteLength(bundle)} bytes)`);
