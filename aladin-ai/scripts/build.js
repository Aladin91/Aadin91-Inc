import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here=dirname(fileURLToPath(import.meta.url));
const root=resolve(here,'..');

function stripModuleSyntax(source,{core=false}={}){
  let s=source;
  s=s.replace(/^import[\s\S]*?from\s+['"][^'"]+['"];?\s*/m,'');
  s=s.replace(/^import\s+[^;]+;?\s*$/gm,'');
  s=s.replace(/export\s+default\s+AladinAI\s*;?/g,'');
  s=s.replace(/export\s+(?=(class|const|function)\b)/g,'');
  if(core)s=s.replace(/if\(typeof window!=='undefined'\)window\.AladinAI=AladinAI;?/g,'');
  return s.trim();
}

const sourceNames=['aladin-ai.js','advanced.js','extra.js','neural.js','browser.js'];
const parts=[];
for(const [i,name] of sourceNames.entries())parts.push(stripModuleSyntax(await readFile(resolve(root,`src/${name}`),'utf8'),{core:i===0}));

const names=[
  'SeededRandom','Validation','LabelEncoder','OneHotEncoder','LinearRegression','LogisticRegression',
  'RandomForestClassifier','KMeansPlusPlus','ZScoreAnomalyDetector','IQRAnomalyDetector','Pipeline',
  'kFoldIndices','crossValidate','DecisionEngine','SerializableModel','createDefaultRegistry',
  'PolynomialFeatures','PCA','SoftmaxRegression','HierarchicalClustering','RobustScaler','ExponentialMovingAverage',
  'NeuralNetworkClassifier','NeuralNetworkRegressor','schemaFingerprint','ModelPackage','IndexedDBModelStore',
  'processInBatches','getRuntimeCapabilities'
];

const banner='/* AladinAI.js v0.2.0 | offline-first ML/NLP engine | generated file: do not edit directly */';
const fullRegistry='function createFullRegistry(extra={}){return createDefaultRegistry({PolynomialFeatures,PCA,SoftmaxRegression,HierarchicalClustering,RobustScaler,ExponentialMovingAverage,NeuralNetworkClassifier,NeuralNetworkRegressor,...extra});}';
const bundle=`${banner}\n(function(global){\n'use strict';\n${parts.join('\n\n')}\n\n${fullRegistry}\nconst AladinAIFull={...AladinAI,version:ALADIN_AI_ADVANCED_VERSION,${names.join(',')},createFullRegistry};\nglobal.AladinAI=AladinAIFull;\nif(typeof global.dispatchEvent==='function'&&typeof global.CustomEvent==='function')global.dispatchEvent(new global.CustomEvent('aladinai:ready',{detail:{version:AladinAIFull.version}}));\n})(typeof window!=='undefined'?window:globalThis);\n`;

await writeFile(resolve(root,'dist/aladin-ai.js'),bundle,'utf8');
console.log(`Built dist/aladin-ai.js (${Buffer.byteLength(bundle)} bytes)`);
