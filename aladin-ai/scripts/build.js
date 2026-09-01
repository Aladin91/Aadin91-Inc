import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here=dirname(fileURLToPath(import.meta.url));
const root=resolve(here,'..');

function stripModuleSyntax(source,{advanced=false}={}){
  let s=source;
  s=s.replace(/^import\s+\{[\s\S]*?\}\s+from\s+['"].*?['"];?\s*/m,'');
  s=s.replace(/^import\s+[^;]+;?\s*$/gm,'');
  s=s.replace(/export\s+default\s+AladinAI\s*;?/g,'');
  s=s.replace(/export\s+(?=(class|const|function)\b)/g,'');
  if(!advanced){
    s=s.replace(/if\(typeof window!=='undefined'\)window\.AladinAI=AladinAI;?/g,'');
  }
  return s.trim();
}

const core=stripModuleSyntax(await readFile(resolve(root,'src/aladin-ai.js'),'utf8'));
const advanced=stripModuleSyntax(await readFile(resolve(root,'src/advanced.js'),'utf8'),{advanced:true});

const advancedNames=[
  'SeededRandom','Validation','LabelEncoder','OneHotEncoder','LinearRegression','LogisticRegression',
  'RandomForestClassifier','KMeansPlusPlus','ZScoreAnomalyDetector','IQRAnomalyDetector','Pipeline',
  'kFoldIndices','crossValidate','DecisionEngine','SerializableModel','createDefaultRegistry'
];

const banner=`/* AladinAI.js v0.2.0 | offline-first ML/NLP engine | generated file: do not edit directly */`;
const bundle=`${banner}\n(function(global){\n'use strict';\n${core}\n\n${advanced}\n\nconst AladinAIFull={...AladinAI,version:ALADIN_AI_ADVANCED_VERSION,${advancedNames.join(',')}};\nglobal.AladinAI=AladinAIFull;\n})(typeof window!=='undefined'?window:globalThis);\n`;

await writeFile(resolve(root,'dist/aladin-ai.js'),bundle,'utf8');
console.log(`Built dist/aladin-ai.js (${Buffer.byteLength(bundle)} bytes)`);
