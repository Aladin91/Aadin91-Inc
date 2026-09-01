import AladinAI from '../src/index.js';

function now(){return typeof performance!=='undefined'?performance.now():Date.now();}
function bench(name,fn){const t0=now();const result=fn();const ms=now()-t0;console.log(`${name}: ${ms.toFixed(2)} ms`);return result;}

const docs=Array.from({length:1000},(_,i)=>`document ${i} equipment airflow voltage power controls piping requirement ${(i%20)}`);
const queries=['power voltage','airflow equipment','controls requirement','piping document'];

const bm25=bench('BM25 fit 1,000 docs',()=>new AladinAI.BM25Index().fit(docs));
bench('BM25 4 queries',()=>queries.map(q=>bm25.search(q,{topK:5})));

const tfidf=bench('TF-IDF similarity fit 1,000 docs',()=>new AladinAI.TextSimilarityIndex().fit(docs));
bench('TF-IDF 4 queries',()=>queries.map(q=>tfidf.search(q,{topK:5})));

const X=Array.from({length:500},(_,i)=>[i%13,(i*7)%17,(i*3)%11,(i*5)%19]);
const y=X.map(r=>r[0]+r[1]>14?'high':'low');
const forest=bench('RandomForest fit 500 rows',()=>new AladinAI.RandomForestClassifier({trees:25,maxDepth:6,seed:42}).fit(X,y));
bench('RandomForest predict 500 rows',()=>forest.predict(X));

const km=bench('KMeans++ fit 500 rows',()=>new AladinAI.KMeansPlusPlus({clusters:5,seed:42}).fit(X));
console.log('KMeans inertia:',km.inertia.toFixed(2));
console.log('AladinAI benchmark complete.');
