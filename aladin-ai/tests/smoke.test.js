import assert from 'node:assert/strict';
import { Tokenizer, DecisionTreeClassifier, KMeans, MultinomialNaiveBayesText, Metrics } from '../src/aladin-ai.js';

const tokenizer = new Tokenizer({removeStopWords:true});
assert.deepEqual(tokenizer.tokenize('The pump requires 480V power.'), ['pump','requires','480v','power']);

const tree = new DecisionTreeClassifier({maxDepth:4}).fit([[1],[2],[8],[9]], ['low','low','high','high']);
assert.equal(tree.predict([[8.5]])[0], 'high');

const km = new KMeans({clusters:2}).fit([[1,1],[1.1,1],[9,9],[9.2,9]]);
assert.equal(new Set(km.labels).size, 2);

const nb = new MultinomialNaiveBayesText();
nb.train('power voltage disconnect','electrical').train('duct airflow cfm','hvac').fit();
assert.equal(nb.predict('voltage and power'), 'electrical');

assert.equal(Metrics.accuracy(['a','b'],['a','b']),1);
console.log('AladinAI smoke tests passed.');
