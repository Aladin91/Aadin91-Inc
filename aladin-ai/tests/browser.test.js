import assert from 'node:assert/strict';
import AladinAI, { ModelPackage, schemaFingerprint, createFullRegistry } from '../src/index.js';

assert.equal(AladinAI.version, '0.4.0');
assert.equal(schemaFingerprint({a:1}), schemaFingerprint({a:1}));
assert.notEqual(schemaFingerprint({a:1}), schemaFingerprint({a:2}));

const model = new AladinAI.DecisionTreeClassifier({maxDepth:3}).fit([[1],[2],[8],[9]], ['low','low','high','high']);
const pkg = ModelPackage.create(model, {name:'size-model', featureNames:['size'], targetName:'class', trainingRows:4});
assert.equal(pkg.metadata.engineVersion, '0.4.0');
assert.equal(ModelPackage.validateSchema(pkg, {featureNames:['size'],targetName:'class'}).valid, true);
assert.equal(ModelPackage.validateSchema(pkg, {featureNames:['different'],targetName:'class'}).valid, false);

const restored = ModelPackage.restore(pkg, createFullRegistry());
assert.deepEqual(restored.predict([[8.5]]), ['high']);

console.log('AladinAI browser/package tests passed.');
