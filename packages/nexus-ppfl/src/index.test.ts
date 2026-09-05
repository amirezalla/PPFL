import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  PrivacyEngine,
  SecureAggregationClient,
  NexusNetworkNode,
  TOPICS,
  WeightSerializer,
  EdgeTrainer,
  AggregatorNode,
  MODEL_SHAPE,
} from './index.js';

test('re-exports resolve to the real classes/values from each underlying package', () => {
  assert.equal(typeof PrivacyEngine.addGaussianNoise, 'function');
  assert.equal(typeof SecureAggregationClient, 'function');
  assert.equal(typeof NexusNetworkNode, 'function');
  assert.equal(TOPICS.WEIGHT_UPDATES, 'nexus-ppfl/weight-updates/1.0.0');
  assert.equal(typeof WeightSerializer, 'function');
  assert.equal(typeof EdgeTrainer, 'function');
  assert.equal(typeof AggregatorNode.FedAvg, 'function');
  assert.equal(MODEL_SHAPE.inputDim, 8);
});

test('re-exported PrivacyEngine actually runs (not just present)', () => {
  const { noised, sigma } = PrivacyEngine.addGaussianNoise(new Float32Array([1, 2, 3]), {
    epsilon: 1,
    delta: 1e-5,
    sensitivity: 1,
  });
  assert.equal(noised.length, 3);
  assert.ok(sigma > 0);
});

test('re-exported AggregatorNode.FedAvg produces the correct average', () => {
  const result = AggregatorNode.FedAvg(new Float32Array([0, 0]), [new Float32Array([2, 4]), new Float32Array([4, 8])]);
  assert.ok(Math.abs(result[0]! - 3) < 1e-6);
  assert.ok(Math.abs(result[1]! - 6) < 1e-6);
});
