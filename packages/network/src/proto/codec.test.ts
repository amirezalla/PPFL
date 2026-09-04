import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  decodeGlobalModel,
  decodeRoundControl,
  decodeWeightUpdate,
  encodeGlobalModel,
  encodeRoundControl,
  encodeWeightUpdate,
} from './codec.js';

test('WeightUpdate round-trips clientId, round, numSamples, and Float32 weights exactly', async () => {
  const weights = new Float32Array([1.5, -2.25, 0, 3.125, -0.001]);
  const encoded = await encodeWeightUpdate({ clientId: 'client-7', round: 3, numSamples: 42, weights });
  const decoded = await decodeWeightUpdate(encoded);

  assert.equal(decoded.clientId, 'client-7');
  assert.equal(decoded.round, 3);
  assert.equal(decoded.numSamples, 42);
  assert.deepEqual(Array.from(decoded.weights), Array.from(weights));
});

test('WeightUpdate round-trips an empty weights vector', async () => {
  const encoded = await encodeWeightUpdate({ clientId: 'empty', round: 0, numSamples: 0, weights: new Float32Array(0) });
  const decoded = await decodeWeightUpdate(encoded);
  assert.equal(decoded.weights.length, 0);
});

test('GlobalModel round-trips round, weights, and modelHash', async () => {
  const weights = new Float32Array([10, 20, 30]);
  const encoded = await encodeGlobalModel({ round: 5, weights, modelHash: 'abc123' });
  const decoded = await decodeGlobalModel(encoded);

  assert.equal(decoded.round, 5);
  assert.equal(decoded.modelHash, 'abc123');
  assert.deepEqual(Array.from(decoded.weights), Array.from(weights));
});

test('RoundControl round-trips every phase value', async () => {
  for (const phase of ['UNKNOWN', 'ROUND_START', 'ROUND_CLOSED'] as const) {
    const encoded = await encodeRoundControl({ phase, round: 1, expectedClients: 10, completionThreshold: 0.8 });
    const decoded = await decodeRoundControl(encoded);
    assert.equal(decoded.phase, phase);
    assert.equal(decoded.round, 1);
    assert.equal(decoded.expectedClients, 10);
    assert.ok(Math.abs(decoded.completionThreshold - 0.8) < 1e-9);
  }
});
